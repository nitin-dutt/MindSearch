from sentence_transformers import SentenceTransformer
from qdrant_client.models import (VectorParams, Distance, PointStruct, PayloadSchemaType, Filter,
                                  FieldCondition, MatchValue, FilterSelector, SparseVectorParams, SparseVector,
                                  Modifier, Prefetch, FusionQuery, Fusion, HasIdCondition)
import atexit
import uuid

from settings import get_settings

from doc_identity import point_id as make_point_id
from encryptor import record_aad
from sparse import SparseEncoder

EMBEDDING_MODEL_NAME = get_settings().embedding_model     # EMBEDDING_MODEL (settings.py)
PAYLOAD_SCHEMA_VERSION = 5  # v5: named dense + keyword (sparse) vectors for hybrid search; v4: deterministic point ids + doc_id/chunk_index; v3: ACL + ciphertext bound to point id and ACL
DENSE = "dense"          # name of the embedding vector
KEYWORDS = "keywords"    # name of the BM25-style sparse vector (sparse.py)
HYBRID_POOL = 50         # candidates each branch (dense, keywords) contributes before fusion

model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
# Taken from the model itself, so switching EMBEDDING_MODEL needs no code change (the collection must be re-indexed).
EMBEDDING_DIM = (getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension)()

# One embedded-Qdrant client per storage path. Local mode takes an exclusive file lock,
# so a path must only ever be opened once per process.
_qdrant_clients = {}


def is_server(location: str) -> bool:
    return str(location).startswith(("http://", "https://"))


def get_qdrant_client(path: str = "./qdrant_db"):
    """`path` is either a folder (embedded Qdrant, used by tests and the evaluation harness) or the URL of a
    Qdrant server (used by the application, so the API and the worker can share one database)."""
    if path not in _qdrant_clients:
        from qdrant_client import QdrantClient
        _qdrant_clients[path] = QdrantClient(url=path, timeout=30) if is_server(path) else QdrantClient(path=path)
    return _qdrant_clients[path]


def close_all_qdrant_clients():
    for path in list(_qdrant_clients):
        close_qdrant_client(path)


atexit.register(close_all_qdrant_clients)  # avoids a noisy local-client __del__ error at interpreter exit


def _defaults(db_path, collection_name):
    """Callers normally pass both explicitly (from PipelineConfig); otherwise the central settings decide."""
    cfg = get_settings()
    return db_path or cfg.qdrant_location, collection_name or cfg.qdrant_collection


class IndexSchemaError(RuntimeError):
    pass


_checked = set()


def _check_schema(client, path, collection_name):
    """An index built before hybrid search has one unnamed vector and no keyword vector. Fail with a clear instruction."""
    key = (path, collection_name)
    if key in _checked:
        return
    info = client.get_collection(collection_name)
    sparse = info.config.params.sparse_vectors or {}
    dense = info.config.params.vectors
    if KEYWORDS not in sparse or not isinstance(dense, dict) or DENSE not in dense:
        raise IndexSchemaError(f"Collection '{collection_name}' was created by an older version (no keyword vectors). "
                               "Re-index: python reset_qdrant.py, then upload the documents again.")
    _checked.add(key)


def close_qdrant_client(path: str):
    """Release the file lock for a path (used before rebuilding an evaluation index)."""
    client = _qdrant_clients.pop(path, None)
    if client is not None:
        client.close()


def build_embeddings(plaintext_chunks, encryptor, acl, db_path=None, collection_name=None, doc_id=None, pages=None):
    """Embed chunks and store them encrypted. `acl` (access_policy.DocACL) is stored on every point and
    enforced at query time; each ciphertext is bound to its point id and ACL so it cannot be moved or
    have its permissions edited without failing decryption.

    With a `doc_id`, point ids are deterministic (doc_id + chunk position), so re-ingesting the same document
    overwrites its points instead of duplicating them (safe retries, safe re-uploads).

    `pages` (optional) is [(first page, last page)] per chunk for PDFs; it is stored so answers can cite a page."""
    db_path, collection_name = _defaults(db_path, collection_name)
    acl_payload = acl.to_payload()
    client = get_qdrant_client(db_path)

    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={DENSE: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)},
            # Modifier.IDF: Qdrant weights rare words higher at query time (the other half of BM25)
            sparse_vectors_config={KEYWORDS: SparseVectorParams(modifier=Modifier.IDF)},
        )
        if is_server(db_path):
            # Indexes on the fields the access filter uses (they have no effect in embedded mode)
            client.create_payload_index(collection_name, "min_rank", PayloadSchemaType.INTEGER)
            client.create_payload_index(collection_name, "groups", PayloadSchemaType.KEYWORD)
            client.create_payload_index(collection_name, "users", PayloadSchemaType.KEYWORD)
            client.create_payload_index(collection_name, "doc_id", PayloadSchemaType.KEYWORD)

    _check_schema(client, db_path, collection_name)
    embeddings = model.encode(plaintext_chunks, convert_to_numpy=True)
    sparse = SparseEncoder(encryptor.subkey("keyword-terms"))

    points = []
    for index, (chunk, embedding) in enumerate(zip(plaintext_chunks, embeddings)):
        term_ids, term_weights = sparse.document(chunk)
        point_id = make_point_id(doc_id, index) if doc_id else str(uuid.uuid4())
        encrypted = encryptor.encrypt(chunk, aad=record_aad(point_id, acl_payload))
        points.append(
            PointStruct(
                id=point_id,
                vector={DENSE: embedding.tolist(), KEYWORDS: SparseVector(indices=term_ids, values=term_weights)},
                # Qdrant payloads must be JSON serializable, so the bytes are stored as hex
                payload={"chunk_data": encrypted.hex(), **acl_payload,
                         **({"doc_id": doc_id, "chunk_index": index} if doc_id else {}),
                         **_page_payload(pages, index)}
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )
    return True


def _page_payload(pages, index):
    """{"page": first page, "page_end": last page} when the chunk has pages (PDF), else nothing."""
    if not pages or pages[index][0] is None:
        return {}
    first, last = pages[index]
    return {"page": first, **({"page_end": last} if last != first else {})}


def search_embeddings(query, user, policy, db_path=None, collection_name=None, k=5, hybrid=False, encryptor=None,
                      hybrid_pool=HYBRID_POOL):
    """Search restricted to what `user` may read. There is deliberately no default user.
    Dense by default; with hybrid=True the dense and keyword rankings are fused (RRF) and `encryptor` is needed to
    build the keyed keyword query. The access filter is applied to BOTH branches.
    Returns (scores, encrypted payloads, associated data for decryption, source metadata), best first.
    Source metadata is {doc_id, chunk_index} per hit, so answers can point back to their document."""
    db_path, collection_name = _defaults(db_path, collection_name)
    client = get_qdrant_client(db_path)

    if not client.collection_exists(collection_name=collection_name):
        return [], [], [], []

    _check_schema(client, db_path, collection_name)
    q_vector = model.encode([query], convert_to_numpy=True)[0]
    query_filter = policy.qdrant_filter(user)  # None only for super roles

    term_ids, term_weights = ([], [])
    if hybrid:
        if encryptor is None:
            raise ValueError("hybrid search needs the encryptor (it keys the keyword index)")
        term_ids, term_weights = SparseEncoder(encryptor.subkey("keyword-terms")).query(query)
    if hybrid and term_ids:
        # The filter must be on each branch: it is not applied to candidates that prefetch has already chosen.
        results = client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(query=q_vector.tolist(), using=DENSE, limit=hybrid_pool, filter=query_filter),
                Prefetch(query=SparseVector(indices=term_ids, values=term_weights), using=KEYWORDS, limit=hybrid_pool,
                         filter=query_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF), limit=k, query_filter=query_filter,
        )
    else:                                   # dense only (also the fallback when the question has no usable keywords)
        results = client.query_points(
            collection_name=collection_name,
            query=q_vector.tolist(),
            using=DENSE,
            limit=k,
            query_filter=query_filter
        )

    scores, payloads, aads, metas = [], [], [], []
    # Qdrant client.query_points returns a QueryResponse which contains .points
    points = results.points if hasattr(results, 'points') else results
    for hit in points:
        scores.append(hit.score)
        # Reconstruct bytes from hex string
        hex_data = hit.payload.get("chunk_data", "")
        payloads.append(bytes.fromhex(hex_data) if hex_data else b'')
        aads.append(record_aad(hit.id, hit.payload))
        metas.append({"doc_id": hit.payload.get("doc_id"), "chunk_index": hit.payload.get("chunk_index"), "page": hit.payload.get("page")})

    return scores, payloads, aads, metas


def fetch_chunks(refs, user, policy, db_path=None, collection_name=None):
    """Load specific passages by (doc_id, chunk_index) - e.g. the ones a graph hit points to - with the SAME access filter
    as a search, so a reference the user may not read simply returns nothing.
    Returns (encrypted payloads, associated data, source metadata) for the passages that exist and are readable."""
    db_path, collection_name = _defaults(db_path, collection_name)
    client = get_qdrant_client(db_path)
    if not refs or not client.collection_exists(collection_name=collection_name):
        return [], [], []
    _check_schema(client, db_path, collection_name)
    ids = [make_point_id(doc_id, index) for doc_id, index in refs]
    access = policy.qdrant_filter(user)
    flt = Filter(must=[HasIdCondition(has_id=ids)] + ([access] if access is not None else []))
    points, _ = client.scroll(collection_name, scroll_filter=flt, limit=len(ids), with_payload=True, with_vectors=False)
    payloads, aads, metas = [], [], []
    for hit in points:
        hex_data = hit.payload.get("chunk_data", "")
        payloads.append(bytes.fromhex(hex_data) if hex_data else b"")
        aads.append(record_aad(hit.id, hit.payload))
        metas.append({"doc_id": hit.payload.get("doc_id"), "chunk_index": hit.payload.get("chunk_index"), "page": hit.payload.get("page")})
    return payloads, aads, metas


def count_document_chunks(doc_id, db_path=None, collection_name=None):
    db_path, collection_name = _defaults(db_path, collection_name)
    client = get_qdrant_client(db_path)
    if not client.collection_exists(collection_name=collection_name):
        return 0
    flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    return client.count(collection_name, count_filter=flt, exact=True).count


def delete_document_chunks(doc_id, db_path=None, collection_name=None):
    """Remove every chunk of one document. Returns how many were deleted. Idempotent."""
    db_path, collection_name = _defaults(db_path, collection_name)
    client = get_qdrant_client(db_path)
    n = count_document_chunks(doc_id, db_path, collection_name)
    if n:
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        client.delete(collection_name, points_selector=FilterSelector(filter=flt), wait=True)
    return n
