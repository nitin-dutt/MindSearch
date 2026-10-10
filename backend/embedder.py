from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
import atexit
import uuid

from doc_identity import point_id as make_point_id
from encryptor import record_aad

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension
PAYLOAD_SCHEMA_VERSION = 4  # v4: deterministic point ids + doc_id/chunk_index; v3: ACL + ciphertext bound to point id and ACL

model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

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


def close_qdrant_client(path: str):
    """Release the file lock for a path (used before rebuilding an evaluation index)."""
    client = _qdrant_clients.pop(path, None)
    if client is not None:
        client.close()


def build_embeddings(plaintext_chunks, encryptor, acl, db_path="./qdrant_db", collection_name="mindsearch_docs", doc_id=None):
    """Embed chunks and store them encrypted. `acl` (access_policy.DocACL) is stored on every point and
    enforced at query time; each ciphertext is bound to its point id and ACL so it cannot be moved or
    have its permissions edited without failing decryption.

    With a `doc_id`, point ids are deterministic (doc_id + chunk position), so re-ingesting the same document
    overwrites its points instead of duplicating them (safe retries, safe re-uploads)."""
    acl_payload = acl.to_payload()
    client = get_qdrant_client(db_path)

    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if is_server(db_path):
            # Indexes on the fields the access filter uses (they have no effect in embedded mode)
            client.create_payload_index(collection_name, "min_rank", PayloadSchemaType.INTEGER)
            client.create_payload_index(collection_name, "groups", PayloadSchemaType.KEYWORD)
            client.create_payload_index(collection_name, "users", PayloadSchemaType.KEYWORD)

    embeddings = model.encode(plaintext_chunks, convert_to_numpy=True)

    points = []
    for index, (chunk, embedding) in enumerate(zip(plaintext_chunks, embeddings)):
        point_id = make_point_id(doc_id, index) if doc_id else str(uuid.uuid4())
        encrypted = encryptor.encrypt(chunk, aad=record_aad(point_id, acl_payload))
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                # Qdrant payloads must be JSON serializable, so the bytes are stored as hex
                payload={"chunk_data": encrypted.hex(), **acl_payload,
                         **({"doc_id": doc_id, "chunk_index": index} if doc_id else {})}
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )
    return True


def search_embeddings(query, user, policy, db_path="./qdrant_db", collection_name="mindsearch_docs", k=5):
    """Dense search restricted to what `user` may read. There is deliberately no default user.
    Returns (scores, encrypted payloads, associated data for decryption), best first."""
    client = get_qdrant_client(db_path)

    if not client.collection_exists(collection_name=collection_name):
        return [], [], []

    q_vector = model.encode([query], convert_to_numpy=True)[0]
    query_filter = policy.qdrant_filter(user)  # None only for super roles

    results = client.query_points(
        collection_name=collection_name,
        query=q_vector.tolist(),
        limit=k,
        query_filter=query_filter
    )

    scores, payloads, aads = [], [], []
    # Qdrant client.query_points returns a QueryResponse which contains .points
    points = results.points if hasattr(results, 'points') else results
    for hit in points:
        scores.append(hit.score)
        # Reconstruct bytes from hex string
        hex_data = hit.payload.get("chunk_data", "")
        payloads.append(bytes.fromhex(hex_data) if hex_data else b'')
        aads.append(record_aad(hit.id, hit.payload))

    return scores, payloads, aads
