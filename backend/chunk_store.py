"""Where the passages (chunks) of every document live: embedded, encrypted, and only ever handed back to someone allowed to read them.

    store = ChunkStore(location, collection, encryptor, policy)
    store.add(chunks, acl, doc_id=...)                 # embed + encrypt + index
    store.search(query, user, k=20, hybrid=True)       # -> passages `user` may read, best first
    store.fetch([(doc_id, chunk_index), ...], user)    # -> those passages, if `user` may read them
    store.count(doc_id) / store.delete(doc_id)

A passage is a dict: {"chunk": text, "score": float, "doc_id", "chunk_index", "page"[, "ocr": True]}.

The three security rules of stored text are enforced HERE and nowhere else, so no caller can forget one:
  1. Every read carries the access filter of the user it is for (policy.qdrant_filter). There is deliberately no default user and no
     unfiltered read. Narrowing a search to chosen documents can only remove results: the access filter always stays.
  2. Text is stored encrypted, and each ciphertext is bound to its record id and access rule (encryptor.record_aad), so it cannot be
     moved to another record or have its permissions edited in the database without failing decryption.
  3. A passage that fails that integrity check is dropped, logged and counted. It is never served and never replaced by empty text.
"""
import logging
import uuid

from qdrant_client.models import (Distance, FieldCondition, Filter, FilterSelector, Fusion, FusionQuery, HasIdCondition, MatchAny,
                                  MatchValue, Modifier, PayloadSchemaType, PointStruct, Prefetch, SparseVector, SparseVectorParams,
                                  VectorParams)

import embedder
from doc_identity import point_id as make_point_id
from encryptor import record_aad
from metrics import DECRYPT_FAILURES, timed
from sparse import SparseEncoder
from vector_db import get_qdrant_client, is_server

log = logging.getLogger("mindsearch.security")

PAYLOAD_SCHEMA_VERSION = 5  # v5: named dense + keyword (sparse) vectors for hybrid search; v4: deterministic point ids + doc_id/chunk_index; v3: ACL + ciphertext bound to point id and ACL
DENSE = "dense"          # name of the embedding vector
KEYWORDS = "keywords"    # name of the BM25-style sparse vector (sparse.py)
HYBRID_POOL = 50         # candidates each branch (dense, keywords) contributes before fusion


class IndexSchemaError(RuntimeError):
    pass


class ChunkStore:
    def __init__(self, location, collection, encryptor, policy):
        """`location` is a folder or a Qdrant server URL (vector_db.py). Nothing is opened until the store is first used."""
        self.location, self.collection, self.encryptor, self.policy = location, collection, encryptor, policy
        self._schema_ok = False
        self._sparse = None

    # ------------------------------------------------------------------ internals

    @property
    def _client(self):
        return get_qdrant_client(self.location)

    @property
    def _keywords(self) -> SparseEncoder:
        """The keyword index is keyed (word ids are HMACs under a key derived from the data key), so the database never sees words."""
        if self._sparse is None:
            self._sparse = SparseEncoder(self.encryptor.subkey("keyword-terms"))
        return self._sparse

    def _exists(self) -> bool:
        """True if there is anything to read. Also checks, once, that the index was built by this version."""
        if not self._client.collection_exists(collection_name=self.collection):
            return False
        self._check_schema()
        return True

    def _check_schema(self):
        """An index built before hybrid search has one unnamed vector and no keyword vector. Fail with a clear instruction."""
        if self._schema_ok:
            return
        params = self._client.get_collection(self.collection).config.params
        if KEYWORDS not in (params.sparse_vectors or {}) or not isinstance(params.vectors, dict) or DENSE not in params.vectors:
            raise IndexSchemaError(f"Collection '{self.collection}' was created by an older version (no keyword vectors). "
                                   "Re-index: python reset_qdrant.py, then upload the documents again.")
        self._schema_ok = True

    def _create(self):
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config={DENSE: VectorParams(size=embedder.EMBEDDING_DIM, distance=Distance.COSINE)},
            # Modifier.IDF: Qdrant weights rare words higher at query time (the other half of BM25)
            sparse_vectors_config={KEYWORDS: SparseVectorParams(modifier=Modifier.IDF)},
        )
        if is_server(self.location):
            # Indexes on the fields the access filter uses (they have no effect in embedded mode)
            for name, kind in (("min_rank", PayloadSchemaType.INTEGER), ("groups", PayloadSchemaType.KEYWORD),
                               ("users", PayloadSchemaType.KEYWORD), ("doc_id", PayloadSchemaType.KEYWORD)):
                self._client.create_payload_index(self.collection, name, kind)

    def _access(self, user, *extra):
        """The filter for a read on behalf of `user`: their access rule AND every extra condition. The access rule is None only for a
        super role (who may read everything); extra conditions can only narrow."""
        access = self.policy.qdrant_filter(user)
        must = list(extra) + ([access] if access is not None else [])
        return Filter(must=must) if must else None

    def _open(self, hits, scores=None):
        """Decrypt stored points into passages, dropping any that fail the integrity check (rule 3)."""
        passages = []
        with timed("decrypt"):
            for i, hit in enumerate(hits):
                payload = hit.payload
                try:
                    text = self.encryptor.decrypt(bytes.fromhex(payload.get("chunk_data", "")), record_aad(hit.id, payload))
                except Exception:  # noqa: BLE001 - InvalidTag, short payload, bad hex, bad utf-8: all mean "do not serve"
                    log.error("SECURITY: a stored chunk failed its integrity check and was skipped (possible tampering)")
                    DECRYPT_FAILURES.inc()
                    continue
                passages.append({"chunk": text, "score": scores[i] if scores is not None else 0.0,
                                 "doc_id": payload.get("doc_id"), "chunk_index": payload.get("chunk_index"),
                                 "page": payload.get("page"), **({"ocr": True} if payload.get("ocr") else {})})
        return passages

    # ------------------------------------------------------------------ writing

    def add(self, chunks, acl, doc_id=None, pages=None, ocr_flags=None) -> int:
        """Embed `chunks` (plain text) and store them encrypted under `acl` (access_policy.DocACL). Returns how many were stored.

        With a `doc_id`, record ids are deterministic (doc_id + position), so adding the same document again overwrites its records
        instead of duplicating them (safe retries, safe re-uploads).
        `pages` (optional) is [(first page, last page)] per chunk for PDFs; it is stored so answers can cite a page.
        `ocr_flags` (optional) is one bool per chunk: True when the text was read from a scanned page or a picture (RET-06)."""
        if not chunks:
            return 0
        acl_payload = acl.to_payload()
        if not self._client.collection_exists(collection_name=self.collection):
            self._create()
        self._check_schema()
        vectors = embedder.model.encode(chunks, convert_to_numpy=True)
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            term_ids, term_weights = self._keywords.document(chunk)
            point_id = make_point_id(doc_id, index) if doc_id else str(uuid.uuid4())
            encrypted = self.encryptor.encrypt(chunk, aad=record_aad(point_id, acl_payload))
            points.append(PointStruct(
                id=point_id,
                vector={DENSE: vector.tolist(), KEYWORDS: SparseVector(indices=term_ids, values=term_weights)},
                # Qdrant payloads must be JSON serializable, so the bytes are stored as hex
                payload={"chunk_data": encrypted.hex(), **acl_payload,
                         **({"doc_id": doc_id, "chunk_index": index} if doc_id else {}),
                         **_page_payload(pages, index),
                         **({"ocr": True} if ocr_flags and ocr_flags[index] else {})}))
        self._client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def delete(self, doc_id) -> int:
        """Remove every passage of one document. Returns how many were deleted. Idempotent."""
        n = self.count(doc_id)
        if n:
            self._client.delete(self.collection, points_selector=FilterSelector(filter=_of_document(doc_id)), wait=True)
        return n

    # ------------------------------------------------------------------ reading

    def search(self, query, user, k=5, hybrid=False, hybrid_pool=HYBRID_POOL, doc_ids=None) -> list:
        """The `k` passages most similar to `query` among those `user` may read (and, with `doc_ids`, among those documents), best
        first. Dense (embedding) search by default; with hybrid=True the dense and the keyword rankings are fused (RRF)."""
        if not self._exists():
            return []
        with timed("embed_query"):
            vector = embedder.model.encode([query], convert_to_numpy=True)[0].tolist()
        scope = [FieldCondition(key="doc_id", match=MatchAny(any=list(doc_ids)))] if doc_ids else []
        flt = self._access(user, *scope)
        term_ids, term_weights = self._keywords.query(query) if hybrid else ([], [])
        with timed("vector_search"):
            if term_ids:
                # The filter must be on each branch: it is not applied to candidates that prefetch has already chosen.
                found = self._client.query_points(
                    collection_name=self.collection,
                    prefetch=[Prefetch(query=vector, using=DENSE, limit=hybrid_pool, filter=flt),
                              Prefetch(query=SparseVector(indices=term_ids, values=term_weights), using=KEYWORDS, limit=hybrid_pool,
                                       filter=flt)],
                    query=FusionQuery(fusion=Fusion.RRF), limit=k, query_filter=flt)
            else:                           # dense only (also the fallback when the question has no usable keywords)
                found = self._client.query_points(collection_name=self.collection, query=vector, using=DENSE, limit=k,
                                                  query_filter=flt)
        return self._open(found.points, [hit.score for hit in found.points])

    def fetch(self, refs, user) -> list:
        """The passages at these (doc_id, chunk_index) positions - e.g. the ones a graph hit or a citation points to - that exist
        and that `user` may read. A position the user may not read simply is not in the answer. Order is not guaranteed."""
        if not refs or not self._exists():
            return []
        ids = [make_point_id(doc_id, index) for doc_id, index in refs]
        points, _ = self._client.scroll(self.collection, scroll_filter=self._access(user, HasIdCondition(has_id=ids)),
                                        limit=len(ids), with_payload=True, with_vectors=False)
        return self._open(points)

    def count(self, doc_id) -> int:
        """How many passages of one document are stored (a number only: no text, so no user is needed)."""
        if not self._client.collection_exists(collection_name=self.collection):
            return 0
        return self._client.count(self.collection, count_filter=_of_document(doc_id), exact=True).count


def _of_document(doc_id):
    return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])


def _page_payload(pages, index):
    """{"page": first page, "page_end": last page} when the chunk has pages (PDF), else nothing."""
    if not pages or pages[index][0] is None:
        return {}
    first, last = pages[index]
    return {"page": first, **({"page_end": last} if last != first else {})}
