"""Document registry (alignment_plan INF-04, INF-05): one entry per uploaded document.

Why it exists: users need to see what is in the knowledge base, see whether an upload has finished, delete a document,
and (later) show where an answer came from. Chunks already carry `doc_id`; this registry holds the document-level facts.

Storage: its own Qdrant collection ("documents"), one point per document, so listing reuses the exact access filter
(`AccessPolicy.qdrant_filter`) that is tested exhaustively for chunks. A document is listed only to people who may read it.

Privacy: the filename, uploader and size are sensitive ("layoffs_2027.pdf"), so they are stored encrypted, bound to the
document id and access rule (same record-bound AES-GCM as chunk text). Status, timestamps and the access rule stay in
plaintext because they are needed for filtering and display.
"""
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from qdrant_client.models import (Distance, FieldCondition, Filter, FilterSelector, MatchValue, PayloadSchemaType,
                                  PointStruct, VectorParams)

from access_policy import DocACL
from doc_identity import _NAMESPACE
from embedder import get_qdrant_client, is_server
from encryptor import record_aad

log = logging.getLogger("mindsearch.security")

COLLECTION = "documents"
STATUSES = ("queued", "processing", "done", "failed")
UNREADABLE = "(unreadable)"


def registry_point_id(doc_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"registry:{doc_id}"))


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DocumentEntry:
    doc_id: str
    status: str
    chunks: int
    created_at: str
    updated_at: str
    error: Optional[str]
    acl: DocACL
    filename: str = UNREADABLE
    uploader: str = UNREADABLE
    size: int = 0

    def to_public(self, policy, include_uploader: bool) -> dict:
        level = policy.level_name(self.acl.min_rank)
        out = {"doc_id": self.doc_id, "filename": self.filename, "status": self.status, "chunks": self.chunks,
               "size": self.size, "uploaded_at": self.created_at, "updated_at": self.updated_at,
               "level": level, "groups": list(self.acl.groups), "error": self.error}
        if include_uploader:
            out["uploaded_by"] = self.uploader
        return out


class DocumentRegistry:
    def __init__(self, location: str, encryptor, policy, collection: str = COLLECTION):
        self.location, self.encryptor, self.policy, self.collection = location, encryptor, policy, collection
        self._ready = False

    # ------------------------------------------------------------------ internals
    def _client(self):
        client = get_qdrant_client(self.location)
        if not self._ready:
            if not client.collection_exists(self.collection):
                client.create_collection(self.collection, vectors_config=VectorParams(size=1, distance=Distance.COSINE))
                if is_server(self.location):
                    for field_name, schema in (("min_rank", PayloadSchemaType.INTEGER), ("groups", PayloadSchemaType.KEYWORD),
                                               ("users", PayloadSchemaType.KEYWORD), ("doc_id", PayloadSchemaType.KEYWORD),
                                               ("status", PayloadSchemaType.KEYWORD)):
                        client.create_payload_index(self.collection, field_name, schema)
            self._ready = True
        return client

    def _entry(self, point) -> DocumentEntry:
        p = point.payload
        acl = DocACL(min_rank=int(p["min_rank"]), groups=tuple(p.get("groups", ())), users=tuple(p.get("users", ())))
        entry = DocumentEntry(doc_id=p["doc_id"], status=p["status"], chunks=int(p.get("chunks", 0)),
                              created_at=p["created_at"], updated_at=p["updated_at"], error=p.get("error"), acl=acl)
        try:
            meta = json.loads(self.encryptor.decrypt(bytes.fromhex(p["meta_enc"]), record_aad(point.id, acl.to_payload())))
            entry.filename, entry.uploader, entry.size = meta["filename"], meta["uploader"], int(meta.get("size", 0))
        except Exception:  # noqa: BLE001 - tampered or wrong key: show the entry, hide the metadata, log it
            log.error("SECURITY: a document registry entry failed its integrity check (possible tampering)")
        return entry

    # ------------------------------------------------------------------ writes
    def register(self, doc_id: str, acl: DocACL, filename: str, uploader: str, size: int) -> DocumentEntry:
        """Create (or reset) the entry for a document with status 'queued'."""
        client = self._client()
        pid = registry_point_id(doc_id)
        meta = json.dumps({"filename": filename, "uploader": uploader, "size": size})
        now = _now()
        payload = {"doc_id": doc_id, "status": "queued", "chunks": 0, "created_at": now, "updated_at": now, "error": None,
                   **acl.to_payload(),
                   "meta_enc": self.encryptor.encrypt(meta, aad=record_aad(pid, acl.to_payload())).hex()}
        client.upsert(self.collection, [PointStruct(id=pid, vector=[0.0], payload=payload)], wait=True)
        return self.get(doc_id)

    def set_status(self, doc_id: str, status: str, chunks: Optional[int] = None, error: Optional[str] = None):
        assert status in STATUSES, status
        payload = {"status": status, "updated_at": _now(), "error": (error or None) if status == "failed" else None}
        if chunks is not None:
            payload["chunks"] = chunks
        self._client().set_payload(self.collection, payload=payload, points=[registry_point_id(doc_id)], wait=True)

    def delete(self, doc_id: str):
        self._client().delete(self.collection, points_selector=[registry_point_id(doc_id)], wait=True)

    # ------------------------------------------------------------------ reads
    def get(self, doc_id: str, user=None) -> Optional[DocumentEntry]:
        """The entry, or None. With `user`, None also means 'not visible to this user' (so existence is not leaked)."""
        client = self._client()
        must = [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        access = self.policy.qdrant_filter(user) if user is not None else None
        flt = Filter(must=must + ([access] if access is not None else []))
        points, _ = client.scroll(self.collection, scroll_filter=flt, limit=1, with_payload=True)
        return self._entry(points[0]) if points else None

    def list(self, user, limit: int = 1000) -> List[DocumentEntry]:
        """Documents this user may read, newest first."""
        client = self._client()
        entries, offset = [], None
        flt = self.policy.qdrant_filter(user)
        while len(entries) < limit:
            points, offset = client.scroll(self.collection, scroll_filter=flt, limit=200, offset=offset, with_payload=True)
            entries += [self._entry(p) for p in points]
            if offset is None:
                break
        return sorted(entries, key=lambda e: e.created_at, reverse=True)[:limit]

    def filenames(self, doc_ids) -> Dict[str, str]:
        """doc_id -> filename for documents that were just retrieved (already authorized by the chunk filter)."""
        ids = {d: registry_point_id(d) for d in set(doc_ids)}
        if not ids:
            return {}
        points = self._client().retrieve(self.collection, ids=list(ids.values()), with_payload=True)
        by_pid = {str(p.id): self._entry(p) for p in points}
        return {d: by_pid[pid].filename for d, pid in ids.items() if pid in by_pid}
