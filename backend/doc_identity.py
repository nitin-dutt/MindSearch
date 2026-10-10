"""Deterministic document and chunk identity (alignment_plan INF-03).

A document's id is a hash of its content AND its access rule, and a chunk's Qdrant point id is derived from the
document id and the chunk's position. So ingesting the same file with the same access rule twice (a re-upload, or
an automatic task retry after a partial failure) writes the SAME ids and overwrites instead of duplicating.
The same file with a DIFFERENT access rule is a different document, so permissions never get mixed up.
"""
import hashlib
import json
import uuid

_NAMESPACE = uuid.UUID("5d0f1b0e-6a7e-4a55-9c1e-2f6f0a8e9c11")


def compute_doc_id(file_path: str, acl) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    h.update(b"\0" + json.dumps(acl.to_payload(), sort_keys=True).encode())
    return h.hexdigest()[:32]


def point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{chunk_index}"))
