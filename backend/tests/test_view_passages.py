"""'View source': reading a document's indexed passages (GET /v1/documents/{id}/passages).
API rules with the fake pipeline, then the real pipeline for order and access control."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_api import doc_id_of, env, upload  # noqa: E402,F401  (env is a fixture)


def test_someone_who_may_read_the_document_gets_its_passages_with_the_filename_and_total(env):  # noqa: F811
    staff = env.login("staff")
    d = doc_id_of(upload(env, staff, name="handbook.txt", data=b"some text"))
    env.registry.set_status(d, "done", chunks=3)
    r = env.client.get(f"/v1/documents/{d}/passages?start=1&limit=5", headers=staff)
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == d and body["filename"] == "handbook.txt" and body["total"] == 3 and body["start"] == 1
    assert [p["chunk_index"] for p in body["passages"]] == [1, 2] and body["passages"][0]["page"] == 2
    assert env.rag.viewed == [(d, "staff", 1, 5)]                           # read as the caller, never as someone else


def test_a_document_the_user_cannot_read_looks_like_it_does_not_exist(env):  # noqa: F811
    d = doc_id_of(upload(env, env.login("root"), data=b"board memo", level="Manager"))
    staff = env.login("staff")
    assert env.client.get(f"/v1/documents/{d}/passages", headers=staff).status_code == 404
    assert env.client.get(f"/v1/documents/{'0' * 32}/passages", headers=staff).status_code == 404   # same answer as a real one
    assert getattr(env.rag, "viewed", []) == []                             # the pipeline was never asked
    assert env.client.get(f"/v1/documents/{d}/passages").status_code == 401


@pytest.mark.parametrize("query", ["start=-1", "limit=0", "limit=21", "start=abc", "limit=5000"])
def test_out_of_range_windows_are_rejected(env, query):  # noqa: F811
    staff = env.login("staff")
    d = doc_id_of(upload(env, staff, data=b"some text"))
    assert env.client.get(f"/v1/documents/{d}/passages?{query}", headers=staff).status_code == 422


def test_a_window_past_the_end_is_empty_not_an_error(env):  # noqa: F811
    staff = env.login("staff")
    d = doc_id_of(upload(env, staff, data=b"some text"))
    r = env.client.get(f"/v1/documents/{d}/passages?start=50", headers=staff)
    assert r.status_code == 200 and r.json()["passages"] == []


def test_every_view_is_audited_with_the_document_id_but_not_its_text_or_name(env):  # noqa: F811
    staff = env.login("staff")
    d = doc_id_of(upload(env, staff, name="salaries_2027.txt", data=b"some text"))
    env.client.get(f"/v1/documents/{d}/passages", headers=staff)
    rows = [json.loads(line) for line in open(env.audit_path, encoding="utf-8")]
    view = next(r for r in rows if r["event"] == "document_view")
    assert view["user"] == "staff" and view["doc_id"] == d and view["passages"] == 3
    assert "passage 0 of the document" not in json.dumps(view) and "salaries" not in json.dumps(view)


# ------------------------------------------------------------------ the real pipeline: order, pages, access control

def test_the_real_pipeline_returns_passages_in_reading_order_only_to_people_who_may_read_them(tmp_path):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("qdrant_client")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from access_policy import get_default_policy
    from vector_db import close_qdrant_client
    from encryptor import Encryptor
    from pipeline_config import get_preset
    from rag_pipeline import RAGPipeline

    policy = get_default_policy()
    manager, employee = policy.make_user("mgr", "Manager"), policy.make_user("emp", "Employee")
    cfg = get_preset("dense_only").with_overrides(qdrant_path=str(tmp_path / "q"), collection="t", chunk_size=12, chunk_overlap=0)
    rag = RAGPipeline(cfg, encryptor=Encryptor(key=AESGCM.generate_key(bit_length=256)), policy=policy)
    lines = [f"Section {i} of the manager handbook covers topic number {i}." for i in range(6)]
    f = tmp_path / "handbook.md"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        doc = rag.ingest_document(str(f), policy.make_acl("Manager", []))
        assert doc["chunks"] == 6
        window = rag.read_passages(doc["doc_id"], manager, start=2, limit=3)
        assert [p["chunk_index"] for p in window] == [2, 3, 4]
        assert [p["text"] for p in window] == lines[2:5] and all(p["page"] is None and "scanned" not in p for p in window)
        assert rag.read_passages(doc["doc_id"], manager, start=5, limit=5) == [{"chunk_index": 5, "text": lines[5], "page": None}]
        assert rag.read_passages(doc["doc_id"], employee, start=0, limit=5) == []      # the chunk filter refuses, whatever the caller checked
        assert rag.read_passages("f" * 32, manager) == []
    finally:
        close_qdrant_client(cfg.qdrant_path)
