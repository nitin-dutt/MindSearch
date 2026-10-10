import json
import os

import pytest

pytest.importorskip("qdrant_client")

from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402

import doc_registry as R  # noqa: E402
from access_policy import AccessPolicy  # noqa: E402
from embedder import close_qdrant_client, get_qdrant_client  # noqa: E402
from encryptor import Encryptor  # noqa: E402

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policies", "departmental.yaml"))
HR_STAFF = POLICY.make_user("hr_staff", "Staff", ["HR"])
LEGAL_EXEC = POLICY.make_user("legal_exec", "Executive", ["Legal"])
PLAIN = POLICY.make_user("plain", "Staff")
ROOT = POLICY.make_user("root", "Staff", roles=["Admin"])


@pytest.fixture
def reg(tmp_path):
    location = str(tmp_path / "q")
    registry = R.DocumentRegistry(location, Encryptor(key=AESGCM.generate_key(bit_length=256)), POLICY)
    yield registry
    close_qdrant_client(location)


def add(reg, doc_id, filename, level="Staff", groups=(), uploader="hr_staff", size=123):
    return reg.register(doc_id, POLICY.make_acl(level, groups), filename, uploader, size)


def test_register_creates_a_queued_entry_with_decrypted_metadata(reg):
    e = add(reg, "d1", "salary_bands.pdf", "Staff", ["HR"])
    assert (e.doc_id, e.status, e.chunks, e.filename, e.uploader, e.size) == ("d1", "queued", 0, "salary_bands.pdf", "hr_staff", 123)


def test_filename_and_uploader_are_not_stored_in_plaintext(reg):
    add(reg, "d1", "layoffs_2027.pdf", "Staff", ["HR"], uploader="alice")
    points, _ = get_qdrant_client(reg.location).scroll(reg.collection, limit=10, with_payload=True)
    raw = json.dumps([p.payload for p in points])
    assert "layoffs_2027" not in raw and "alice" not in raw
    assert points[0].payload["status"] == "queued"          # non-sensitive fields stay readable for filtering


def test_status_lifecycle_and_failure_message(reg):
    add(reg, "d1", "a.txt")
    reg.set_status("d1", "processing")
    assert reg.get("d1").status == "processing"
    reg.set_status("d1", "done", chunks=7)
    e = reg.get("d1")
    assert (e.status, e.chunks, e.error) == ("done", 7, None)
    reg.set_status("d1", "failed", error="Neo4j unavailable")
    e = reg.get("d1")
    assert e.status == "failed" and e.error == "Neo4j unavailable"
    reg.set_status("d1", "processing")
    assert reg.get("d1").error is None                       # a retry clears the old error


def test_listing_only_shows_documents_the_user_may_read(reg):
    add(reg, "hr", "hr.txt", "Staff", ["HR"])
    add(reg, "legal", "legal.txt", "Executive", ["Legal"])
    add(reg, "all", "canteen.txt", "Staff", [])
    assert {e.doc_id for e in reg.list(HR_STAFF)} == {"hr", "all"}
    assert {e.doc_id for e in reg.list(LEGAL_EXEC)} == {"legal", "all"}
    assert {e.doc_id for e in reg.list(PLAIN)} == {"all"}
    assert {e.doc_id for e in reg.list(ROOT)} == {"hr", "legal", "all"}


def test_get_hides_documents_the_user_cannot_read_instead_of_revealing_they_exist(reg):
    add(reg, "legal", "legal.txt", "Executive", ["Legal"])
    assert reg.get("legal", PLAIN) is None and reg.get("legal", LEGAL_EXEC).doc_id == "legal"
    assert reg.get("does-not-exist", ROOT) is None


def test_newest_documents_are_listed_first(reg, monkeypatch):
    stamps = iter(["2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00", "2026-01-03T00:00:00+00:00"] * 3)
    monkeypatch.setattr(R, "_now", lambda: next(stamps))
    for name in ("old", "mid", "new"):
        add(reg, name, f"{name}.txt")
    assert [e.doc_id for e in reg.list(ROOT)] == ["new", "mid", "old"]


def test_editing_the_access_rule_in_the_database_makes_metadata_unreadable_not_exposed(reg):
    add(reg, "d1", "secret.pdf", "Executive", ["Legal"])
    client = get_qdrant_client(reg.location)
    client.set_payload(reg.collection, payload={"min_rank": 0, "groups": []}, points=[R.registry_point_id("d1")])
    e = reg.get("d1", PLAIN)                                 # the widened filter now lets PLAIN find it...
    assert e is not None and e.filename == R.UNREADABLE and e.uploader == R.UNREADABLE   # ...but not read the metadata


def test_filenames_resolves_retrieved_documents(reg):
    add(reg, "a", "a.pdf")
    add(reg, "b", "b.pdf")
    assert reg.filenames(["a", "b", "missing"]) == {"a": "a.pdf", "b": "b.pdf"}
    assert reg.filenames([]) == {}


def test_delete_removes_the_entry_and_is_idempotent(reg):
    add(reg, "d1", "a.txt")
    reg.delete("d1")
    reg.delete("d1")
    assert reg.get("d1") is None and reg.list(ROOT) == []


def test_public_view_hides_the_uploader_unless_asked(reg):
    e = add(reg, "d1", "a.txt", "Staff", ["HR"], uploader="alice")
    assert "uploaded_by" not in e.to_public(POLICY, include_uploader=False)
    pub = e.to_public(POLICY, include_uploader=True)
    assert pub["uploaded_by"] == "alice" and pub["level"] == "Staff" and pub["groups"] == ["HR"]
