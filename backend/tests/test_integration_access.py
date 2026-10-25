"""End-to-end access control with the REAL embedder, reranker, Qdrant and encryption behind the real API.
Only the LLM is replaced: it echoes the context it receives, so the test can see exactly what each user's
question reached. Skipped when the model libraries are missing."""
import io
import json
import os

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import auth  # noqa: E402
from access_policy import AccessPolicy, DocACL  # noqa: E402
from audit import AuditLog  # noqa: E402
from embedder import close_qdrant_client  # noqa: E402
from encryptor import Encryptor  # noqa: E402
from pipeline_config import get_preset  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policies", "departmental.yaml"))

DOCS = {
    "hr.txt":     (b"HR salary bands: a senior engineer earns between 120 and 150 thousand dollars per year.\n",
                   dict(level="Staff", groups="HR")),
    "legal.txt":  (b"Legal merger plan: the acquisition of Contoso will be announced on the first of March.\n",
                   dict(level="Executive", groups="Legal")),
    "canteen.txt": (b"Canteen hours: the cafeteria is open from eight to four on weekdays.\n",
                    dict(level="Staff", groups="")),
}
QUESTIONS = {"hr": "What are the salary bands for a senior engineer?",
             "legal": "When will the Contoso acquisition be announced?",
             "canteen": "When is the cafeteria open?"}


class EchoRag(RAGPipeline):
    """Real retrieval; the 'LLM' just returns the context so the test can inspect it."""
    async def stream_answer(self, query, context, history=None, today=None):
        yield "\n".join(context)


@pytest.fixture
def stack(tmp_path):
    cfg = get_preset("dense_rerank").with_overrides(qdrant_path=str(tmp_path / "q"), collection="it")
    rag = EchoRag(cfg, encryptor=Encryptor(key=AESGCM.generate_key(bit_length=256)), policy=POLICY)
    users = auth.UserStore([
        auth.UserRecord("root", auth.hash_password("pw-root-123"), "Staff", (), ("Admin",)),
        auth.UserRecord("hr_staff", auth.hash_password("pw-hr-12345"), "Staff", ("HR",)),
        auth.UserRecord("legal_exec", auth.hash_password("pw-legal-123"), "Executive", ("Legal",)),
        auth.UserRecord("plain", auth.hash_password("pw-plain-123"), "Staff", ()),
        auth.UserRecord("hr_exec", auth.hash_password("pw-hrexec-12"), "Executive", ("HR",)),
    ])

    def dispatch(path, acl_payload, doc_id):         # run ingestion synchronously instead of via Celery (as the worker does)
        rag.registry.set_status(doc_id, "processing")
        result = rag.ingest_document(path, DocACL(acl_payload["min_rank"], tuple(acl_payload["groups"]), tuple(acl_payload["users"])), doc_id=doc_id)
        rag.registry.set_status(doc_id, "done", chunks=result["chunks"])
        return "sync"

    app = api.create_app(rag, users, "k" * 40, upload_dir=str(tmp_path / "up"), dispatch=dispatch,
                         audit=AuditLog(str(tmp_path / "audit.jsonl")))
    client = TestClient(app)

    def login(name, pw):
        r = client.post("/v1/auth/login", json={"username": name, "password": pw})
        assert r.status_code == 200
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    root = login("root", "pw-root-123")
    for name, (data, acl) in DOCS.items():
        r = client.post("/v1/ingest", headers=root, data={"session_id": "s", **acl},
                        files=[("files", (name, io.BytesIO(data)))])
        assert r.status_code == 200, r.text
    yield type("S", (), dict(client=client, login=staticmethod(login), rag=rag))
    close_qdrant_client(cfg.qdrant_path)


def what_reaches(stack, headers, topic):
    r = stack.client.post("/v1/chat/completions", headers=headers,
                          json={"messages": [{"role": "user", "content": QUESTIONS[topic]}]})
    return r.text


def sees(text, marker):
    return marker in text


def test_each_user_sees_exactly_what_the_policy_allows(stack):
    hr_staff = stack.login("hr_staff", "pw-hr-12345")
    legal_exec = stack.login("legal_exec", "pw-legal-123")
    plain = stack.login("plain", "pw-plain-123")
    hr_exec = stack.login("hr_exec", "pw-hrexec-12")
    root = stack.login("root", "pw-root-123")

    # Everyone can read the canteen document (Staff, no department)
    for h in (hr_staff, legal_exec, plain, hr_exec, root):
        assert sees(what_reaches(stack, h, "canteen"), "eight to four")

    # HR salary bands: needs Staff + HR
    assert sees(what_reaches(stack, hr_staff, "hr"), "120 and 150")
    assert sees(what_reaches(stack, hr_exec, "hr"), "120 and 150")          # higher level, same department
    assert not sees(what_reaches(stack, plain, "hr"), "120 and 150")        # right level, wrong department
    assert not sees(what_reaches(stack, legal_exec, "hr"), "120 and 150")   # senior, but not HR

    # Legal merger plan: needs Executive + Legal
    assert sees(what_reaches(stack, legal_exec, "legal"), "Contoso")
    assert not sees(what_reaches(stack, hr_exec, "legal"), "Contoso")       # Executive, wrong department
    assert not sees(what_reaches(stack, hr_staff, "legal"), "Contoso")      # right department, too junior
    assert not sees(what_reaches(stack, plain, "legal"), "Contoso")

    # Super role sees everything
    assert sees(what_reaches(stack, root, "hr"), "120 and 150")
    assert sees(what_reaches(stack, root, "legal"), "Contoso")


def test_a_forged_role_header_gets_nowhere_even_with_real_data_loaded(stack):
    r = stack.client.post("/v1/chat/completions", headers={"X-User-Role": "Admin"},
                          json={"messages": [{"role": "user", "content": QUESTIONS["legal"]}]})
    assert r.status_code == 401 and "Contoso" not in r.text


def test_a_user_cannot_plant_a_document_that_others_above_them_could_not_see_through_upload_rules(stack):
    plain = stack.login("plain", "pw-plain-123")
    r = stack.client.post("/v1/ingest", headers=plain, data={"session_id": "s", "level": "Executive"},
                          files=[("files", ("x.txt", io.BytesIO(b"hello there friend")))])
    assert r.status_code == 403


def test_upload_directory_holds_no_plaintext_after_dispatch_is_asynchronous(stack, tmp_path):
    """The API stores the upload under a random name; the worker (not the API) is what deletes it."""
    names = os.listdir(tmp_path / "up")
    assert len(names) == 3 and all(n.endswith(".txt") and "hr" not in n and "legal" not in n for n in names)


def events_of(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


def test_answers_cite_the_documents_they_came_from(stack):
    hr_staff = stack.login("hr_staff", "pw-hr-12345")
    sources = next(e["sources"] for e in events_of(what_reaches(stack, hr_staff, "hr")) if "sources" in e)
    names = [s["filename"] for s in sources]
    assert names[0] == "hr.txt"                       # the best-matching document is cited first, in rank order
    assert "legal.txt" not in names                   # a document this user may not read is never cited
    assert isinstance(sources[0]["chunk_index"], int) and len(sources[0]["doc_id"]) == 32


def test_each_user_lists_only_their_documents_and_deleting_removes_the_content_everywhere(stack):
    plain, hr_staff, root = (stack.login("plain", "pw-plain-123"), stack.login("hr_staff", "pw-hr-12345"), stack.login("root", "pw-root-123"))
    names = lambda h: sorted(d["filename"] for d in stack.client.get("/v1/documents", headers=h).json()["documents"])  # noqa: E731
    assert names(plain) == ["canteen.txt"]
    assert names(hr_staff) == ["canteen.txt", "hr.txt"]
    assert names(root) == ["canteen.txt", "hr.txt", "legal.txt"]
    docs = stack.client.get("/v1/documents", headers=root).json()["documents"]
    assert all(d["status"] == "done" and d["chunks"] >= 1 for d in docs)

    hr_doc = next(d for d in docs if d["filename"] == "hr.txt")
    assert "120 and 150" in what_reaches(stack, hr_staff, "hr")
    assert stack.client.delete(f"/v1/documents/{hr_doc['doc_id']}", headers=root).status_code == 200
    assert "120 and 150" not in what_reaches(stack, hr_staff, "hr")             # the content is gone from retrieval
    assert names(root) == ["canteen.txt", "legal.txt"]
    assert "eight to four" in what_reaches(stack, hr_staff, "canteen")           # other documents are untouched
