"""API security behaviour with a fake pipeline (no models, no databases)."""
import io
import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import auth  # noqa: E402
import uploads  # noqa: E402
from access_policy import AccessPolicy, PolicyError  # noqa: E402
from audit import AuditLog  # noqa: E402
from ratelimit import RateLimiter  # noqa: E402

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policies", "departmental.yaml"))
SECRET = "s" * 40


class FakeRag:
    def __init__(self, context=True):
        self.policy, self.context = POLICY, context
        self.asked_as = []

    def retrieve_context(self, query, user):
        self.asked_as.append(user)
        return [f"<doc>visible to {user.id} (rank {user.rank})</doc>"] if self.context else []

    async def stream_answer(self, query, context):
        for tok in ("Hello ", "world"):
            yield tok


@pytest.fixture
def env(tmp_path):
    records = [
        auth.UserRecord("staff", auth.hash_password("pw-staff"), "Staff", ("HR",)),
        auth.UserRecord("mgr", auth.hash_password("pw-mgr"), "Manager", ("HR", "Finance")),
        auth.UserRecord("root", auth.hash_password("pw-root"), "Staff", (), ("Admin",)),
    ]
    store = auth.UserStore(records)
    dispatched, rag = [], FakeRag()
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    app = api.create_app(rag, store, SECRET, upload_dir=str(tmp_path / "up"),
                         dispatch=lambda path, acl: dispatched.append((path, acl)) or f"task-{len(dispatched)}",
                         audit=audit, login_limiter=RateLimiter(max_attempts=5, window_seconds=60))
    client = TestClient(app)

    def login(user):
        r = client.post("/v1/auth/login", json={"username": user, "password": f"pw-{user}"})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    return type("Env", (), dict(client=client, rag=rag, dispatched=dispatched, audit_path=str(tmp_path / "audit.jsonl"),
                                upload_dir=str(tmp_path / "up"), login=staticmethod(login), store=store))


CHAT = {"messages": [{"role": "user", "content": "what is the secret plan?"}]}


def chat(env, headers=None):
    return env.client.post("/v1/chat/completions", json=CHAT, headers=headers or {})


# ------------------------------------------------------------ authentication

def test_every_protected_route_rejects_anonymous_requests(env):
    assert chat(env).status_code == 401
    assert env.client.post("/v1/ingest", data={"session_id": "s"},
                           files=[("files", ("a.txt", b"hi"))]).status_code == 401
    assert env.client.get("/v1/me").status_code == 401
    assert env.client.get("/v1/policy").status_code == 401


def test_the_old_forged_role_header_no_longer_grants_anything(env):
    """Before: anyone could send X-User-Role: Admin. Now the header is ignored and no token means no access."""
    r = chat(env, {"X-User-Role": "Admin"})
    assert r.status_code == 401
    assert env.rag.asked_as == []              # retrieval was never reached


def test_login_success_and_failures(env):
    ok = env.client.post("/v1/auth/login", json={"username": "staff", "password": "pw-staff"})
    assert ok.status_code == 200 and ok.json()["token_type"] == "bearer"
    assert env.client.post("/v1/auth/login", json={"username": "staff", "password": "bad"}).status_code == 401
    assert env.client.post("/v1/auth/login", json={"username": "ghost", "password": "x"}).status_code == 401


def test_failed_logins_give_the_same_message_for_known_and_unknown_users(env):
    a = env.client.post("/v1/auth/login", json={"username": "staff", "password": "bad"}).json()
    b = env.client.post("/v1/auth/login", json={"username": "ghost", "password": "bad"}).json()
    assert a == b                                # does not reveal which usernames exist


def test_repeated_failed_logins_are_rate_limited(env):
    codes = [env.client.post("/v1/auth/login", json={"username": "staff", "password": "bad"}).status_code
             for _ in range(7)]
    assert codes[:5] == [401] * 5 and codes[5:] == [429, 429]


def test_tampered_or_garbage_tokens_are_rejected(env):
    good = env.login("staff")["Authorization"].split()[1]
    head, payload, sig = good.split(".")
    evil = auth._b64e(json.dumps({"sub": "root", "exp": 9999999999}).encode())
    assert chat(env, {"Authorization": f"Bearer {head}.{evil}.{sig}"}).status_code == 401
    assert chat(env, {"Authorization": "Bearer nonsense"}).status_code == 401
    assert chat(env, {"Authorization": "Basic abc"}).status_code == 401


def test_token_for_a_removed_user_stops_working(env):
    headers = env.login("staff")
    assert chat(env, headers).status_code == 200
    env.store._by_name.pop("staff")             # user deleted from users.yaml
    assert chat(env, headers).status_code == 401


# --------------------------------------------------------------- authorization

def test_chat_runs_as_the_authenticated_user_not_as_anyone_else(env):
    chat(env, env.login("staff"))
    chat(env, env.login("mgr"))
    staff, mgr = env.rag.asked_as
    assert (staff.id, staff.rank, staff.groups) == ("staff", 1, ("HR",))
    assert (mgr.id, mgr.rank, mgr.groups) == ("mgr", 2, ("HR", "Finance"))
    assert not staff.is_super and not mgr.is_super


def test_super_role_comes_from_the_user_store(env):
    chat(env, env.login("root"))
    assert env.rag.asked_as[-1].is_super


def test_chat_with_nothing_retrievable_skips_the_llm(env):
    env.rag.context = False
    r = chat(env, env.login("staff"))
    assert api.NO_CONTEXT_MESSAGE in r.text and "Hello" not in r.text


def test_chat_streams_tokens_when_context_exists(env):
    r = chat(env, env.login("staff"))
    assert r.status_code == 200 and "Hello " in r.text and r.text.strip().endswith("[DONE]")


def test_me_and_policy_describe_what_the_user_may_do(env):
    me = env.client.get("/v1/me", headers=env.login("staff")).json()
    assert me == {"username": "staff", "level": "Staff", "groups": ["HR"], "is_super": False}
    pol = env.client.get("/v1/policy", headers=env.login("staff")).json()
    assert pol["levels"] == ["Staff"] and pol["groups"] == ["HR"]
    pol_root = env.client.get("/v1/policy", headers=env.login("root")).json()
    assert pol_root["levels"] == ["Staff", "Manager", "Executive"] and "Legal" in pol_root["groups"]


# ----------------------------------------------------------------------- ingest

def upload(env, headers, name="doc.txt", data=b"hello world", **form):
    return env.client.post("/v1/ingest", headers=headers, data={"session_id": "s", **form},
                           files=[("files", (name, io.BytesIO(data)))])


def test_upload_defaults_to_the_uploaders_own_level_and_groups(env):
    r = upload(env, env.login("mgr"))
    assert r.status_code == 200 and r.json()["documents"][0]["task_id"] == "task-1"
    _, acl = env.dispatched[0]
    assert acl == {"min_rank": 2, "groups": ["HR", "Finance"], "users": []}


def test_user_cannot_upload_above_their_own_clearance(env):
    assert upload(env, env.login("staff"), level="Manager").status_code == 403
    assert env.dispatched == []


def test_user_cannot_grant_a_department_they_do_not_belong_to(env):
    assert upload(env, env.login("staff"), groups="Legal").status_code == 403


def test_super_role_can_choose_any_level_and_group(env):
    r = upload(env, env.login("root"), level="Executive", groups="Legal")
    assert r.status_code == 200 and env.dispatched[0][1]["min_rank"] == 3


def test_unknown_level_or_group_is_a_validation_error(env):
    assert upload(env, env.login("root"), level="Intern").status_code == 422
    assert upload(env, env.login("root"), groups="Marketing").status_code == 422


def test_path_traversal_filename_cannot_escape_the_upload_folder(env):
    r = upload(env, env.login("staff"), name="../../../etc/cron.d/evil.txt")
    assert r.status_code == 200
    stored = env.dispatched[0][0]
    assert os.path.dirname(os.path.abspath(stored)) == os.path.abspath(env.upload_dir)
    assert "evil" not in os.path.basename(stored) and os.path.exists(stored)


def test_two_uploads_with_the_same_name_do_not_overwrite_each_other(env):
    h = env.login("staff")
    upload(env, h, name="report.txt", data=b"first")
    upload(env, h, name="report.txt", data=b"second")
    p1, p2 = env.dispatched[0][0], env.dispatched[1][0]
    assert p1 != p2 and open(p1, "rb").read() == b"first" and open(p2, "rb").read() == b"second"


@pytest.mark.parametrize("name,data,code", [
    ("tool.exe", b"MZ....", 400),                 # extension not allowed
    ("fake.pdf", b"MZ not really a pdf", 400),    # signature does not match
    ("empty.txt", b"", 400),
    ("noext", b"hello", 400),
])
def test_bad_uploads_are_rejected(env, name, data, code):
    assert upload(env, env.login("staff"), name=name, data=data).status_code == code
    assert env.dispatched == []


def test_oversized_upload_is_rejected_with_413(env, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 10)
    assert upload(env, env.login("staff"), data=b"x" * 11).status_code == 413


def test_one_bad_file_in_a_batch_saves_nothing(env):
    files = [("files", ("ok.txt", io.BytesIO(b"fine"))), ("files", ("bad.exe", io.BytesIO(b"MZ")))]
    r = env.client.post("/v1/ingest", headers=env.login("staff"), data={"session_id": "s"}, files=files)
    assert r.status_code == 400 and env.dispatched == [] and os.listdir(env.upload_dir) == []


# ------------------------------------------------------------------------ audit

def test_audit_log_records_events_without_storing_query_text(env):
    env.client.post("/v1/auth/login", json={"username": "staff", "password": "bad"})
    chat(env, env.login("staff"))
    upload(env, env.login("staff"), level="Manager")          # denied
    rows = [json.loads(line) for line in open(env.audit_path, encoding="utf-8")]
    events = [r["event"] for r in rows]
    assert {"login_failure", "login_success", "chat", "ingest_denied"} <= set(events)
    assert "secret plan" not in open(env.audit_path, encoding="utf-8").read()     # query kept as a hash only
    chat_row = next(r for r in rows if r["event"] == "chat")
    assert chat_row["query"]["chars"] == len("what is the secret plan?") and chat_row["user"] == "staff"


# --------------------------------------------------------------------- startup

def test_inconsistent_user_file_fails_at_startup(tmp_path):
    bad = auth.UserStore([auth.UserRecord("x", auth.hash_password("p"), "Intern")])
    with pytest.raises(PolicyError):
        api.create_app(FakeRag(), bad, SECRET, upload_dir=str(tmp_path / "up"))
