"""INF-02: retrieval must not block the event loop. Uses a deliberately slow, blocking fake retrieval."""
import asyncio
import os
import threading
import time

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

import api  # noqa: E402
import auth  # noqa: E402
from access_policy import AccessPolicy  # noqa: E402
from audit import AuditLog  # noqa: E402
from ratelimit import RateLimiter  # noqa: E402

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policy.yaml"))
SECRET = "c" * 40
DELAY = 0.3


class SlowRag:
    """retrieve BLOCKS (time.sleep) like real CPU-bound work, and records how many ran at once."""
    def __init__(self, tokens=2):
        self.policy, self.tokens = POLICY, tokens
        self.active = self.max_active = 0
        self._lock = threading.Lock()

    def retrieve(self, query, user):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(DELAY)
        with self._lock:
            self.active -= 1
        from types import SimpleNamespace
        return SimpleNamespace(to_context=lambda: ["<doc>some context</doc>"], sources=lambda: [])

    async def stream_answer(self, query, context):
        for i in range(self.tokens):
            yield f"t{i} "


def build(tmp_path, rag, workers):
    store = auth.UserStore([auth.UserRecord("u", auth.hash_password("pw-long-enough"), "Employee")])
    return api.create_app(rag, store, SECRET, upload_dir=str(tmp_path / "up"), audit=AuditLog(str(tmp_path / "a.jsonl")),
                          login_limiter=RateLimiter(max_attempts=1000), retrieval_workers=workers)


async def run_scenario(app, n_requests):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        tok = (await c.post("/v1/auth/login", json={"username": "u", "password": "pw-long-enough"})).json()["access_token"]
        headers = {"Authorization": f"Bearer {tok}"}
        health, done = [], asyncio.Event()

        async def probe():
            while not done.is_set():
                t0 = time.perf_counter()
                r = await c.get("/healthz")
                assert r.status_code == 200
                health.append(time.perf_counter() - t0)
                await asyncio.sleep(0.02)

        async def chat():
            r = await c.post("/v1/chat/completions", headers=headers,
                             json={"messages": [{"role": "user", "content": "hi"}]})
            assert r.status_code == 200
            return r

        probe_task = asyncio.create_task(probe())
        t0 = time.perf_counter()
        await asyncio.gather(*(chat() for _ in range(n_requests)))
        wall = time.perf_counter() - t0
        done.set()
        await probe_task
        return wall, health


def test_healthz_is_unauthenticated_and_trivial(tmp_path):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build(tmp_path, SlowRag(), 2)), base_url="http://t") as c:
            return await c.get("/healthz")
    r = asyncio.run(go())
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_event_loop_stays_responsive_while_retrieval_blocks(tmp_path):
    """With the old code every blocking retrieval froze the loop, so /healthz stalled ~DELAY or more."""
    rag = SlowRag()
    wall, health = asyncio.run(run_scenario(build(tmp_path, rag, workers=4), n_requests=8))
    assert health, "the probe never ran"
    assert max(health) < DELAY / 2, f"event loop was blocked: slowest /healthz took {max(health) * 1000:.0f} ms"


def test_requests_overlap_but_never_exceed_the_worker_cap(tmp_path):
    rag = SlowRag()
    wall, _ = asyncio.run(run_scenario(build(tmp_path, rag, workers=4), n_requests=8))
    assert 2 <= rag.max_active <= 4                       # real parallelism, bounded
    assert wall < 8 * DELAY * 0.6                         # ~2 waves of 4, not 8 in a row


def test_one_worker_serializes_as_configured(tmp_path):
    rag = SlowRag()
    wall, _ = asyncio.run(run_scenario(build(tmp_path, rag, workers=1), n_requests=4))
    assert rag.max_active == 1 and wall >= 4 * DELAY * 0.9


def test_tokens_stream_without_an_artificial_per_token_delay(tmp_path):
    """The old loop slept 10 ms after every token (0.5 s for 50 tokens) for no reason."""
    rag = SlowRag(tokens=60)
    wall, _ = asyncio.run(run_scenario(build(tmp_path, rag, workers=4), n_requests=1))
    assert wall < DELAY + 0.25
