"""Load test for the chat endpoint (alignment_plan OBS-03): how does latency grow as users are added?

    python -m loadtest.chat_load --label baseline --levels 1,5,10,20

Run from backend/ (inside the worker container so it can reach http://api:8000). It simulates N users who
each ask questions one after another (a "closed loop"), and reports for every level:
  * time to first token (TTFT) and total time, p50 / p95: what a user feels
  * server-side retrieval time (from the X-Retrieval-Ms header): the CPU-heavy part
  * /healthz latency sampled every 100 ms DURING the test: if the server's event loop is blocked by heavy
    work, even this trivial endpoint stalls, which is exactly the problem INF-02 fixes
  * throughput and error counts
Credentials: LOAD_USER / LOAD_PASSWORD env vars, or the first demo user in demo_credentials.txt (local dev).
Results are written to loadtest/results/<label>__<timestamp>.json.
"""
import argparse
import asyncio
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
BACKEND = os.path.dirname(HERE)


def credentials():
    user, pw = os.getenv("LOAD_USER"), os.getenv("LOAD_PASSWORD")
    if user and pw:
        return user, pw
    path = os.path.join(BACKEND, "demo_credentials.txt")
    for line in open(path, encoding="utf-8").read().splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "demo_employee":
            return parts[0], parts[1]
    raise SystemExit("Set LOAD_USER and LOAD_PASSWORD (or create demo users with seed_demo_users.py)")


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))] if values else None


async def one_request(client, base, headers, question):
    payload = {"model": "rag-model", "stream": True, "messages": [{"role": "user", "content": question}]}
    t0, first, tokens = time.perf_counter(), None, 0
    try:
        async with client.stream("POST", f"{base}/v1/chat/completions", json=payload, headers=headers) as r:
            status = r.status_code
            retrieval_ms = float(r.headers.get("x-retrieval-ms", "nan"))
            async for line in r.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        delta = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        tokens += 1
                        if first is None:
                            first = time.perf_counter() - t0
        ok = status == 200 and first is not None
        return {"ok": ok, "status": status, "ttft": first, "total": time.perf_counter() - t0,
                "retrieval_ms": retrieval_ms, "chunks": tokens}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status": type(e).__name__, "ttft": None, "total": time.perf_counter() - t0,
                "retrieval_ms": float("nan"), "chunks": 0}


async def health_probe(client, base, stop, samples):
    while not stop.is_set():
        t0 = time.perf_counter()
        try:
            await client.get(f"{base}/healthz")
            samples.append((time.perf_counter() - t0) * 1000)
        except Exception:  # noqa: BLE001
            samples.append(float("inf"))
        await asyncio.sleep(0.1)


async def run_level(base, headers, questions, users, requests_total):
    queue = list(questions[i % len(questions)] for i in range(requests_total))
    results, health, stop = [], [], asyncio.Event()
    limits = httpx.Limits(max_connections=users + 5)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), limits=limits) as client, \
            httpx.AsyncClient(timeout=30.0) as probe_client:
        probe = asyncio.create_task(health_probe(probe_client, base, stop, health))

        async def user():
            while queue:
                results.append(await one_request(client, base, headers, queue.pop()))

        t0 = time.perf_counter()
        await asyncio.gather(*(user() for _ in range(users)))
        wall = time.perf_counter() - t0
        stop.set()
        await probe

    good = [r for r in results if r["ok"]]
    finite_health = [h for h in health if h != float("inf")]
    row = {
        "users": users, "requests": len(results), "errors": len(results) - len(good),
        "wall_s": round(wall, 1), "throughput_rps": round(len(good) / wall, 2),
        "ttft_p50_ms": round(pct([r["ttft"] * 1000 for r in good], 0.5)) if good else None,
        "ttft_p95_ms": round(pct([r["ttft"] * 1000 for r in good], 0.95)) if good else None,
        "total_p50_ms": round(pct([r["total"] * 1000 for r in good], 0.5)) if good else None,
        "total_p95_ms": round(pct([r["total"] * 1000 for r in good], 0.95)) if good else None,
        "retrieval_p50_ms": round(statistics.median(r["retrieval_ms"] for r in good)) if good else None,
        "health_p50_ms": round(pct(finite_health, 0.5)) if finite_health else None,
        "health_p95_ms": round(pct(finite_health, 0.95)) if finite_health else None,
        "health_max_ms": round(max(finite_health)) if finite_health else None,
        "health_timeouts": len(health) - len(finite_health),
    }
    return row


async def main_async(args):
    base = args.base_url.rstrip("/")
    user, pw = credentials()
    with open(os.path.join(BACKEND, "eval_data", "squad_dataset.json"), encoding="utf-8") as f:
        questions = [q["question"] for q in json.load(f)]
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{base}/v1/auth/login", json={"username": user, "password": pw})
        r.raise_for_status()
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        await c.get(f"{base}/healthz")

    # warm-up: loads models / caches so the first measured level is not penalized
    async with httpx.AsyncClient(timeout=120) as c:
        for q in questions[:3]:
            await one_request(c, base, headers, q)

    rows = []
    print(f"{'users':>5} {'reqs':>5} {'err':>4} {'rps':>6} | {'TTFT p50':>9} {'p95':>7} | {'total p50':>10} {'p95':>7} | "
          f"{'retr p50':>8} | {'/healthz p50':>12} {'p95':>6} {'max':>6}")
    for users in args.levels:
        n = max(args.min_requests, users * args.per_user)
        row = await run_level(base, headers, questions, users, n)
        rows.append(row)
        print(f"{row['users']:>5} {row['requests']:>5} {row['errors']:>4} {row['throughput_rps']:>6} | "
              f"{row['ttft_p50_ms']:>7} ms {row['ttft_p95_ms']:>5} | {row['total_p50_ms']:>8} ms {row['total_p95_ms']:>5} | "
              f"{row['retrieval_p50_ms']:>6} ms | {row['health_p50_ms']:>10} ms {row['health_p95_ms']:>6} {row['health_max_ms']:>6}")

    os.makedirs(RESULTS, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = {"label": args.label, "created_utc": stamp, "base_url": base, "levels": args.levels,
           "machine": {"python": platform.python_version(), "cpu_count": os.cpu_count()},
           "note": args.note, "rows": rows}
    path = os.path.join(RESULTS, f"{args.label}__{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="e.g. baseline or after-threadpool")
    ap.add_argument("--base-url", default=os.getenv("LOAD_BASE_URL", "http://api:8000"))
    ap.add_argument("--levels", type=lambda s: [int(x) for x in s.split(",")], default=[1, 5, 10, 20])
    ap.add_argument("--per-user", type=int, default=4, help="requests each simulated user sends")
    ap.add_argument("--min-requests", type=int, default=8)
    ap.add_argument("--note", default="")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
