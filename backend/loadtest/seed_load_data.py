"""Load documents into the RUNNING system through the real upload API, for demos and load tests.

    python -m loadtest.seed_load_data [--with-demo] [--twice]

Uploads eval_data/squad_corpus.md (97 passages, level Employee) as demo_admin. With --with-demo it also uploads two small
documents at different levels (a canteen notice for Employee, a board memo for Manager). With --twice the corpus is
uploaded a second time, to demonstrate that ingestion is idempotent (the point count must not change).
Run inside the worker container. Local development only: it logs in with the demo credentials.
"""
import argparse
import io
import os
import time

import requests

API = os.getenv("LOAD_BASE_URL", "http://api:8000")
QDRANT = os.getenv("QDRANT_URL", "http://qdrant:6333")
BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def admin_headers():
    creds = {}
    for line in open(os.path.join(BACKEND, "demo_credentials.txt"), encoding="utf-8").read().splitlines()[2:]:
        p = line.split()
        if len(p) >= 2:
            creds[p[0]] = p[1]
    r = requests.post(f"{API}/v1/auth/login", json={"username": "demo_admin", "password": creds["demo_admin"]})
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def points():
    r = requests.get(f"{QDRANT}/collections/mindsearch_docs")
    return r.json()["result"]["points_count"] if r.status_code == 200 else 0


def upload(headers, name, data, level):
    r = requests.post(f"{API}/v1/ingest", headers=headers, data={"session_id": "seed", "level": level},
                      files=[("files", (name, io.BytesIO(data)))])
    r.raise_for_status()


def wait_until_documents_finished(headers, label, timeout=1500):
    """Ingestion is asynchronous. The document registry says exactly where each upload is, so wait until none of
    them is 'queued' or 'processing' (more reliable than asking the single-process Celery worker, which cannot answer
    while it is busy, and than counting points, which settle before the last upload even starts)."""
    start = time.time()
    while time.time() - start < timeout:
        docs = requests.get(f"{API}/v1/documents", headers=headers).json()["documents"]
        if docs and all(d["status"] in ("done", "failed") for d in docs):
            break
        time.sleep(4)
    n = points()
    summary = ", ".join(f"{d['filename']}={d['status']}({d['chunks']})" for d in docs)
    print(f"{label}: {n} points | {summary}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-demo", action="store_true")
    ap.add_argument("--twice", action="store_true", help="upload the corpus twice to prove idempotency")
    args = ap.parse_args()

    headers = admin_headers()
    corpus = open(os.path.join(BACKEND, "eval_data", "squad_corpus.md"), "rb").read()
    upload(headers, "squad_corpus.md", corpus, "Employee")
    if args.with_demo:
        upload(headers, "canteen.txt", b"Canteen hours: the cafeteria is open from eight to four on weekdays. Lunch is served between twelve and two, and the soup of the day is tomato.\n", "Employee")
        upload(headers, "board_memo.txt", b"Board memo: the acquisition of Contoso will be announced on the first of March. The purchase price is two hundred million dollars and Sarah Jenkins will lead the integration.\n", "Manager")
    first = wait_until_documents_finished(headers, "after first upload")
    if args.twice:
        upload(headers, "squad_corpus.md", corpus, "Employee")
        second = wait_until_documents_finished(headers, "after uploading the same corpus again")
        print("IDEMPOTENT: point count unchanged" if second == first else f"NOT IDEMPOTENT: {first} -> {second}")


if __name__ == "__main__":
    main()
