"""Retrieval lab: which embedder, and does keyword search (BM25) help? Judge-free (alignment_plan RET-02 / RET-03).

    python -m evaluation.retrieval_bench --dataset squad_dev_xl --models all-MiniLM-L6-v2 BAAI/bge-base-en-v1.5
    python -m evaluation.retrieval_bench --dataset squad_dev_xl --models all-MiniLM-L6-v2 --rerank

It chunks the corpus exactly as the pipeline does, embeds every chunk with each candidate model (CPU, cached on disk), and
for each question ranks the chunks four ways:

    dense    cosine similarity of embeddings
    bm25     classic keyword scoring (exact words, names, numbers)
    hybrid   dense + bm25 combined by Reciprocal Rank Fusion (RRF: score = sum of 1/(60+rank) over both lists)
    rerank   (with --rerank) the cross-encoder re-scores the top candidates of the hybrid / dense list

and reports recall@k / hit@k / MRR from the exact gold passage (no LLM judge), with bootstrap confidence intervals.
Why in memory and not through Qdrant: it isolates the ranking question from storage details, runs in minutes, and the
winning setup is then implemented in the real pipeline and re-measured end to end.
Run inside the worker container (models are cached in its volume).
"""
import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone

import numpy as np

from chunker import run_chunking
from evaluation import metrics as M
from evaluation.datasets import load_dataset
from evaluation.generate import EVAL_INDEX_DIR, RESULTS_DIR

KS = (1, 3, 5, 10, 20)
RRF_K = 60
POOL = 50                      # candidates each ranking method keeps for fusion / reranking

# Instructions some models need in front of queries / passages (from each model's documentation).
PREFIXES = {
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
}

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the of in on at to for and or is are was were be by with as from that this it its which who what when where how why did do does".split())


def tokenize(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def embed_cached(model_name, texts, prefix, kind, cache_key):
    """Embeddings for `texts`, cached in qdrant_eval/embcache so a repeated run costs nothing."""
    path = os.path.join(EVAL_INDEX_DIR, "embcache", f"{re.sub(r'[^A-Za-z0-9]+', '_', model_name)}__{kind}__{cache_key}.npy")
    if os.path.exists(path):
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    model = embed_cached.models.get(model_name)
    if model is None:
        model = embed_cached.models[model_name] = SentenceTransformer(model_name, device="cpu")
        model.max_seq_length = min(model.max_seq_length or 512, 512)
    t0 = time.time()
    vecs = model.encode([prefix + t for t in texts], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    print(f"    embedded {len(texts)} {kind} with {model_name} in {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, vecs.astype(np.float32))
    return vecs.astype(np.float32)


embed_cached.models = {}


def rrf(rankings, k=RRF_K, top=POOL):
    scores = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:top]


def score_ranking(order, chunks, gold):
    return M.retrieval_metrics([chunks[i] for i in order], gold, ks=KS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="squad_dev_xl")
    ap.add_argument("--models", nargs="+", default=["all-MiniLM-L6-v2"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--rerank", action="store_true", help="also measure cross-encoder reranking of the top candidates")
    ap.add_argument("--rerank-pool", type=int, default=20)
    args = ap.parse_args()
    if args.dataset == "squad":
        raise SystemExit("Use a dev dataset; the frozen test set is for final reporting only.")

    corpus_path, items = load_dataset(args.dataset, args.n)
    chunks = run_chunking(corpus_path)
    cache_key = hashlib.sha256(("\n".join(chunks)).encode()).hexdigest()[:12]
    questions = [it["question"] for it in items]
    print(f"{args.dataset}: {len(chunks)} chunks, {len(items)} questions")

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokenize(c) for c in chunks])
    bm25_orders = []
    for q in questions:
        sc = bm25.get_scores(tokenize(q))
        bm25_orders.append(list(np.argsort(-sc)[:POOL]))

    reranker = None
    if args.rerank:
        from reranker import ReRanker
        reranker = ReRanker()

    results, per_question = {}, {}
    methods = {}
    methods["bm25"] = lambda mi, qi: bm25_orders[qi]
    for model_name in args.models:
        qpref, ppref = PREFIXES.get(model_name, ("", ""))
        print(f"== {model_name}")
        cv = embed_cached(model_name, chunks, ppref, "chunks", cache_key)
        qv = embed_cached(model_name, questions, qpref, "queries", hashlib.sha256("\n".join(questions).encode()).hexdigest()[:12])
        sims = qv @ cv.T
        dense_orders = [list(np.argsort(-sims[i])[:POOL]) for i in range(len(questions))]
        hybrid_orders = [rrf([dense_orders[i], bm25_orders[i]]) for i in range(len(questions))]
        variants = {f"{model_name} | dense": dense_orders, f"{model_name} | hybrid (dense+bm25)": hybrid_orders}
        if reranker:
            for label, orders in (("dense", dense_orders), ("hybrid", hybrid_orders)):
                reranked = []
                t0 = time.time()
                for i, q in enumerate(questions):
                    pool = orders[i][:args.rerank_pool]
                    rr = reranker.rerank(q, [chunks[j] for j in pool], top_k=len(pool))
                    reranked.append([pool[r["original_index"]] for r in rr])
                print(f"    reranked {label} in {time.time() - t0:.0f}s")
                variants[f"{model_name} | {label} + rerank@{args.rerank_pool}"] = reranked
        for name, orders in variants.items():
            per_question[name] = [score_ranking(orders[i], chunks, items[i]["gold_passages"]) for i in range(len(items))]
    per_question["bm25 only"] = [score_ranking(bm25_orders[i], chunks, items[i]["gold_passages"]) for i in range(len(items))]

    keys = ["hit@1", "hit@3", "recall@5", "recall@10", "recall@20", "mrr"]
    for name, rows in per_question.items():
        results[name] = M.summarize(rows, keys)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"retrieval_bench__{args.dataset}__{stamp}")
    os.makedirs(run_dir)
    out = {"dataset": args.dataset, "n_questions": len(items), "n_chunks": len(chunks), "rrf_k": RRF_K, "pool": POOL,
           "rerank_pool": args.rerank_pool if reranker else None, "variants": results}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(run_dir, "per_question.json"), "w", encoding="utf-8") as f:
        json.dump(per_question, f)

    print(f"\n=== {args.dataset}: {len(chunks)} chunks, {len(items)} questions (mean [95% CI]) ===")
    print(f"{'variant':<58}" + "".join(f"{k:>16}" for k in keys))
    for name, s in sorted(results.items(), key=lambda kv: -kv[1]["hit@1"]["mean"]):
        print(f"{name:<58}" + "".join(f"{s[k]['mean']:>8.3f}        " for k in keys))
    print(f"\nSaved {run_dir}")


if __name__ == "__main__":
    main()
