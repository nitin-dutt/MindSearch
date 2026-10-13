"""Tune how much context to send to the LLM, on the DEV set only (alignment_plan RET-04, GEN-05).

    python -m evaluation.tune_context --dataset squad_dev            # grid of cutoff rules
    python -m evaluation.tune_context --dataset squad_dev --latency  # reranker speed options

The cross-encoder scores every dense candidate once; every rule (how many candidates to rerank,
how many chunks to keep) is then replayed offline on those same scores, so a whole grid costs
one pass. Judge-free: coverage is exact (does a kept chunk contain the gold passage).
"""
import argparse
import statistics
import time

from evaluation import metrics as M
from evaluation.datasets import load_dataset
from evaluation.generate import build_pipeline
from pipeline_config import get_preset

MAX_KEEP = 3


def collect(rag, items):
    """For each question: gold, dense candidate texts (dense order) and their reranker scores."""
    rows = []
    for it in items:
        pool = rag.retrieve(it["question"], role="Admin").dense_pool
        scores = [float(s) for s in rag.reranker.model.predict([[it["question"], t] for t in pool])]
        rows.append({"question": it["question"], "gold": it["gold_passages"], "pool": pool, "scores": scores})
    return rows


def select(row, k, margin):
    """Rerank only the first k dense candidates; keep the best, plus up to MAX_KEEP-1 more whose
    score is within `margin` of the best (margin=None keeps a fixed top MAX_KEEP)."""
    cand = list(range(min(k, len(row["pool"]))))
    order = sorted(cand, key=lambda i: -row["scores"][i])
    if not order:
        return []
    top = row["scores"][order[0]]
    keep = [order[0]]
    for i in order[1:MAX_KEEP]:
        if margin is None or row["scores"][i] >= top - margin:
            keep.append(i)
    return keep


def evaluate(rows, k, margin):
    recall, hit, prec, words, count, mrr = [], [], [], [], [], []
    for row in rows:
        keep = select(row, k, margin)
        texts = [row["pool"][i] for i in keep]
        cover = M.coverage(texts, row["gold"])
        recall.append(M.recall_at_k(cover, len(row["gold"]), len(texts)))
        hit.append(M.hit_at_k(cover, len(texts)))
        prec.append(M.precision(cover))
        mrr.append(M.mrr(cover))
        words.append(sum(len(t.split()) for t in texts))
        count.append(len(texts))
    mean = statistics.fmean
    return {"recall": mean(recall), "mrr": mean(mrr), "precision": mean(v for v in prec if v is not None),
            "words": mean(words), "chunks": mean(count)}


def grid(rows):
    base = evaluate(rows, 20, None)
    print(f"n={len(rows)} dev questions. Baseline = rerank 20 candidates, keep 3 "
          f"(recall {base['recall']:.3f}, precision {base['precision']:.3f}, {base['words']:.0f} words)\n")
    print(f"{'candidates':>10} {'cutoff':>12} {'recall':>8} {'d_recall':>9} {'MRR':>7} {'precision':>10} {'words':>7} {'chunks':>7}")
    for k in (20, 10, 5):
        for margin in (None, 6.0, 4.0, 3.0, 2.0, 1.0, 0.0):
            r = evaluate(rows, k, margin)
            label = "top-3" if margin is None else ("top-1" if margin == 0.0 else f"within {margin:g}")
            print(f"{k:>10} {label:>12} {r['recall']:>8.3f} {r['recall'] - base['recall']:>+9.3f} {r['mrr']:>7.3f} "
                  f"{r['precision']:>10.3f} {r['words']:>7.0f} {r['chunks']:>7.2f}")


def latency(rag, rows, n=60):
    """Time the cross-encoder on the first n questions for different candidate counts / settings."""
    model = rag.reranker.model

    def timed(k, label):
        times = []
        for r in rows[:n]:
            pairs = [[r["question"], t] for t in r["pool"][:k]]
            t0 = time.perf_counter()
            model.predict(pairs)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        print(f"  {label:<44} p50 {statistics.median(times):6.0f} ms   p95 {times[int(len(times) * 0.95) - 1]:6.0f} ms")

    model.predict([[rows[0]["question"], rows[0]["pool"][0]]])  # warm-up
    print("Reranker latency (CPU), per query:")
    for k in (20, 10, 5):
        timed(k, f"{k} candidates, default settings")
    default_len = model.max_length
    model.max_length = 256
    for k in (20, 10):
        timed(k, f"{k} candidates, max_length=256")
    model.max_length = default_len
    # int8 dynamic quantization was tried and does not work with this CrossEncoder wrapper (BatchEncoding
    # is passed to the quantized module); ONNX/OpenVINO backends are the untested next option.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="squad_dev")
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()
    if args.dataset == "squad":
        raise SystemExit("Refusing to tune on the frozen test set 'squad'. Use squad_dev.")

    corpus, items = load_dataset(args.dataset, limit=60 if args.latency else None)
    rag, _, _ = build_pipeline(args.dataset, corpus, get_preset("dense_rerank"))
    rows = collect(rag, items)
    if args.latency:
        latency(rag, rows)
    else:
        grid(rows)


if __name__ == "__main__":
    main()
