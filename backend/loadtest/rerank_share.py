"""How much of retrieval is the cross-encoder? (alignment_plan PERF-03, step 0: the most a faster re-ranker could save.)

    python -m loadtest.rerank_share [--n 40] [--parallel 4]
    python -m loadtest.rerank_share --agreement [--n 100] [--other onnx|cuda]   # does the other re-ranker pick the same passages as torch/cpu?

Run from backend/ inside the worker container, against the live index (seed it with loadtest.seed_load_data). It builds the same
pipeline the API builds (PIPELINE_PRESET, torch threads = cores / RETRIEVAL_WORKERS), runs `retrieve` for the load-test questions with
`--parallel` at a time (the API's retrieval pool size), and times every cross-encoder call separately from the rest. Read-only.
Prints per-question means: total retrieval seconds, of which cross-encoder seconds, plus the cross-encoder's pairs per call.
"""
import argparse
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--agreement", action="store_true", help="compare torch/cpu with --other on the same search pools")
    ap.add_argument("--other", choices=["onnx", "cuda"], default="onnx", help="onnx = onnx/cpu, cuda = torch/cuda (needs the GPU image)")
    args = ap.parse_args()
    if args.agreement:
        return agreement(args.n, args.other)

    import torch
    from settings import get_settings
    from pipeline_config import get_preset
    from rag_pipeline import RAGPipeline

    cfg = get_settings()
    torch.set_num_threads(cfg.torch_threads or max(1, (os.cpu_count() or 4) // cfg.retrieval_workers))
    rag = RAGPipeline(get_preset(cfg.pipeline_preset))
    user = rag.policy.superuser()
    questions = [q["question"] for q in json.load(open(os.path.join(BACKEND, "eval_data", "squad_dataset.json"), encoding="utf-8"))][: args.n]

    local = threading.local()
    pairs_per_call = []
    original = rag.reranker.rerank

    def timed_rerank(query, chunks, top_k=5):
        t0 = time.perf_counter()
        try:
            return original(query, chunks, top_k=top_k)
        finally:
            local.ce += time.perf_counter() - t0
            pairs_per_call.append(len(chunks))

    rag.reranker.rerank = timed_rerank

    def one(q):
        local.ce = 0.0
        t0 = time.perf_counter()
        rag.retrieve(q, user)
        return time.perf_counter() - t0, local.ce

    one(questions[0])                                  # warm-up: model load, first-call overheads
    pairs_per_call.clear()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        rows = list(pool.map(one, questions))
    wall = time.perf_counter() - t0
    total = [r[0] for r in rows]
    ce = [r[1] for r in rows]
    print(json.dumps({
        "preset": cfg.pipeline_preset, "torch_threads": torch.get_num_threads(), "parallel": args.parallel, "questions": len(rows),
        "wall_s": round(wall, 1), "questions_per_s": round(len(rows) / wall, 2),
        "retrieve_mean_s": round(statistics.mean(total), 3), "cross_encoder_mean_s": round(statistics.mean(ce), 3),
        "cross_encoder_share": round(sum(ce) / sum(total), 3),
        "calls_per_question": round(len(pairs_per_call) / len(rows), 2), "pairs_per_call_mean": round(statistics.mean(pairs_per_call), 1),
    }, indent=1))


def agreement(n, other="onnx"):
    """Same search pool, both re-rankers: how often the best passage and the set of passages sent to the model (final_k) agree."""
    from settings import get_settings
    from pipeline_config import get_preset
    from rag_pipeline import RAGPipeline
    from reranker import ReRanker

    cfg = get_settings()
    preset = get_preset(cfg.pipeline_preset)
    rag = RAGPipeline(preset)
    user = rag.policy.superuser()
    a = ReRanker(backend="torch", device="cpu")
    b = ReRanker(backend="onnx", device="cpu") if other == "onnx" else ReRanker(backend="torch", device="cuda")
    questions = [q["question"] for q in json.load(open(os.path.join(BACKEND, "eval_data", "squad_dataset.json"), encoding="utf-8"))][:n]
    top1 = same_set = 0
    diffs, pools = [], 0
    for q in questions:
        found = rag.store.search(q, user, k=preset.candidate_k, hybrid=preset.hybrid, hybrid_pool=preset.hybrid_pool)
        pool = [p["chunk"] for p in found]
        if not pool:
            continue
        pools += 1
        ra, rb = a.rerank(q, pool, top_k=len(pool)), b.rerank(q, pool, top_k=len(pool))
        top1 += ra[0]["original_index"] == rb[0]["original_index"]
        same_set += {r["original_index"] for r in ra[:preset.final_k]} == {r["original_index"] for r in rb[:preset.final_k]}
        sb = {r["original_index"]: r["score"] for r in rb}
        diffs += [abs(r["score"] - sb[r["original_index"]]) for r in ra]
    print(json.dumps({"other": other, "preset": preset.name, "questions": pools, "pool_size": preset.candidate_k, "final_k": preset.final_k,
                      "same_best_passage": round(top1 / pools, 3), "same_final_set": round(same_set / pools, 3),
                      "score_abs_diff_mean": round(statistics.mean(diffs), 3), "score_abs_diff_max": round(max(diffs), 3)}, indent=1))


if __name__ == "__main__":
    main()
