"""Evaluation pass 1: build a clean index, retrieve, generate, and score everything that needs no judge.

    python -m evaluation.generate --dataset squad --config dense_only
    python -m evaluation.generate --dataset hotpot --config full --reset-graph
    python -m evaluation.generate --dataset squad --config dense_rerank --no-generate   # retrieval only, no LLM needed

Run from backend/. Writes evaluation/results/<run_id>/{answers.jsonl, meta.json, summary.json}.
Pass 2 (evaluation.judge) adds the RAGAS scores, using a *different* model than the generator.

Isolation: the index lives in qdrant_eval/<dataset>/ with its own encryption key, never in
qdrant_db/. It is rebuilt only when the corpus, chunking or embedder changes (content hash).
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from evaluation import metrics as M
from evaluation.datasets import load_dataset, DATASETS
from pipeline_config import PRESETS, get_preset

BACKEND_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EVAL_INDEX_DIR = os.path.join(BACKEND_DIR, "qdrant_eval")

from access_policy import DocACL

EVAL_ACL = DocACL(min_rank=0)  # evaluation documents are readable by everyone; queries run as the superuser
MAX_KG_FAILURE_RATE = 0.25  # abort the run if more than this share of chunks fail extraction
POOL_KS = (1, 3, 5, 10, 20)
FINAL_KS = (1, 3)


# ------------------------------------------------------------------ index

def index_fingerprint(corpus_path, cfg):
    from embedder import EMBEDDING_MODEL_NAME, EMBEDDING_DIM, PAYLOAD_SCHEMA_VERSION
    h = hashlib.sha256()
    with open(corpus_path, "rb") as f:
        h.update(f.read())
    h.update(json.dumps({"chunk_size": cfg.chunk_size, "overlap": cfg.chunk_overlap,
                         "embedder": EMBEDDING_MODEL_NAME, "dim": EMBEDDING_DIM,
                         "payload_schema": PAYLOAD_SCHEMA_VERSION}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def build_pipeline(dataset, corpus_path, cfg, rebuild=False):
    """Return (pipeline, n_chunks, index_info) over a clean evaluation-only index."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from embedder import close_qdrant_client, get_qdrant_client
    from encryptor import Encryptor
    from rag_pipeline import RAGPipeline

    index_path = os.path.join(EVAL_INDEX_DIR, dataset)
    collection = f"eval_{dataset}"
    fp_file, key_file = os.path.join(index_path, ".fingerprint"), os.path.join(index_path, "eval.key")
    fingerprint = index_fingerprint(corpus_path, cfg)

    reuse = (not rebuild and os.path.exists(fp_file) and os.path.exists(key_file)
             and open(fp_file).read().strip() == fingerprint)
    if not reuse:
        close_qdrant_client(index_path)
        shutil.rmtree(index_path, ignore_errors=True)
        os.makedirs(index_path)
        with open(key_file, "wb") as f:
            f.write(AESGCM.generate_key(bit_length=256))

    with open(key_file, "rb") as f:
        encryptor = Encryptor(key=f.read())
    # the benchmark corpora hold one whole passage per line and gold passages must stay whole, so no sentence splitting here
    run_cfg = cfg.with_overrides(qdrant_path=index_path, collection=collection, split_long_paragraphs=False)
    rag = RAGPipeline(run_cfg, encryptor=encryptor)

    if not reuse:
        print(f"Building evaluation index for '{dataset}' (fingerprint {fingerprint})...")
        rag.ingest_vectors(corpus_path, EVAL_ACL)
        with open(fp_file, "w") as f:
            f.write(fingerprint)
    else:
        print(f"Reusing evaluation index for '{dataset}' (fingerprint {fingerprint}).")

    n_chunks = get_qdrant_client(index_path).count(collection_name=collection).count
    return rag, n_chunks, {"path": index_path, "collection": collection, "fingerprint": fingerprint, "reused": reuse}


def build_graph(rag, corpus_path):
    """Wipe Neo4j and rebuild the graph from the corpus, extracting chunk by chunk.

    Extractions are cached next to the evaluation index, so a crashed or repeated run resumes
    instead of paying the CPU cost again. Aborts if the graph is empty or too many chunks failed.
    """
    from chunker import run_chunking
    from kg_extractors import CachedExtractor
    cfg = rag.config
    with rag.kg_builder.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    # Outside the index folder on purpose: that folder is deleted whenever the index is rebuilt.
    rag.kg_builder.extractor = CachedExtractor(
        rag.kg_builder.extractor, os.path.join(EVAL_INDEX_DIR, "kg_cache", f"{cfg.collection}.jsonl"))
    chunks = run_chunking(corpus_path, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap, split_long=cfg.split_long_paragraphs)
    t0 = time.time()
    from doc_identity import compute_doc_id
    # the graph must use the same document id as the vector index: it is how a graph hit finds its passage
    stats = rag.ingest_graph(chunks, EVAL_ACL, doc_id=compute_doc_id(corpus_path, EVAL_ACL))
    graph = graph_stats(rag, build_seconds=round(time.time() - t0, 1))
    graph["extraction"] = stats.to_dict()
    if graph["entities"] == 0 or graph["relations"] == 0:
        sys.exit("ABORT: the graph build produced no entities/relations. Nothing was saved, because "
                 "this run would not measure the knowledge graph.")
    if stats.failure_rate > MAX_KG_FAILURE_RATE:
        sys.exit(f"ABORT: {stats.chunks_failed}/{stats.chunks_ok + stats.chunks_failed} chunks failed extraction "
                 f"({stats.failure_rate:.0%} > {MAX_KG_FAILURE_RATE:.0%}). The graph would not be representative.")
    return graph


def graph_stats(rag, build_seconds=None):
    with rag.kg_builder.driver.session() as session:
        nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        edges = session.run("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS c").single()["c"]
    return {"entities": nodes, "relations": edges, "build_seconds": build_seconds}


# --------------------------------------------------------------- retrieval

def retrieve_all(rag, items):
    records = []
    for n, item in enumerate(items, start=1):
        gold = item["gold_passages"]
        res = rag.retrieve(item["question"], rag.policy.superuser())
        final_texts = [r["chunk"] for r in res.chunks]

        rec = {
            "id": item["id"], "question": item["question"], "ground_truth": item["ground_truth"],
            "n_gold": len(gold), "type": item.get("type"),
            "contexts": res.to_context(),
            "n_context_items": len(res.chunks) + len(res.graph_facts),
            "context_words": sum(len(c.split()) for c in res.to_context()),
            "graph_facts": [r["chunk"] for r in res.graph_facts],
            "graph_pool_size": len(res.graph_pool),
            "graph_chunks_added": res.graph_chunks_added,
            "timings_ms": {k: round(v, 2) for k, v in res.timings_ms.items()},
            "final_covers": [sorted(s) for s in M.coverage(final_texts, gold)],
        }
        for k, v in M.retrieval_metrics(final_texts, gold, ks=FINAL_KS).items():
            rec[f"final_{k}"] = v
        for k, v in M.retrieval_metrics(res.dense_pool, gold, ks=POOL_KS).items():
            rec[f"pool_{k}"] = v
        records.append(rec)
        if n % 25 == 0 or n == len(items):
            print(f"  retrieved {n}/{len(items)}")
    return records


# -------------------------------------------------------------- generation

async def generate_all(rag, records, concurrency):
    from llm import LLM_ERROR_PREFIX
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(rec):
        nonlocal done
        if not rec["contexts"]:
            rec.update(answer="", failed=False, no_context=True, gen_ms=None, ttft_ms=None)
        else:
            async with sem:
                t0, first, parts = time.perf_counter(), None, []
                async for tok in rag.stream_answer(rec["question"], rec["contexts"]):
                    if first is None:
                        first = time.perf_counter() - t0
                    parts.append(tok)
                answer = "".join(parts)
                rec.update(answer=answer, failed=answer.startswith(LLM_ERROR_PREFIX) or not answer.strip(),
                           no_context=False, gen_ms=round((time.perf_counter() - t0) * 1000, 1),
                           ttft_ms=round(first * 1000, 1) if first else None)
        if not rec["failed"] and not rec["no_context"]:
            rec.update({f"ans_{k}": v for k, v in M.answer_metrics(rec["answer"], rec["ground_truth"]).items()})
        done += 1
        if done % 25 == 0 or done == len(records):
            print(f"  generated {done}/{len(records)}")

    await asyncio.gather(*(one(r) for r in records))


# ----------------------------------------------------------------- summary

def summarize_run(records, generated):
    keys = sorted({k for r in records for k in r if k.startswith(("final_", "pool_"))
                   and isinstance(r[k], (int, float)) and not isinstance(r[k], bool)})
    summary = {"n_questions": len(records), "retrieval": M.summarize(records, keys)}
    if any(r.get("n_gold", 0) > 1 for r in records):
        summary["by_type"] = {}
        for t in sorted({r["type"] for r in records if r.get("type")}):
            sub = [r for r in records if r.get("type") == t]
            summary["by_type"][t] = {"n": len(sub), **M.summarize(sub, [k for k in keys if "recall@" in k or "full_support@" in k])}

    summary["context"] = {"avg_items": sum(r["n_context_items"] for r in records) / len(records),
                          "avg_words": sum(r["context_words"] for r in records) / len(records)}
    summary["graph"] = {
        "questions_with_graph_facts": sum(1 for r in records if r["graph_facts"]) / len(records),
        "questions_with_raw_graph_pool": sum(1 for r in records if r["graph_pool_size"]) / len(records),
    }
    total_ms = [sum(r["timings_ms"].values()) for r in records]
    summary["latency_ms"] = {"retrieval_p50": M.percentile(total_ms, 0.5), "retrieval_p95": M.percentile(total_ms, 0.95)}

    if generated:
        scored = [r for r in records if not r["failed"] and not r["no_context"]]
        summary["generation"] = {
            "n_failed": sum(1 for r in records if r["failed"]),
            "n_no_context": sum(1 for r in records if r["no_context"]),
            "answers": M.summarize(scored, ["ans_exact_match", "ans_contains_gold", "ans_token_f1", "ans_token_recall"]),
            "ttft_ms_p50": M.percentile([r["ttft_ms"] for r in scored], 0.5),
            "gen_ms_p50": M.percentile([r["gen_ms"] for r in scored], 0.5),
        }
    return summary


def print_summary(summary, config_name, dataset):
    r = summary["retrieval"]

    def fmt(key):
        s = r.get(key)
        return "n/a" if not s or s["mean"] is None else f"{s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"

    print(f"\n==== {dataset} / {config_name} / n={summary['n_questions']} (mean [95% CI]) ====")
    print("Dense pool (before rerank):")
    for k in (1, 3, 5, 10, 20):
        print(f"  recall@{k:<2} {fmt(f'pool_recall@{k}')}   hit@{k:<2} {fmt(f'pool_hit@{k}')}")
    print(f"  MRR {fmt('pool_mrr')}   nDCG@10 {fmt('pool_ndcg@10')}")
    print("Final context (what the LLM sees):")
    print(f"  recall@3 {fmt('final_recall@3')}   hit@3 {fmt('final_hit@3')}   MRR {fmt('final_mrr')}")
    print(f"  context precision (exact) {fmt('final_precision')}   avg context words {summary['context']['avg_words']:.0f}")
    if "final_full_support@3" in r:
        print(f"  full-support@3 (all hops) {fmt('final_full_support@3')}")
    g = summary["graph"]
    print(f"Graph: {g['questions_with_graph_facts']:.0%} of questions got >=1 graph fact "
          f"({g['questions_with_raw_graph_pool']:.0%} had a non-empty raw pool)")
    print(f"Retrieval latency p50/p95: {summary['latency_ms']['retrieval_p50']:.0f} / {summary['latency_ms']['retrieval_p95']:.0f} ms")
    if "generation" in summary:
        a = summary["generation"]["answers"]
        print(f"Answers: contains-gold {a['ans_contains_gold']['mean']:.3f}  token-recall {a['ans_token_recall']['mean']:.3f}  "
              f"(failed {summary['generation']['n_failed']}, no-context {summary['generation']['n_no_context']})")


# -------------------------------------------------------------------- meta

def git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BACKEND_DIR, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=BACKEND_DIR, text=True, stderr=subprocess.DEVNULL).strip())
        return {"sha": sha, "dirty": dirty}
    except Exception:  # noqa: BLE001 - git is absent inside the container
        return {"sha": os.getenv("GIT_SHA", "unknown"), "dirty": None}


def library_versions():
    versions = {"python": platform.python_version()}
    for lib in ("sentence_transformers", "qdrant_client", "ragas", "openai"):
        try:
            versions[lib] = __import__(lib).__version__
        except Exception:  # noqa: BLE001
            versions[lib] = None
    return versions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--config", required=True, choices=list(PRESETS))
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke tests)")
    ap.add_argument("--candidate-k", type=int)
    ap.add_argument("--final-k", type=int)
    ap.add_argument("--concurrency", type=int, default=4, help="parallel LLM requests")
    ap.add_argument("--no-generate", action="store_true", help="retrieval metrics only; no LLM calls")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--reset-graph", action="store_true",
                    help="REQUIRED for graph configs: WIPES ALL Neo4j DATA and rebuilds the graph from the corpus")
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    cfg = get_preset(args.config)
    overrides = {k: v for k, v in (("candidate_k", args.candidate_k), ("final_k", args.final_k)) if v}
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    if cfg.use_graph and not args.reset_graph:
        sys.exit("Config '%s' uses the knowledge graph. Re-run with --reset-graph to confirm that ALL data in "
                 "the connected Neo4j database may be wiped and rebuilt from this corpus." % cfg.name)

    corpus_path, items = load_dataset(args.dataset, args.limit)
    print(f"Dataset {args.dataset}: {len(items)} questions | config {cfg.name}")

    rag, n_chunks, index_info = build_pipeline(args.dataset, corpus_path, cfg, rebuild=args.rebuild_index)
    graph = build_graph(rag, corpus_path) if cfg.use_graph else None
    if graph:
        print(f"Graph built: {graph}")

    print("Retrieving...")
    records = retrieve_all(rag, items)
    generated = not args.no_generate
    if generated:
        print("Generating answers...")
        asyncio.run(generate_all(rag, records, args.concurrency))

    summary = summarize_run(records, generated)
    print_summary(summary, cfg.name, args.dataset)

    from embedder import EMBEDDING_MODEL_NAME
    from llm import SYSTEM_PROMPT
    from settings import get_settings
    from reranker import RERANKER_MODEL_NAME
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{args.dataset}__{cfg.name}__{stamp}" + (f"__{args.tag}" if args.tag else "")
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir)

    meta = {
        "run_id": run_id, "created_utc": stamp, "dataset": args.dataset, "n_questions": len(items),
        "config": cfg.to_dict(), "queried_as": "superuser", "generated": generated,
        "models": {"embedder": EMBEDDING_MODEL_NAME, "reranker": RERANKER_MODEL_NAME if cfg.use_reranker else None,
                   "generator": cfg.generator_model if generated else None},
        "system_prompt_sha": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12],
        "index": {**index_info, "n_chunks": n_chunks}, "graph": graph,
        "git": git_info(), "versions": library_versions(), "command": " ".join(sys.argv),
        "settings": get_settings().describe(),     # every model name / URL / knob in effect (settings.py; contains no secrets)
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "answers.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved {run_dir}")


if __name__ == "__main__":
    main()
