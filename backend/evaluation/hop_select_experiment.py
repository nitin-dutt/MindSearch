"""Which way of choosing the final passages finds the second hop? Offline comparison on one benchmark, judge-free.

    python -m evaluation.hop_select_experiment --dataset hotpot

The first-hop passage looks like the question; the second-hop passage does not (it matches the first passage). A cross-encoder
that scores each passage against the question alone therefore ranks it too low. This script retrieves once per question, then
compares ways of filling the final slots, e.g. scoring graph-linked passages against "question + the bridge entity" instead:

    base      top passages by cross-encoder score on the question (what the pipeline did before)
    keep2     keep the 2 best, fill the last slot with the best graph-linked passage scored on question + bridge entities
    keep1     keep the best, fill 2 slots the same way
    keep2_p   like keep2 but the query is the question + the best passage's text (instead of entity names)

Graph-linked = shares an entity with the seed passages (access-checked in Neo4j and Qdrant). Run in the worker container after
`evaluation.generate --dataset <d> --config hybrid_graph_chunks --reset-graph` has built the index and graph.
"""
import argparse
from collections import defaultdict

import numpy as np

from evaluation import metrics as M
from evaluation.datasets import load_dataset
from evaluation.generate import build_pipeline
from pipeline_config import get_preset
from retriever import load_chunks, retrieve_detailed


def ce(reranker, pairs):
    return list(map(float, reranker.model.predict([[q, c] for q, c in pairs]))) if pairs else []


def pick(base_sorted, extra, keep, k=3):
    """keep the first `keep` of base_sorted, then the best `k-keep` of extra (by its own score), then back-fill from base."""
    chosen = list(base_sorted[:keep])
    seen = {c["chunk"] for c in chosen}
    for c in sorted(extra, key=lambda x: -x["aug"]):
        if len(chosen) >= k:
            break
        if c["chunk"] not in seen:
            chosen.append(c)
            seen.add(c["chunk"])
    for c in base_sorted:
        if len(chosen) >= k:
            break
        if c["chunk"] not in seen:
            chosen.append(c)
            seen.add(c["chunk"])
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hotpot")
    ap.add_argument("--config", default="hybrid_graph_chunks")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-bridge", type=int, default=3)
    args = ap.parse_args()
    if args.dataset == "squad":
        raise SystemExit("Use a dev dataset or hotpot.")
    corpus, items = load_dataset(args.dataset, args.limit)
    rag, _, _ = build_pipeline(args.dataset, corpus, get_preset(args.config))
    cfg, kg, rr = rag.config, rag.kg_builder, rag.reranker
    user = rag.policy.superuser()
    rows = defaultdict(list)

    for item in items:
        q, gold = item["question"], item["gold_passages"]
        d = retrieve_detailed(q, cfg.qdrant_path, user, rag.policy, candidate_k=cfg.candidate_k, final_k=cfg.final_k,
                              reranker=None, encryptor=rag.encryptor, collection_name=cfg.collection,
                              hybrid=cfg.hybrid, hybrid_pool=cfg.hybrid_pool)
        pool, metas = d["pool"], d["pool_meta"]
        if not pool:
            continue
        scores = ce(rr, [(q, c) for c in pool])
        base = sorted(({"chunk": c, "score": s, **m} for c, s, m in zip(pool, scores, metas)), key=lambda x: -x["score"])

        def evaluate(name, chosen):
            rows[name].append(M.retrieval_metrics([c["chunk"] for c in chosen], gold, ks=(1, 3)))

        evaluate("base", base[:3])

        for label, seed_n in (("keep2", 2), ("keep1", 1)):
            keep = 3 - (1 if label == "keep2" else 2)
            seed_refs = [(c["doc_id"], c["chunk_index"]) for c in base[:max(seed_n, keep)] if c.get("doc_id") is not None]
            seeds = kg.entities_in_chunks(seed_refs, user)
            have = {(c.get("doc_id"), c.get("chunk_index")) for c in base[:keep]}
            cand = {ref: ks for ref, ks in kg.chunk_refs_for_entities(seeds, user, cfg.graph_hub_limit).items() if ref not in have}
            best = sorted(cand, key=lambda ref: (-len(cand[ref]), ref))[:cfg.graph_chunk_pool]
            texts = load_chunks(best, user, rag.policy, rag.encryptor, cfg.qdrant_path, cfg.collection)
            ref_of = {t["chunk"]: (t["doc_id"], t["chunk_index"]) for t in texts}
            aug_q = [(f"{q} {', '.join(cand[ref_of[t['chunk']]][:args.max_bridge])}", t["chunk"]) for t in texts]
            for t, s in zip(texts, ce(rr, aug_q)):
                t["aug"] = s
            evaluate(label, pick(base, texts, keep))
            if label == "keep2":
                top_text = " ".join(base[0]["chunk"].split()[:60])
                for t, s in zip(texts, ce(rr, [(f"{q} {top_text}", t["chunk"]) for t in texts])):
                    t["aug"] = s
                evaluate("keep2_p", pick(base, texts, keep))
        if len(rows["base"]) % 50 == 0:
            print(f"  {len(rows['base'])}/{len(items)}")

    print(f"\n=== {args.dataset}: selection policies, n={len(rows['base'])} ===")
    for name, rs in rows.items():
        fs = [r.get("full_support@3", r["recall@3"]) for r in rs]
        print(f"  {name:<8} both-passages-in-top-3 {np.mean(fs):.3f}   recall@3 {np.mean([r['recall@3'] for r in rs]):.3f}   "
              f"hit@1 {np.mean([r['hit@1'] for r in rs]):.3f}")


if __name__ == "__main__":
    main()
