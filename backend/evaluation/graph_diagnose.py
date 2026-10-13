"""Where does graph-guided retrieval lose the second hop? Judge-free, stage by stage.

    python -m evaluation.graph_diagnose --dataset hotpot --config hybrid_graph_chunks

For every question whose final top passages do not contain ALL supporting passages, it traces: were seed entities found,
did the graph propose candidate passages, was a missing supporting passage among those candidates, and where did the
cross-encoder rank it against the passages already selected? The Neo4j graph and the evaluation index must already exist
(run evaluation.generate with --reset-graph for the same dataset first). Run inside the worker container.
"""
import argparse
from collections import Counter

from evaluation import metrics as M
from evaluation.datasets import load_dataset
from evaluation.generate import build_pipeline
from pipeline_config import get_preset
from retriever import load_chunks, retrieve_detailed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hotpot")
    ap.add_argument("--config", default="hybrid_graph_chunks")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.dataset == "squad":
        raise SystemExit("Use a dev dataset or hotpot.")
    corpus, items = load_dataset(args.dataset, args.limit)
    cfg = get_preset(args.config)
    rag, _, _ = build_pipeline(args.dataset, corpus, cfg)
    cfg = rag.config                       # the evaluation index and collection, not the preset's defaults
    kg, user = rag.kg_builder, rag.policy.superuser()
    c = Counter()
    ranks = []
    for item in items:
        gold = item["gold_passages"]
        d = retrieve_detailed(item["question"], cfg.qdrant_path, user, rag.policy, candidate_k=cfg.candidate_k, final_k=cfg.final_k,
                              reranker=rag.reranker, encryptor=rag.encryptor, collection_name=cfg.collection,
                              rerank_margin=cfg.rerank_margin, hybrid=cfg.hybrid, hybrid_pool=cfg.hybrid_pool)
        final = d["results"]
        cover = M.coverage([r["chunk"] for r in final], gold)
        found = set().union(*cover) if cover else set()
        c["questions"] += 1
        if len(found) == len(gold):
            c["already_fully_supported"] += 1
            continue
        c["need_second_hop"] += 1
        missing = [i for i in range(len(gold)) if i not in found]
        pool_cover = M.coverage(d["pool"], gold)
        pool_found = set().union(*pool_cover) if pool_cover else set()
        if all(i in pool_found for i in missing):
            c["missing_passage_is_in_the_20_pool"] += 1
        seed_refs = [(r["doc_id"], r["chunk_index"]) for r in final[:cfg.graph_seed_k] if r.get("doc_id") is not None]
        seeds = kg.entities_in_chunks(seed_refs, user)
        qkeys = kg.query_entity_keys(item["question"])
        if seeds:
            c["seed_entities_from_top_passages"] += 1
        if qkeys:
            c["question_entities_found"] += 1
        keys = list(dict.fromkeys(seeds + qkeys))
        have = {(r.get("doc_id"), r.get("chunk_index")) for r in final}
        cand = {ref: k for ref, k in kg.chunk_refs_for_entities(keys, user, cfg.graph_hub_limit).items() if ref not in have}
        if cand:
            c["graph_proposed_candidates"] += 1
        best = sorted(cand, key=lambda ref: (-len(cand[ref]), ref))[:cfg.graph_chunk_pool]
        texts = load_chunks(best, user, rag.policy, rag.encryptor, cfg.qdrant_path, cfg.collection)
        cov = M.coverage([t["chunk"] for t in texts], gold)
        if any(i in set().union(*cov) for i in missing) if cov else False:
            c["missing_passage_among_proposed_candidates"] += 1
            scored = rag.reranker.rerank(item["question"], [t["chunk"] for t in texts], top_k=len(texts))
            order = [texts[s["original_index"]]["chunk"] for s in scored]
            rank = next(i for i, ch in enumerate(order, 1) if any(M.normalize_passage(gold[m]) in M.normalize_passage(ch) for m in missing))
            ranks.append((rank, scored[rank - 1]["score"], final[-1]["score"] if final else None))
            if scored[rank - 1]["score"] > (final[-1]["score"] if final else -99):
                c["would_beat_the_weakest_selected_passage"] += 1
    n = c["need_second_hop"]
    print(f"\n=== {args.dataset} / {args.config}: {c['questions']} questions, {c['already_fully_supported']} already fully supported, {n} need a second hop ===")
    for k in ("missing_passage_is_in_the_20_pool", "question_entities_found", "seed_entities_from_top_passages", "graph_proposed_candidates",
              "missing_passage_among_proposed_candidates", "would_beat_the_weakest_selected_passage"):
        print(f"  {k:<46} {c[k]:>4} / {n}  ({c[k] / max(n, 1):.0%})")
    if ranks:
        print("  rank of the missing passage among graph candidates (cross-encoder):", Counter(min(r, 6) for r, _, _ in ranks))


if __name__ == "__main__":
    main()
