"""Print what the pipeline sees for a few multi-hop questions that fail: gold passages, selected passages, graph candidates.

    python -m evaluation.graph_inspect --dataset hotpot --n 4
"""
import argparse

from evaluation import metrics as M
from evaluation.datasets import load_dataset
from evaluation.generate import build_pipeline
from pipeline_config import get_preset
from retriever import load_chunks, retrieve_detailed


def short(t, n=26):
    return " ".join(t.split()[:n])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hotpot")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--skip", type=int, default=0)
    args = ap.parse_args()
    corpus, items = load_dataset(args.dataset)
    cfg0 = get_preset("hybrid_graph_chunks")
    rag, _, _ = build_pipeline(args.dataset, corpus, cfg0)
    cfg, kg, user = rag.config, rag.kg_builder, rag.policy.superuser()
    shown = 0
    skipped = 0
    for item in items:
        q, gold = item["question"], item["gold_passages"]
        d = retrieve_detailed(q, cfg.qdrant_path, user, rag.policy, candidate_k=20, final_k=3, reranker=rag.reranker,
                              encryptor=rag.encryptor, collection_name=cfg.collection, hybrid=True)
        final = d["results"]
        cover = M.coverage([r["chunk"] for r in final], gold)
        found = set().union(*cover) if cover else set()
        if len(found) == len(gold):
            continue
        if skipped < args.skip:
            skipped += 1
            continue
        print("=" * 100)
        print("Q:", q, "| answer:", item["ground_truth"])
        for i, g in enumerate(gold):
            print(f"  GOLD {i} {'(found)' if i in found else '(MISSING)'}: {short(g)}")
        for r in final:
            print(f"  SELECTED score {r['score']:.1f}: {short(r['chunk'])}")
        qkeys = kg.query_entity_keys(q)
        refs = [(r["doc_id"], r["chunk_index"]) for r in final if r.get("doc_id")]
        seeds = kg.entities_in_chunks(refs, user)
        print("  question entities:", qkeys)
        print("  entities of selected passages:", seeds[:25])
        missing_idx = [i for i in range(len(gold)) if i not in found]
        for m in missing_idx:
            hit = [(ref, ks) for ref, ks in kg.chunk_refs_for_entities(seeds + qkeys, user, 12).items()]
            texts = load_chunks([ref for ref, _ in hit], user, rag.policy, rag.encryptor, cfg.qdrant_path, cfg.collection)
            ks_of = {ref: ks for ref, ks in hit}
            for t in texts:
                if M.normalize_passage(gold[m]) in M.normalize_passage(t["chunk"]):
                    print(f"  >> graph DOES propose missing gold {m} via entities {ks_of[(t['doc_id'], t['chunk_index'])]}")
                    break
            else:
                print(f"  >> graph does NOT propose missing gold {m} ({len(hit)} candidates proposed)")
                print("     gold mentions:", short(gold[m], 40))
        shown += 1
        if shown >= args.n:
            break


if __name__ == "__main__":
    main()
