"""Fit the hop-aware ranker (graph_ranker.py) and measure it honestly with cross-validation by question.

    python -m evaluation.hop_tune --dataset hotpot            # computes and caches candidate features, then fits
    python -m evaluation.hop_tune --dataset hotpot --reuse    # refit from the cached features

For every question the pipeline's graph_candidates() yields each candidate passage (the search pool plus passages the graph
links in) with a cross-encoder score and graph features. A logistic regression learns which candidates are supporting passages;
the ranker then picks the top 3. The reported "both supporting passages in the top 3" is a 5-fold cross-validated number:
each question is scored by a model that never saw it. The weights written for graph_ranker.DEFAULT_WEIGHTS are fitted on all
questions, and the final confirmation is run once on a different benchmark (hotpot_b) that played no part in fitting.
Run in the worker container after `evaluation.generate --dataset hotpot --config hybrid_graph_chunks --reset-graph`.
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

import graph_ranker
from evaluation import metrics as M
from evaluation.datasets import load_dataset
from evaluation.generate import RESULTS_DIR, build_pipeline
from pipeline_config import get_preset
from retriever import retrieve_detailed

FEATURE_SETS = {
    "cross-encoder only": ["ce"],
    "cross-encoder + entity coverage": ["ce", "cover", "cover_frac"],
    "cross-encoder + coverage + pages": ["ce", "cover", "cover_frac", "page_q", "page_s"],
    "all features": graph_ranker.FEATURES,
}


def compute_features(args):
    corpus, items = load_dataset(args.dataset, args.limit)
    rag, _, _ = build_pipeline(args.dataset, corpus, get_preset(args.config))
    cfg, user = rag.config, rag.policy.superuser()
    out = []
    for n, item in enumerate(items, 1):
        d = retrieve_detailed(item["question"], cfg.qdrant_path, user, rag.policy, candidate_k=cfg.candidate_k, final_k=cfg.final_k,
                              reranker=None, encryptor=rag.encryptor, collection_name=cfg.collection,
                              hybrid=cfg.hybrid, hybrid_pool=cfg.hybrid_pool)
        cands, _ = rag.graph_candidates(item["question"], user, d)
        labels = [1 if M.coverage([c["chunk"]], item["gold_passages"])[0] else 0 for c in cands]
        out.append({"gold": item["gold_passages"], "cands": [{k: v for k, v in c.items() if k != "chunk"} | {"chunk": c["chunk"]} for c in cands],
                    "labels": labels})
        if n % 50 == 0:
            print(f"  features {n}/{len(items)}")
    return out


def matrix(rows, feats):
    X = np.array([[float(c.get(f, 0.0)) for f in feats] for r in rows for c in r["cands"]])
    y = np.array([l for r in rows for l in r["labels"]])
    return X, y


def full_support(rows, scorer):
    out = []
    for r in rows:
        scores = scorer(r)
        top = [r["cands"][i]["chunk"] for i in np.argsort(-scores)[:3]]
        m = M.retrieval_metrics(top, r["gold"], ks=(1, 3))
        out.append(m.get("full_support@3", m["recall@3"]))
    return float(np.mean(out))


def fit(rows, feats, C=1.0):
    X, y = matrix(rows, feats)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    model = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit((X - mu) / sd, y)
    w = model.coef_[0] / sd                      # weights in raw feature units
    return dict(zip(feats, map(float, w)))


def cv_score(rows, feats, folds=5, C=1.0):
    idx = np.arange(len(rows))
    out = []
    for k in range(folds):
        test = [rows[i] for i in idx if i % folds == k]
        train = [rows[i] for i in idx if i % folds != k]
        w = fit(train, feats, C)
        out.append(full_support(test, lambda r: np.array([graph_ranker.score(c, w) for c in r["cands"]])))
    return float(np.mean(out)), out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hotpot")
    ap.add_argument("--config", default="hybrid_graph_chunks")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--reuse", action="store_true", help="refit from the cached candidate features")
    args = ap.parse_args()
    if args.dataset == "squad":
        raise SystemExit("Use a dev dataset or hotpot.")
    cache = os.path.join(os.path.dirname(RESULTS_DIR), "..", "qdrant_eval", f"hop_features_{args.dataset}.json")
    if args.reuse and os.path.exists(cache):
        rows = json.load(open(cache, encoding="utf-8"))
    else:
        rows = compute_features(args)
        json.dump(rows, open(cache, "w", encoding="utf-8"))

    def reaches(r):
        cover = M.coverage([c["chunk"] for c in r["cands"]], r["gold"])
        return len(set().union(*cover) if cover else set()) == len(r["gold"])

    ceiling = float(np.mean([reaches(r) for r in rows]))
    summary = {"dataset": args.dataset, "n": len(rows), "ceiling_both_passages_among_candidates": ceiling, "cross_validated": {}}
    for name, feats in FEATURE_SETS.items():
        mean, per_fold = cv_score(rows, feats)
        summary["cross_validated"][name] = {"both_passages_in_top3": round(mean, 4), "folds": [round(x, 3) for x in per_fold]}
    summary["weights_fitted_on_all"] = fit(rows, graph_ranker.FEATURES)
    print(json.dumps(summary, indent=2))
    with open(os.path.join(RESULTS_DIR, f"hop_tune__{args.dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
