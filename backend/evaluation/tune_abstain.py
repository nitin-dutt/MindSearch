"""How well can retrieval alone tell "answerable" from "unanswerable"? Picks the refusal-gate threshold honestly (alignment_plan GEN-03).

    python -m evaluation.tune_abstain --dataset squad_dev_unans --config hybrid_rerank

No language model is involved. For every question it takes the cross-encoder score of the best retrieved passage and asks: is the answer
in the corpus? The gate refuses when that score is below a threshold, so a good threshold separates the two groups. Reports the AUC (the
chance a random answerable question scores higher than a random unanswerable one: 0.5 is useless, 1.0 is perfect), the trade-off curve,
and a split-half check: the threshold is chosen on half of the questions (even positions) and measured on the other half (odd), and the
other way round, so the reported numbers were never used to pick the threshold.

Rule for choosing the threshold: refuse as many unanswerable questions as possible while wrongly refusing at most --max-false-refusal
(default 5%) of the answerable ones: refusing a valid question is the costlier mistake.
The frozen test version (squad_unans) is refused here: report on it once, with the chosen threshold, via evaluation.generate.
"""
import argparse
import json
import os

import numpy as np

from evaluation.datasets import load_dataset
from evaluation.generate import RESULTS_DIR, build_pipeline
from pipeline_config import get_preset


def auc(pos, neg):
    """P(score of a random positive > score of a random negative), ties count half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def rates(threshold, answerable_scores, unanswerable_scores):
    """(share of unanswerable refused, share of answerable wrongly refused) for a gate that refuses below `threshold`."""
    return (float(np.mean(np.asarray(unanswerable_scores) < threshold)), float(np.mean(np.asarray(answerable_scores) < threshold)))


def choose_threshold(answerable_scores, unanswerable_scores, max_false_refusal):
    """Highest refusal of unanswerable questions subject to a cap on wrongly refused answerable ones."""
    candidates = sorted(set(answerable_scores) | set(unanswerable_scores))
    best = None
    for t in candidates:
        refused_unans, false_refusal = rates(t, answerable_scores, unanswerable_scores)
        if false_refusal <= max_false_refusal and (best is None or refused_unans > best[1] or (refused_unans == best[1] and t > best[0])):
            best = (t, refused_unans, false_refusal)
    return best[0] if best else min(candidates)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="squad_dev_unans")
    ap.add_argument("--config", default="hybrid_rerank")
    ap.add_argument("--max-false-refusal", type=float, default=0.05)
    args = ap.parse_args()
    if args.dataset == "squad_unans":
        raise SystemExit("squad_unans is the frozen test set: tune on squad_dev_unans, then report once with evaluation.generate.")

    corpus, items = load_dataset(args.dataset)
    rag, _, _ = build_pipeline(args.dataset, corpus, get_preset(args.config))
    user = rag.policy.superuser()
    rows = []
    for n, item in enumerate(items, 1):
        res = rag.retrieve(item["question"], user)
        top = res.top_relevance()
        rows.append({"answerable": item["answerable"], "score": float(top) if top is not None else -99.0})
        if n % 100 == 0:
            print(f"  scored {n}/{len(items)}")

    ans = [r["score"] for r in rows if r["answerable"]]
    unans = [r["score"] for r in rows if not r["answerable"]]
    summary = {"dataset": args.dataset, "config": args.config, "n_answerable": len(ans), "n_unanswerable": len(unans),
               "median_best_score": {"answerable": float(np.median(ans)), "unanswerable": float(np.median(unans))},
               "auc": auc(ans, unans), "max_false_refusal": args.max_false_refusal}

    grid = np.round(np.arange(-8, 8.5, 1.0), 1)
    summary["curve"] = [{"threshold": float(t), "unanswerable_refused": round(rates(t, ans, unans)[0], 3),
                         "answerable_wrongly_refused": round(rates(t, ans, unans)[1], 3)} for t in grid]

    folds = {}
    for name, (fit_idx, test_idx) in (("chosen on even, measured on odd", (0, 1)), ("chosen on odd, measured on even", (1, 0))):
        def split(parity):
            sub = [r for i, r in enumerate(rows) if i % 2 == parity]
            return [r["score"] for r in sub if r["answerable"]], [r["score"] for r in sub if not r["answerable"]]
        fa, fu = split(fit_idx)
        ta, tu = split(test_idx)
        t = choose_threshold(fa, fu, args.max_false_refusal)
        folds[name] = {"threshold": round(t, 3), "on_chosen_half": dict(zip(("unanswerable_refused", "answerable_wrongly_refused"), map(lambda x: round(x, 3), rates(t, fa, fu)))),
                       "on_other_half": dict(zip(("unanswerable_refused", "answerable_wrongly_refused"), map(lambda x: round(x, 3), rates(t, ta, tu))))}
    summary["split_half"] = folds
    summary["threshold_on_all"] = round(choose_threshold(ans, unans, args.max_false_refusal), 3)

    print(json.dumps(summary, indent=2))
    with open(os.path.join(RESULTS_DIR, f"tune_abstain__{args.dataset}__{args.config}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
