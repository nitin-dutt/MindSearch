"""Turn result directories into the markdown tables used in the README.

    python -m evaluation.report evaluation/results/squad__dense_only__* evaluation/results/squad__full__* \
        --baseline squad__dense_only

Every number comes from summary.json / ragas_scores.json written by the harness. With --baseline, each
other run gets a paired-bootstrap difference per metric, so a gain is only claimed when its 95% CI excludes 0.
"""
import argparse
import glob
import json
import os

from evaluation import metrics as M

RETRIEVAL_COLS = [("pool_recall@20", "Dense recall@20"), ("final_recall@3", "Recall@3 (final)"),
                  ("final_mrr", "MRR (final)"), ("final_full_support@3", "Full-support@3")]
ANSWER_COLS = [("ans_contains_gold", "Contains-gold"), ("ans_token_recall", "Token recall")]
RAGAS_COLS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]


def load_run(path):
    out = {"dir": path, "name": os.path.basename(path.rstrip("/\\"))}
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        out["meta"] = json.load(f)
    with open(os.path.join(path, "summary.json"), encoding="utf-8") as f:
        out["summary"] = json.load(f)
    ragas = os.path.join(path, "ragas_scores.json")
    out["ragas"] = json.load(open(ragas, encoding="utf-8")) if os.path.exists(ragas) else None
    with open(os.path.join(path, "answers.jsonl"), encoding="utf-8") as f:
        out["records"] = {json.loads(line)["id"]: json.loads(line) for line in f}
    return out


def cell(stat):
    if not stat or stat.get("mean") is None:
        return "n/a"
    return f"{stat['mean']:.3f} ({stat['ci_low']:.3f}-{stat['ci_high']:.3f})"


def paired(run, base, key):
    ids = [i for i in run["records"] if i in base["records"]]
    a = [run["records"][i].get(key) for i in ids]
    b = [base["records"][i].get(key) for i in ids]
    diff, lo, hi, n = M.paired_bootstrap_diff(a, b)
    if diff is None:
        return "n/a"
    sig = "*" if (lo > 0 or hi < 0) else ""
    return f"{diff:+.3f} ({lo:+.3f} to {hi:+.3f}){sig}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="result directories (globs allowed)")
    ap.add_argument("--baseline", help="substring identifying the baseline run for paired differences")
    args = ap.parse_args()

    paths = sorted({p for pattern in args.run_dirs for p in glob.glob(pattern)})
    runs = [load_run(p) for p in paths if os.path.exists(os.path.join(p, "meta.json"))]
    if not runs:
        raise SystemExit("No result directories found.")

    print("### Retrieval (judge-free), mean (95% CI)\n")
    cols = RETRIEVAL_COLS
    print("| Run | n | " + " | ".join(c[1] for c in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for r in runs:
        ret = r["summary"]["retrieval"]
        print(f"| {r['name']} | {r['summary']['n_questions']} | " + " | ".join(cell(ret.get(k)) for k, _ in cols) + " |")

    if any("generation" in r["summary"] for r in runs):
        print("\n### Answers (string metrics, judge-free)\n")
        print("| Run | Failed | " + " | ".join(c[1] for c in ANSWER_COLS) + " |")
        print("|---|---|" + "---|" * len(ANSWER_COLS))
        for r in runs:
            g = r["summary"].get("generation")
            if g:
                print(f"| {r['name']} | {g['n_failed']} | " + " | ".join(cell(g["answers"].get(k)) for k, _ in ANSWER_COLS) + " |")

    judged = [r for r in runs if r["ragas"]]
    if judged:
        print("\n### RAGAS (LLM-judged), mean (95% CI)\n")
        print("| Run | Judge | " + " | ".join(RAGAS_COLS) + " |")
        print("|---|---|" + "---|" * len(RAGAS_COLS))
        for r in judged:
            rg = r["ragas"]
            judge = rg["judge_model"] + (" **(self-judged)**" if rg["self_judged"] else "")
            print(f"| {r['name']} | {judge} | " + " | ".join(cell(rg["metrics"].get(m)) for m in RAGAS_COLS) + " |")

    if args.baseline:
        base = next((r for r in runs if args.baseline in r["name"]), None)
        if base is None:
            raise SystemExit(f"No run matches baseline '{args.baseline}'.")
        keys = [k for k, _ in RETRIEVAL_COLS + ANSWER_COLS]
        print(f"\n### Paired difference vs {base['name']} (* = 95% CI excludes 0)\n")
        print("| Run | " + " | ".join(keys) + " |")
        print("|---|" + "---|" * len(keys))
        for r in runs:
            if r is not base:
                print(f"| {r['name']} | " + " | ".join(paired(r, base, k) for k in keys) + " |")


if __name__ == "__main__":
    main()
