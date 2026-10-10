"""Evaluation pass 2: RAGAS scoring of a finished generation run, with an independent judge model.

    python -m evaluation.judge --run-dir evaluation/results/<run_id> \
        --judge-url http://vllm_engine:8000/v1 --judge-model <model served by vLLM>

Why a separate pass: the 8 GB GPU cannot hold the generator and a second model together. Run
pass 1 with the generator served, then restart vLLM with the judge model (VLLM_MODEL in
docker-compose) and run this. Use a different model family than the generator; if they match,
the result is stamped self_judged=true and the report flags it.

Writes ragas_scores.json and ragas_per_question.jsonl into the run directory.
"""
import argparse
import json
import os

from evaluation import metrics as M

RAGAS_METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--judge-url", default=os.getenv("JUDGE_API_BASE", os.getenv("VLLM_API_BASE", "http://localhost:8001/v1")))
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--workers", type=int, default=4, help="RAGAS parallelism; 4 avoided GPU timeouts on 8 GB")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    with open(os.path.join(args.run_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if not meta.get("generated"):
        raise SystemExit("This run was retrieval-only (--no-generate); there are no answers to judge.")
    with open(os.path.join(args.run_dir, "answers.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    scorable = [r for r in records if not r["failed"] and not r["no_context"]]
    if args.limit:
        scorable = scorable[:args.limit]
    excluded = len(records) - len(scorable)
    print(f"Judging {len(scorable)} answers ({excluded} excluded: failed or no context).")

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in scorable],
        "answer": [r["answer"] for r in scorable],
        "contexts": [r["contexts"] for r in scorable],
        "ground_truth": [r["ground_truth"] for r in scorable],
    })
    judge_llm = ChatOpenAI(model=args.judge_model, openai_api_base=args.judge_url, openai_api_key="EMPTY",
                           max_tokens=1000, timeout=900, max_retries=5, temperature=0.0)
    judge_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    result = evaluate(dataset, metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
                      llm=judge_llm, embeddings=judge_embeddings, run_config=RunConfig(max_workers=args.workers))
    df = result.to_pandas()

    generator = (meta.get("models") or {}).get("generator")
    scores = {"judge_model": args.judge_model, "generator_model": generator,
              "self_judged": generator == args.judge_model,
              "judge_embeddings": "all-MiniLM-L6-v2", "n_scored": len(scorable), "n_excluded": excluded,
              "metrics": {}}
    per_question = [{"id": r["id"]} for r in scorable]
    for name in RAGAS_METRICS:
        if name not in df.columns:
            continue
        values = [None if v != v else float(v) for v in df[name].tolist()]  # NaN -> None
        mean, lo, hi, n = M.bootstrap_ci(values)
        scores["metrics"][name] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": n, "n_nan": len(values) - n}
        for row, v in zip(per_question, values):
            row[name] = v

    if scores["self_judged"]:
        print("WARNING: judge model == generator model. These scores are self-graded and optimistic.")
    with open(os.path.join(args.run_dir, "ragas_scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    with open(os.path.join(args.run_dir, "ragas_per_question.jsonl"), "w", encoding="utf-8") as f:
        for row in per_question:
            f.write(json.dumps(row) + "\n")

    for name, s in scores["metrics"].items():
        print(f"  {name:<18} {s['mean']:.4f}  [{s['ci_low']:.4f}, {s['ci_high']:.4f}]  (n={s['n']}, nan={s['n_nan']})")


if __name__ == "__main__":
    main()
