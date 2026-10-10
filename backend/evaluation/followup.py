"""Does rewriting follow-up questions fix retrieval? (alignment_plan INF-06). Judge-free.

    python -m evaluation.followup make --dataset squad_dev --n 400     # writes eval_data/<dataset>_followups.json
    python -m evaluation.followup run  --dataset squad_dev             # retrieval metrics for 3 kinds of query

`make` asks the generator model to turn each standalone question into a chat: an earlier question + answer that introduces
the subject, and a follow-up that refers to the subject only by pronoun or ellipsis ("what about its price?").
`run` then retrieves with (a) the original standalone question (upper bound), (b) the raw follow-up (what happens with no
rewriting) and (c) the follow-up rewritten from the conversation history, and compares exact retrieval metrics with
paired bootstrap confidence intervals.

Caveat: the same model writes the test follow-ups and performs the rewrite, so a shared blind spot could flatter the
result; the examples file lets a human spot-check quality.
Run inside the worker container (it needs the vLLM server).
"""
import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone

from evaluation import metrics as M
from evaluation.datasets import EVAL_DATA_DIR, load_dataset
from evaluation.generate import RESULTS_DIR, build_pipeline
from pipeline_config import get_preset

SCHEMA = {"type": "object", "required": ["previous_question", "previous_answer", "follow_up"],
          "properties": {"previous_question": {"type": "string"}, "previous_answer": {"type": "string"},
                         "follow_up": {"type": "string"}}}

SYSTEM = (
    "You create test data for a chat system. You are given a question about a document and its answer. Write:\n"
    "- previous_question: a short, natural question a person might have asked EARLIER in the same chat that introduces the "
    "subject of the given question (the main person, place, thing or topic), WITHOUT asking the given question itself.\n"
    "- previous_answer: a one-sentence answer to previous_question.\n"
    "- follow_up: the given question rewritten the way a person would ask it as a FOLLOW-UP in that chat. Refer to the subject "
    "only by a pronoun or ellipsis (it, they, he, she, that, 'what about', 'and when'), and never name it. Keep the same meaning.\n"
    "Return JSON with the three fields.")


def followups_path(dataset):
    return os.path.join(EVAL_DATA_DIR, f"{dataset}_followups.json")


async def make(args):
    from llm import client
    _, items = load_dataset(args.dataset, args.n)
    model = get_preset("dense_rerank").generator_model
    sem = asyncio.Semaphore(8)
    out = []

    async def one(item):
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=0.3, max_tokens=300,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": f"Question: {item['question']}\nAnswer: {item['ground_truth']}"}],
                    extra_body={"guided_json": SCHEMA})
                data = json.loads(r.choices[0].message.content)
            except Exception as e:  # noqa: BLE001
                print(f"  skipped {item['id']}: {type(e).__name__}")
                return
            if data["follow_up"].strip().lower() == item["question"].strip().lower():
                return                                  # not actually a follow-up
            out.append({"id": item["id"], "question": item["question"], "ground_truth": item["ground_truth"],
                        "gold_passages": item["gold_passages"], **{k: data[k].strip() for k in SCHEMA["required"]}})

    await asyncio.gather(*(one(i) for i in items))
    out.sort(key=lambda r: r["id"])
    with open(followups_path(args.dataset), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} follow-up conversations to {followups_path(args.dataset)}\nExamples:")
    for r in out[:5]:
        print(f"  earlier : {r['previous_question']}\n  follow-up: {r['follow_up']}\n  original : {r['question']}\n")


def retrieve_texts(rag, query):
    return [r["chunk"] for r in rag.retrieve(query, rag.policy.superuser()).chunks]


async def run(args):
    from conversation import QueryRewriter
    from llm import complete_chat
    with open(followups_path(args.dataset), encoding="utf-8") as f:
        items = json.load(f)
    corpus, _ = load_dataset(args.dataset, 1)
    cfg = get_preset(args.config)
    rag, _, _ = build_pipeline(args.dataset, corpus, cfg)
    rewriter = QueryRewriter(cfg.generator_model, complete_chat, timeout=30)

    rows, t0 = [], time.time()
    for n, it in enumerate(items, start=1):
        history = [{"role": "user", "content": it["previous_question"]}, {"role": "assistant", "content": it["previous_answer"]}]
        rewritten, changed = await rewriter.rewrite(history, it["follow_up"])
        row = {"id": it["id"], "follow_up": it["follow_up"], "rewritten": rewritten, "changed": changed, "original": it["question"]}
        for name, query in (("original", it["question"]), ("raw", it["follow_up"]), ("rewritten", rewritten)):
            for k, v in M.retrieval_metrics(retrieve_texts(rag, query), it["gold_passages"], ks=(1, 3)).items():
                row[f"{name}_{k}"] = v
        rows.append(row)
        if n % 50 == 0:
            print(f"  {n}/{len(items)} ({time.time() - t0:.0f}s)")

    keys = ["recall@3", "hit@3", "mrr"]
    summary = {"n": len(rows), "config": args.config, "dataset": args.dataset, "rewriter_changed_the_question": sum(r["changed"] for r in rows) / len(rows)}
    for name in ("original", "raw", "rewritten"):
        summary[name] = M.summarize(rows, [f"{name}_{k}" for k in keys])
    summary["paired"] = {}
    for a, b in (("rewritten", "raw"), ("original", "rewritten")):
        for k in keys:
            d, lo, hi, n = M.paired_bootstrap_diff([r[f"{a}_{k}"] for r in rows], [r[f"{b}_{k}"] for r in rows])
            summary["paired"][f"{a} minus {b}: {k}"] = {"diff": d, "ci_low": lo, "ci_high": hi, "n": n}

    run_dir = os.path.join(RESULTS_DIR, f"followup__{args.dataset}__{args.config}__{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "examples.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("follow_up", "rewritten", "changed", "original")}, ensure_ascii=False) + "\n")

    print(f"\n=== follow-up retrieval, {args.dataset}, config {args.config}, n={len(rows)} (mean [95% CI]) ===")
    print(f"rewriter changed the question in {summary['rewriter_changed_the_question']:.0%} of cases")
    for name, label in (("original", "original standalone question (upper bound)"), ("raw", "raw follow-up, no rewriting"), ("rewritten", "follow-up rewritten from history")):
        s = summary[name]
        print(f"  {label:<46} " + "  ".join(f"{k} {s[f'{name}_{k}']['mean']:.3f} [{s[f'{name}_{k}']['ci_low']:.3f}, {s[f'{name}_{k}']['ci_high']:.3f}]" for k in keys))
    for key, v in summary["paired"].items():
        print(f"  {key:<34} {v['diff']:+.3f} ({v['ci_low']:+.3f} to {v['ci_high']:+.3f}){' *' if v['ci_low'] > 0 or v['ci_high'] < 0 else ''}")
    print(f"\nSaved {run_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--dataset", default="squad_dev")
    mk.add_argument("--n", type=int, default=400)
    rn = sub.add_parser("run")
    rn.add_argument("--dataset", default="squad_dev")
    rn.add_argument("--config", default="dense_rerank_fast")
    args = ap.parse_args()
    if args.dataset == "squad":
        raise SystemExit("Use squad_dev (the frozen test set is for final reporting only).")
    asyncio.run(make(args) if args.cmd == "make" else run(args))


if __name__ == "__main__":
    main()
