"""Upper bound for answer metrics: give the generator ONLY the gold passages and score its answers.

    python -m evaluation.oracle_ceiling --dataset hotpot

No retrieval can do better than perfect retrieval, so this is the ceiling for answer-level metrics (contains-gold, token F1)
with this generator and prompt. It tells us how much of the gap to a target is retrieval (fixable) and how much is the model
or the metric (not fixable by retrieval work). Run inside the worker container (needs the vLLM server).
"""
import argparse
import asyncio
import json

from evaluation import metrics as M
from evaluation.datasets import load_dataset
from llm import client, get_system_prompt
from settings import get_settings


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hotpot")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.dataset == "squad":
        raise SystemExit("Use a dev dataset or hotpot.")
    _, items = load_dataset(args.dataset, args.limit)
    model = get_settings().generator_model
    sem = asyncio.Semaphore(4)

    async def one(item):
        context = "\n\n".join(f"<doc>{g}</doc>" for g in item["gold_passages"])
        async with sem:
            r = await client.chat.completions.create(
                model=model, temperature=0.1, max_tokens=300,
                messages=[{"role": "system", "content": get_system_prompt()},
                          {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{item['question']}\n\nAnswer:"}])
        answer = r.choices[0].message.content or ""
        return {"contains": M.contains_gold(answer, item["ground_truth"]), "f1": M.token_f1(answer, item["ground_truth"]),
                "failed": not answer.strip()}

    rows = await asyncio.gather(*(one(i) for i in items))
    n = len(rows)
    out = {"dataset": args.dataset, "n": n, "model": model,
           "contains_gold": sum(r["contains"] for r in rows) / n, "token_f1": sum(r["f1"] for r in rows) / n,
           "failed": sum(r["failed"] for r in rows)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
