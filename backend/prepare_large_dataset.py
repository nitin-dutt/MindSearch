"""Builds the SQuAD evaluation set: eval_data/squad_corpus.md + eval_data/squad_dataset.json.

Run from anywhere:
  python backend/prepare_large_dataset.py                                   # frozen TEST set: 100 questions
  python backend/prepare_large_dataset.py --name squad_dev --n 500 --skip 100   # DEV set: 400 NEW questions

The seed is fixed and random.sample's first picks do not depend on the sample size, so the first
100 questions of a 500-question sample are exactly the test set; --skip 100 drops them. Tune on dev,
report on test. Deterministic (fixed seed, stable corpus order). Each question stores its gold passage so
retrieval can be scored exactly, without an LLM judge.

Harder, more realistic corpus (alignment_plan RET-02/03): add --pad-corpus N to pad the corpus with every other validation
passage plus passages from the SQuAD train split until it holds about N passages. The questions are unchanged; only the number of
distractors grows, so retrieval differences between models become visible (a 700-passage corpus is almost too easy):
  python backend/prepare_large_dataset.py --name squad_dev_xl --n 500 --skip 100 --pad-corpus 10000

Note: this uses SQuAD v1.1 ("rajpurkar/squad"), where every question is answerable.
"""
import argparse
import json
import os
import random

from datasets import load_dataset

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data")


def one_line(text):
    """Corpus format is one passage per line; collapse any internal newlines."""
    return " ".join(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="squad", help="output file prefix (squad = test set)")
    ap.add_argument("--n", type=int, default=100, help="questions to sample before skipping")
    ap.add_argument("--skip", type=int, default=0, help="drop the first N sampled questions (the held-out test set)")
    ap.add_argument("--pad-corpus", type=int, default=0, help="grow the corpus to about this many passages with distractors")
    args = ap.parse_args()

    print("Loading SQuAD validation split from HuggingFace...")
    dataset = load_dataset("rajpurkar/squad", split="validation")

    random.seed(SEED)
    indices = random.sample(range(len(dataset)), args.n)[args.skip:]

    items = []
    corpus = []  # first-appearance order, de-duplicated -> deterministic file
    seen = set()
    for i in indices:
        row = dataset[i]
        ctx = one_line(row["context"])
        items.append({
            "id": row["id"],
            "question": row["question"],
            "ground_truth": row["answers"]["text"][0] if row["answers"]["text"] else "",
            "gold_passages": [ctx],
        })
        if ctx not in seen:
            seen.add(ctx)
            corpus.append(ctx)

    print(f"Extracted {len(items)} questions over {len(corpus)} unique passages.")
    if args.pad_corpus > len(corpus):
        for row in dataset:                              # every other validation passage first (stable order)
            ctx = one_line(row["context"])
            if ctx not in seen:
                seen.add(ctx)
                corpus.append(ctx)
        train_contexts = sorted({one_line(r["context"]) for r in load_dataset("rajpurkar/squad", split="train")} - seen)
        random.Random(SEED).shuffle(train_contexts)      # then seeded train passages up to the requested size
        corpus.extend(train_contexts[:max(0, args.pad_corpus - len(corpus))])
        print(f"Padded the corpus with distractors: {len(corpus)} passages in total.")
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, f"{args.name}_corpus.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# Stanford SQuAD Validation Corpus\n\n")
        for ctx in corpus:
            f.write(f"{ctx}\n\n")

    with open(os.path.join(OUT_DIR, f"{args.name}_dataset.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.name}_corpus.md and {args.name}_dataset.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
