"""Builds an UNANSWERABLE-question benchmark from an existing one (alignment_plan GEN-03).

    python backend/prepare_unanswerable.py --source squad_dev --name squad_dev_unans      # for tuning
    python backend/prepare_unanswerable.py --source squad     --name squad_unans          # from the frozen test set: report once

For about half of the passages that hold an answer, the passage is REMOVED from the corpus. The questions about a removed passage become
unanswerable: the documents no longer contain their answer, but related passages (same article, same topic) are still there, so a system
that answers from loosely related text gets caught. The remaining questions stay answerable. Which passages are removed is a seeded shuffle,
so the files are reproducible.

Output: eval_data/<name>_corpus.md and eval_data/<name>_dataset.json, where every item has "answerable": true/false (unanswerable items keep
their original gold passage for reference; it is simply not in the corpus any more).
"""
import argparse
import json
import os
import random

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data")
SEED = 42


def split_unanswerable(items, fraction=0.5, seed=SEED):
    """Pick which gold passages to remove. Returns (items with an "answerable" flag, set of removed passages)."""
    passages = sorted({g for it in items for g in it["gold_passages"]})
    random.Random(seed).shuffle(passages)
    removed = set(passages[: round(len(passages) * fraction)])
    out = []
    for it in items:
        gone = all(g in removed for g in it["gold_passages"])
        out.append({**it, "answerable": not gone})
    return out, removed


def corpus_without(corpus_text, removed):
    """The corpus file text with the removed passages (whole lines) taken out; the header line is kept."""
    kept = [line for line in corpus_text.split("\n") if line.strip() not in removed]
    return "\n".join(kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="dataset name to derive from (squad_dev, squad, ...)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--fraction", type=float, default=0.5)
    args = ap.parse_args()
    src_corpus = os.path.join(OUT_DIR, f"{args.source}_corpus.md")
    src_qa = os.path.join(OUT_DIR, f"{args.source}_dataset.json")
    with open(src_qa, encoding="utf-8") as f:
        items = json.load(f)
    with open(src_corpus, encoding="utf-8") as f:
        corpus = f.read()

    new_items, removed = split_unanswerable(items, args.fraction)
    n_un = sum(1 for it in new_items if not it["answerable"])
    with open(os.path.join(OUT_DIR, f"{args.name}_corpus.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(corpus_without(corpus, removed))
    with open(os.path.join(OUT_DIR, f"{args.name}_dataset.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(new_items, f, indent=2, ensure_ascii=False)
    print(f"{args.name}: {len(new_items) - n_un} answerable + {n_un} unanswerable questions; removed {len(removed)} passages from the corpus")


if __name__ == "__main__":
    main()
