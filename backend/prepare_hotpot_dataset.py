"""Builds the multi-hop evaluation set from HotpotQA (distractor setting):
eval_data/hotpot_corpus.md + eval_data/hotpot_dataset.json.

Run from anywhere:  python backend/prepare_hotpot_dataset.py [--n 200]
A second, untouched set for confirming results after tuning on the first:
                    python backend/prepare_hotpot_dataset.py --name hotpot_b --skip 200 --n 100
(the questions are a fixed shuffle; --skip drops the first N, so hotpot_b never overlaps hotpot)

Every HotpotQA question comes with 10 paragraphs: 2 gold supporting paragraphs (the hops) and 8
distractors. The corpus is the union of those paragraphs for the sampled questions, so a question's
distractors are real competing text. A system answers well only if it retrieves BOTH gold paragraphs.
"""
import argparse
import json
import os
import random
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data")
SEED = 42
# Official HotpotQA dev (distractor) file; used if the HuggingFace loader is unavailable.
HOTPOT_DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def load_rows():
    """Yield dicts with question, answer, type, level, titles, sentences, supporting_titles."""
    try:
        from datasets import load_dataset
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        for r in ds:
            yield {
                "id": r["id"], "question": r["question"], "answer": r["answer"],
                "type": r["type"], "level": r["level"],
                "titles": r["context"]["title"], "sentences": r["context"]["sentences"],
                "supporting_titles": set(r["supporting_facts"]["title"]),
            }
        return
    except Exception as e:  # noqa: BLE001 - any loader failure falls back to the raw file
        print(f"HuggingFace loader failed ({type(e).__name__}: {e}); downloading raw HotpotQA dev file...")
    with urllib.request.urlopen(HOTPOT_DEV_URL) as resp:
        data = json.load(resp)
    for r in data:
        yield {
            "id": r["_id"], "question": r["question"], "answer": r["answer"],
            "type": r["type"], "level": r["level"],
            "titles": [t for t, _ in r["context"]], "sentences": [s for _, s in r["context"]],
            "supporting_titles": {t for t, _ in r["supporting_facts"]},
        }


def passage_line(title, sentences):
    return " ".join(f"{title}. {' '.join(s.strip() for s in sentences)}".split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of questions to sample")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N shuffled questions (used by the held-out set)")
    ap.add_argument("--name", default="hotpot", help="output file prefix")
    args = ap.parse_args()

    rows = list(load_rows())
    print(f"Loaded {len(rows)} HotpotQA validation questions.")
    random.Random(SEED).shuffle(rows)
    rows = rows[args.skip:args.skip + args.n]

    items, corpus, seen = [], [], set()
    for r in rows:
        gold = []
        for title, sents in zip(r["titles"], r["sentences"]):
            line = passage_line(title, sents)
            if line not in seen:
                seen.add(line)
                corpus.append(line)
            if title in r["supporting_titles"]:
                gold.append(line)
        items.append({
            "id": r["id"], "question": r["question"], "ground_truth": r["answer"],
            "gold_passages": gold, "type": r["type"], "level": r["level"],
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{args.name}_corpus.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# HotpotQA Distractor Validation Corpus\n\n")
        for line in corpus:
            f.write(f"{line}\n\n")
    with open(os.path.join(OUT_DIR, f"{args.name}_dataset.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    n_words = sum(len(c.split()) for c in corpus)
    print(f"Wrote {len(items)} questions over {len(corpus)} passages (~{n_words:,} words) to {OUT_DIR}")


if __name__ == "__main__":
    main()
