"""Judge calibration (alignment_plan EVAL-14): how far can we trust the automatic (RAGAS) scores?

  python -m evaluation.calibration make-sheet --run-dir evaluation/results/<run_id>
      Writes calibration/label_sheet.md (for a human to fill in, judge scores HIDDEN) and
      calibration/judge_key.json (the judge's scores for the same items - do not open before labelling).

  python -m evaluation.calibration score
      Reads the filled sheet and compares the human labels with (a) the RAGAS faithfulness score and
      (b) the contains-gold string metric.

The human answers two questions per item:
  Correct   - is the answer a correct answer to the question? (equivalent wording is fine)
  Supported - does the passage shown state everything the answer claims? (this is what "faithfulness" tests)
"""
import argparse
import json
import os
import random
import re

CAL_DIR = os.path.join(os.path.dirname(__file__), "calibration")
SHEET = os.path.join(CAL_DIR, "label_sheet.md")
KEY = os.path.join(CAL_DIR, "judge_key.json")
SEED = 7


def supporting_passage(record):
    """The retrieved <doc> passage most likely to hold the answer (the one containing the gold string,
    else the top-ranked one), shown to the human instead of all chunks to keep labelling quick."""
    docs = [re.sub(r"^<doc>|</doc>$", "", c) for c in record["contexts"] if c.startswith("<doc>")]
    gold = record["ground_truth"].lower()
    for d in docs:
        if gold and gold in d.lower():
            return d
    return docs[0] if docs else "(no passage)"


def pick_items(ragas_rows, n=30, seed=SEED):
    """Stratified by judge faithfulness so the sample includes the judge's likely mistakes:
    all clearly-low scores, then a mix of partial and perfect scores."""
    rng = random.Random(seed)
    scored = [r for r in ragas_rows if r.get("faithfulness") is not None]
    low = [r for r in scored if r["faithfulness"] < 0.5]
    mid = [r for r in scored if 0.5 <= r["faithfulness"] < 0.999]
    top = [r for r in scored if r["faithfulness"] >= 0.999]
    rest = n - len(low)
    n_mid = rest // 2
    chosen = low + rng.sample(mid, min(n_mid, len(mid))) + rng.sample(top, min(rest - n_mid, len(top)))
    rng.shuffle(chosen)  # order must not reveal the judge's score
    return chosen[:n]


def make_sheet(run_dir, n):
    answers = {}
    with open(os.path.join(run_dir, "answers.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            answers[r["id"]] = r
    with open(os.path.join(run_dir, "ragas_per_question.jsonl"), encoding="utf-8") as f:
        ragas = [json.loads(line) for line in f]

    items = pick_items(ragas, n)
    os.makedirs(CAL_DIR, exist_ok=True)
    key = {}
    lines = [
        "# Judge calibration sheet",
        "",
        f"{len(items)} items. For each, fill in the two lines that end with a colon using **Y** or **N**.",
        "",
        "- **Correct**: is the model answer a correct answer to the question? Equivalent wording is fine "
        "(`$50,000` = `fifty thousand dollars`). The reference answer is only a guide.",
        "- **Supported**: does the passage shown say everything the model answer claims? If the answer adds "
        "something the passage does not state, mark **N** even if it happens to be true.",
        "",
        "Do not open `judge_key.json` until you are finished.",
        "",
    ]
    for i, rag in enumerate(items, start=1):
        rec = answers[rag["id"]]
        key[str(i)] = {"id": rag["id"], "faithfulness": rag.get("faithfulness"),
                       "answer_relevancy": rag.get("answer_relevancy"),
                       "contains_gold": rec.get("ans_contains_gold")}
        passage = " ".join(supporting_passage(rec).split())
        lines += [f"## {i}", "",
                  f"**Question:** {rec['question']}", "",
                  f"**Reference answer:** {rec['ground_truth']}", "",
                  f"**Model answer:** {rec['answer'].strip()}", "",
                  f"**Passage:** {passage}", "",
                  "- Correct (Y/N): ",
                  "- Supported (Y/N): ", "", "---", ""]
    with open(SHEET, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    with open(KEY, "w", encoding="utf-8") as f:
        json.dump({"run_dir": run_dir, "items": key}, f, indent=2)
    print(f"Wrote {SHEET} ({len(items)} items) and the hidden key {KEY}")


def parse_sheet(text):
    """Return {item_number: {"correct": bool|None, "supported": bool|None}} from a filled sheet."""
    out = {}
    parts = re.split(r"^## (\d+)\s*$", text, flags=re.M)
    for num, body in zip(parts[1::2], parts[2::2]):
        def label(name):
            m = re.search(rf"^- {name} \(Y/N\):[ \t]*([YyNn])", body, flags=re.M)
            return None if not m else m.group(1).upper() == "Y"
        out[int(num)] = {"correct": label("Correct"), "supported": label("Supported")}
    return out


def agreement(pairs):
    """pairs: list of (human_bool, judge_bool). Returns accuracy and the confusion counts."""
    tp = sum(1 for h, j in pairs if h and j)
    tn = sum(1 for h, j in pairs if not h and not j)
    fp = sum(1 for h, j in pairs if not h and j)   # judge says good, human says bad
    fn = sum(1 for h, j in pairs if h and not j)   # judge says bad, human says good
    n = len(pairs)
    return {"n": n, "agreement": (tp + tn) / n if n else None,
            "judge_ok_human_ok": tp, "judge_bad_human_bad": tn,
            "judge_ok_human_bad": fp, "judge_bad_human_ok": fn}


def score(skip_unfilled=False):
    with open(SHEET, encoding="utf-8") as f:
        labels = parse_sheet(f.read())
    with open(KEY, encoding="utf-8") as f:
        key = json.load(f)["items"]

    missing = [i for i, v in labels.items() if v["correct"] is None or v["supported"] is None]
    if missing and not skip_unfilled:
        raise SystemExit(f"Unfilled items: {sorted(missing)}. Fill every Correct/Supported line with Y or N "
                         f"(or pass --skip-unfilled for a provisional result).")
    labels = {i: v for i, v in labels.items() if i not in missing}
    key = {i: k for i, k in key.items() if int(i) in labels}

    faith_pairs = [(labels[int(i)]["supported"], k["faithfulness"] >= 0.999) for i, k in key.items()
                   if k["faithfulness"] is not None]
    gold_pairs = [(labels[int(i)]["correct"], bool(k["contains_gold"])) for i, k in key.items()
                  if k["contains_gold"] is not None]
    result = {"faithfulness_vs_human_supported": agreement(faith_pairs),
              "contains_gold_vs_human_correct": agreement(gold_pairs),
              "human_correct_rate": sum(v["correct"] for v in labels.values()) / len(labels),
              "human_supported_rate": sum(v["supported"] for v in labels.values()) / len(labels)}
    with open(os.path.join(CAL_DIR, "calibration_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make-sheet")
    mk.add_argument("--run-dir", required=True)
    mk.add_argument("--n", type=int, default=30)
    sc = sub.add_parser("score")
    sc.add_argument("--skip-unfilled", action="store_true", help="provisional result ignoring unfilled items")
    args = ap.parse_args()
    make_sheet(args.run_dir, args.n) if args.cmd == "make-sheet" else score(args.skip_unfilled)


if __name__ == "__main__":
    main()
