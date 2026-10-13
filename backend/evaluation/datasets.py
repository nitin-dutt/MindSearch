"""Dataset registry. Paths resolve relative to backend/eval_data so scripts work from any directory.

A dataset is two files produced by a prepare script:
  <name>_corpus.md      one passage per line; this is what gets ingested
  <name>_dataset.json   [{"id", "question", "ground_truth", "gold_passages": [<exact corpus lines>], ...}]
"""
import json
import os

EVAL_DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "eval_data"))

DATASETS = {
    "squad": {"corpus": "squad_corpus.md", "qa": "squad_dataset.json", "prepare": "prepare_large_dataset.py",
              "description": "SQuAD v1.1 validation sample (single-hop control)"},
    "squad_dev": {"corpus": "squad_dev_corpus.md", "qa": "squad_dev_dataset.json",
                  "prepare": "prepare_large_dataset.py --name squad_dev --n 500 --skip 100",
                  "description": "SQuAD v1.1 DEV set: 400 questions disjoint from the frozen test set, for tuning"},
    "squad_dev_xl": {"corpus": "squad_dev_xl_corpus.md", "qa": "squad_dev_xl_dataset.json",
                     "prepare": "prepare_large_dataset.py --name squad_dev_xl --n 500 --skip 100 --pad-corpus 10000",
                     "description": "The same 400 dev questions over a 10,000-passage corpus (many more distractors) for retrieval comparisons"},
    "hotpot_b": {"corpus": "hotpot_b_corpus.md", "qa": "hotpot_b_dataset.json",
                 "prepare": "prepare_hotpot_dataset.py --name hotpot_b --skip 200 --n 100",
                 "description": "100 further HotpotQA questions, untouched by any tuning: final confirmation of multi-hop results"},
    "hotpot": {"corpus": "hotpot_corpus.md", "qa": "hotpot_dataset.json", "prepare": "prepare_hotpot_dataset.py",
               "description": "HotpotQA distractor validation sample (multi-hop)"},
}


def dataset_paths(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(DATASETS)}")
    spec = DATASETS[name]
    return os.path.join(EVAL_DATA_DIR, spec["corpus"]), os.path.join(EVAL_DATA_DIR, spec["qa"])


def load_dataset(name, limit=None):
    corpus_path, qa_path = dataset_paths(name)
    for path in (corpus_path, qa_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Generate it with: python {DATASETS[name]['prepare']}")
    with open(qa_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    missing = [i for i, it in enumerate(items) if not it.get("gold_passages")]
    if missing:
        raise ValueError(
            f"{qa_path} has items without 'gold_passages' (e.g. index {missing[0]}). "
            f"Regenerate it with: python {DATASETS[name]['prepare']}")
    if limit:
        items = items[:limit]
    return corpus_path, items
