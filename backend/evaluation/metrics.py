"""Judge-free metrics and statistics for the evaluation harness (alignment_plan EVAL-08, EVAL-10).

Retrieval relevance is exact, not LLM-judged: a retrieved chunk "covers" a gold passage when the
normalized gold passage text appears inside the normalized chunk. This is sound because the chunker
only merges whole lines, and every gold passage is written to the corpus as exactly one line.
"""
import math
import random
import re
import string
from collections import Counter

# ---------------------------------------------------------------- retrieval

_WS = re.compile(r"\s+")


def normalize_passage(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


def coverage(ranked_chunks, gold_passages):
    """For each chunk (in rank order) return the set of gold-passage indices it contains."""
    gold_norm = [normalize_passage(g) for g in gold_passages]
    result = []
    for chunk in ranked_chunks:
        c = normalize_passage(chunk)
        result.append({i for i, g in enumerate(gold_norm) if g and g in c})
    return result


def _covered_within(cover, k):
    seen = set()
    for s in cover[:k]:
        seen |= s
    return seen


def hit_at_k(cover, k):
    """1.0 if any top-k chunk contains a gold passage."""
    return 1.0 if _covered_within(cover, k) else 0.0


def recall_at_k(cover, n_gold, k):
    """Fraction of the gold passages found in the top-k chunks."""
    if n_gold == 0:
        return None
    return len(_covered_within(cover, k)) / n_gold


def full_support_at_k(cover, n_gold, k):
    """1.0 if ALL gold passages are in the top-k (the multi-hop requirement)."""
    if n_gold == 0:
        return None
    return 1.0 if len(_covered_within(cover, k)) == n_gold else 0.0


def mrr(cover):
    """Reciprocal rank of the first chunk that contains any gold passage (0 if none)."""
    for rank, s in enumerate(cover, start=1):
        if s:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(cover, n_gold, k):
    """nDCG@k with binary gain: a chunk earns 1 only if it adds a gold passage not seen earlier."""
    if n_gold == 0:
        return None
    seen = set()
    dcg = 0.0
    for rank, s in enumerate(cover[:k], start=1):
        if s - seen:
            dcg += 1.0 / math.log2(rank + 1)
        seen |= s
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(n_gold, k) + 1))
    return dcg / ideal if ideal else None


def precision(cover):
    """Share of the returned chunks that contain a gold passage (exact context precision).
    None when nothing was returned."""
    if not cover:
        return None
    return sum(1 for s in cover if s) / len(cover)


def retrieval_metrics(ranked_chunks, gold_passages, ks=(1, 3, 5, 10, 20)):
    """All judge-free retrieval metrics for one ranked chunk list."""
    cover = coverage(ranked_chunks, gold_passages)
    n_gold = len(gold_passages)
    out = {"mrr": mrr(cover), "precision": precision(cover)}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(cover, k)
        out[f"recall@{k}"] = recall_at_k(cover, n_gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(cover, n_gold, k)
        if n_gold > 1:
            out[f"full_support@{k}"] = full_support_at_k(cover, n_gold, k)
    return out


# ------------------------------------------------------------------- answers

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation and articles, collapse spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def contains_gold(prediction: str, gold: str) -> float:
    """1.0 if the normalized gold answer appears inside the normalized prediction.
    The generator answers in full sentences, so this is the fairer string metric than EM."""
    g = normalize_answer(gold)
    return 1.0 if g and g in normalize_answer(prediction) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not p or not g:
        return 1.0 if p == g else 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, gold: str) -> float:
    """Share of gold-answer tokens present in the prediction."""
    p, g = Counter(normalize_answer(prediction).split()), normalize_answer(gold).split()
    if not g:
        return 0.0
    return sum(1 for t in g if p[t] > 0) / len(g)


def answer_metrics(prediction: str, gold: str):
    return {
        "exact_match": exact_match(prediction, gold),
        "contains_gold": contains_gold(prediction, gold),
        "token_f1": token_f1(prediction, gold),
        "token_recall": token_recall(prediction, gold),
    }


# ---------------------------------------------------------------- statistics

def _clean(values):
    return [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def bootstrap_ci(values, n_resamples=2000, alpha=0.05, seed=0):
    """Mean with a percentile bootstrap confidence interval. Returns (mean, lo, hi, n)."""
    vals = _clean(values)
    n = len(vals)
    if n == 0:
        return None, None, None, 0
    mean = sum(vals) / n
    if n == 1:
        return mean, mean, mean, 1
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_resamples))
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return mean, lo, hi, n


def paired_bootstrap_diff(a, b, n_resamples=2000, alpha=0.05, seed=0):
    """Mean(a - b) over questions both runs scored, with a bootstrap CI. Returns (diff, lo, hi, n).
    If the CI excludes 0 the difference is unlikely to be noise."""
    pairs = [(x, y) for x, y in zip(a, b) if _clean([x]) and _clean([y])]
    return bootstrap_ci([x - y for x, y in pairs], n_resamples, alpha, seed)


def percentile(values, pct):
    vals = sorted(_clean(values))
    if not vals:
        return None
    return vals[min(len(vals) - 1, int(len(vals) * pct))]


def summarize(rows, keys):
    """Mean + 95% CI for each metric key across a list of per-question dicts."""
    out = {}
    for key in keys:
        mean, lo, hi, n = bootstrap_ci([r.get(key) for r in rows])
        out[key] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": n}
    return out
