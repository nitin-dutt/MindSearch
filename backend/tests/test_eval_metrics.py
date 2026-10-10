import math

import pytest

from evaluation import metrics as M

GOLD_A = "Super Bowl 50 was held at Levi's Stadium."
GOLD_B = "Levi's Stadium is in Santa Clara, California."
NOISE = "Completely unrelated text about photosynthesis."


def test_coverage_ignores_case_and_whitespace():
    chunks = ["intro\nsuper  bowl 50 was held\nat LEVI'S stadium.  trailing", NOISE]
    assert M.coverage(chunks, [GOLD_A]) == [{0}, set()]


def test_single_gold_ranking_metrics():
    cover = M.coverage([NOISE, GOLD_A, NOISE], [GOLD_A])
    assert M.hit_at_k(cover, 1) == 0.0
    assert M.hit_at_k(cover, 2) == 1.0
    assert M.recall_at_k(cover, 1, 1) == 0.0
    assert M.recall_at_k(cover, 1, 3) == 1.0
    assert M.mrr(cover) == pytest.approx(0.5)
    assert M.ndcg_at_k(cover, 1, 3) == pytest.approx(1 / math.log2(3))


def test_multi_hop_recall_and_full_support():
    cover = M.coverage([GOLD_A, NOISE, GOLD_B], [GOLD_A, GOLD_B])
    assert M.recall_at_k(cover, 2, 1) == 0.5
    assert M.full_support_at_k(cover, 2, 1) == 0.0
    assert M.full_support_at_k(cover, 2, 3) == 1.0
    # ideal ranking puts both gold first -> nDCG 1.0; ours has the second hop at rank 3
    ideal = 1 + 1 / math.log2(3)
    assert M.ndcg_at_k(cover, 2, 3) == pytest.approx((1 + 1 / math.log2(4)) / ideal)


def test_one_chunk_covering_both_gold_passages():
    merged = f"{GOLD_A}\n{GOLD_B}"
    cover = M.coverage([merged], [GOLD_A, GOLD_B])
    assert M.full_support_at_k(cover, 2, 1) == 1.0


def test_duplicate_chunks_do_not_inflate_ndcg():
    cover = M.coverage([GOLD_A, GOLD_A], [GOLD_A])
    assert M.ndcg_at_k(cover, 1, 2) == pytest.approx(1.0)


def test_no_gold_found():
    cover = M.coverage([NOISE], [GOLD_A])
    assert M.mrr(cover) == 0.0
    assert M.hit_at_k(cover, 5) == 0.0
    assert M.ndcg_at_k(cover, 1, 5) == 0.0


def test_empty_retrieval_is_zero_not_error():
    m = M.retrieval_metrics([], [GOLD_A], ks=(1, 3))
    assert m["mrr"] == 0.0 and m["recall@3"] == 0.0 and m["hit@1"] == 0.0


def test_full_support_key_only_for_multi_hop():
    assert "full_support@3" not in M.retrieval_metrics([GOLD_A], [GOLD_A], ks=(3,))
    assert "full_support@3" in M.retrieval_metrics([GOLD_A], [GOLD_A, GOLD_B], ks=(3,))


def test_answer_normalization_and_em():
    assert M.exact_match("The Denver Broncos.", "denver broncos") == 1.0
    assert M.exact_match("Denver", "Carolina") == 0.0


def test_contains_gold_handles_sentence_answers():
    assert M.contains_gold("The jersey color was white.", "white") == 1.0
    assert M.contains_gold("It was blue.", "white") == 0.0
    assert M.contains_gold("anything", "") == 0.0


def test_token_f1_and_recall():
    assert M.token_f1("alpha beta", "alpha beta") == 1.0
    assert M.token_f1("alpha", "beta") == 0.0
    assert M.token_f1("alpha beta gamma", "alpha beta") == pytest.approx(0.8)
    assert M.token_recall("alpha beta gamma delta", "alpha beta") == 1.0


def test_bootstrap_ci_brackets_mean_and_ignores_missing():
    vals = [0.0, 1.0] * 50 + [None, float("nan")]
    mean, lo, hi, n = M.bootstrap_ci(vals, seed=1)
    assert n == 100 and mean == pytest.approx(0.5)
    assert lo < 0.5 < hi and hi - lo < 0.25


def test_bootstrap_ci_is_deterministic_and_handles_tiny_input():
    assert M.bootstrap_ci([0.2, 0.8, 0.5], seed=7) == M.bootstrap_ci([0.2, 0.8, 0.5], seed=7)
    assert M.bootstrap_ci([]) == (None, None, None, 0)
    assert M.bootstrap_ci([0.4]) == (0.4, 0.4, 0.4, 1)


def test_paired_difference_detects_a_real_gap_and_skips_missing():
    a = [1.0] * 60 + [None]
    b = [0.0] * 60 + [1.0]
    diff, lo, hi, n = M.paired_bootstrap_diff(a, b)
    assert n == 60 and diff == 1.0 and lo > 0
