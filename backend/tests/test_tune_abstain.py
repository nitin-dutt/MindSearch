"""The threshold-tuning maths for the refusal gate."""
import pytest

pytest.importorskip("numpy")

from evaluation.tune_abstain import auc, choose_threshold, rates  # noqa: E402


def test_auc_is_one_for_perfect_separation_half_for_no_signal_and_counts_ties_half():
    assert auc([5, 6, 7], [-1, 0, 1]) == 1.0
    assert auc([1, 2], [1, 2]) == 0.5
    assert auc([1], [1]) == 0.5


def test_rates_measure_both_kinds_of_mistake():
    refused_unans, false_refusal = rates(0.0, answerable_scores=[3, 2, -1, 4], unanswerable_scores=[-5, -2, 1, -3])
    assert refused_unans == 0.75 and false_refusal == 0.25


def test_the_threshold_refuses_as_much_as_it_can_within_the_false_refusal_cap():
    answerable = [5, 4, 3, 2, 1, 0, -1, -2, 6, 7]                    # 10 valid questions
    unanswerable = [-6, -5, -4, -3, -2, 0.5, 1.5, -7, -8, -9]
    t = choose_threshold(answerable, unanswerable, max_false_refusal=0.1)     # at most 1 valid question may be refused
    refused, wrong = rates(t, answerable, unanswerable)
    assert wrong <= 0.1 and refused >= 0.8


def test_a_zero_cap_never_refuses_a_valid_question():
    answerable, unanswerable = [2, 3, 4], [-3, 0, 5]
    t = choose_threshold(answerable, unanswerable, max_false_refusal=0.0)
    assert rates(t, answerable, unanswerable)[1] == 0.0
