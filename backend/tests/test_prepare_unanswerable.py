"""The unanswerable benchmark builder keeps related passages and removes exactly the answer passages."""
import prepare_unanswerable as P

ITEMS = [{"id": f"q{i}", "question": f"question {i}", "ground_truth": "a", "gold_passages": [f"passage {i // 2}."]} for i in range(10)]


def test_questions_about_removed_passages_become_unanswerable_and_the_rest_stay_answerable():
    items, removed = P.split_unanswerable(ITEMS, fraction=0.4)
    assert len(removed) == 2                                               # 5 passages x 0.4
    for it in items:
        assert it["answerable"] == (it["gold_passages"][0] not in removed)
    assert 0 < sum(not it["answerable"] for it in items) < len(items)


def test_two_questions_on_one_passage_are_always_on_the_same_side():
    items, _ = P.split_unanswerable(ITEMS, fraction=0.5)
    by_passage = {}
    for it in items:
        by_passage.setdefault(it["gold_passages"][0], set()).add(it["answerable"])
    assert all(len(v) == 1 for v in by_passage.values())


def test_the_split_is_reproducible_and_depends_on_the_seed():
    a, _ = P.split_unanswerable(ITEMS, seed=1)
    b, _ = P.split_unanswerable(ITEMS, seed=1)
    c, _ = P.split_unanswerable(ITEMS, seed=2)
    assert a == b and [x["answerable"] for x in a] != [x["answerable"] for x in c]


def test_the_corpus_loses_only_the_removed_passages():
    corpus = "# Title\n\npassage 0.\n\npassage 1.\n\npassage 2.\n"
    out = P.corpus_without(corpus, {"passage 1."})
    assert "passage 1." not in out and "passage 0." in out and "passage 2." in out and out.startswith("# Title")
