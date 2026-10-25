"""The prompt always fits the model window (GEN-05): lowest-ranked context goes first, the best passage always survives."""
import budget as B

SYSTEM = "You are a helpful assistant. " * 10


def doc(n_chars, tag="doc"):
    return f"<{tag}>" + "x" * n_chars + f"</{tag}>"


def test_estimates_grow_with_length_and_are_conservative():
    assert B.estimate_tokens("") == 0
    assert B.estimate_tokens("a" * 300) == 100 and B.estimate_tokens("a" * 301) == 101
    # English averages about 4 characters per token, so counting 3 per token over-estimates (safe side)
    assert B.estimate_tokens("The quick brown fox jumps over the lazy dog") >= 43 / 4


def test_everything_that_fits_is_kept_in_order():
    items = [doc(300), doc(300), "<kg_fact>a b</kg_fact>"]
    kept, dropped = B.fit_context(items, budget_tokens=1000)
    assert kept == items and dropped == 0


def test_the_lowest_ranked_items_are_dropped_first_and_no_gaps_are_left():
    items = [doc(300, "doc"), doc(300, "doc"), doc(30, "kg_fact")]
    one_cost = B.estimate_tokens(items[0]) + 2
    kept, dropped = B.fit_context(items, budget_tokens=2 * one_cost)       # room for two big ones
    assert kept == items[:2] and dropped == 1                               # the small, lowest-ranked fact is not squeezed in after a gap
    kept, dropped = B.fit_context(items, budget_tokens=one_cost + 5)
    assert kept == items[:1] and dropped == 2


def test_even_the_best_passage_is_truncated_rather_than_lost_and_stays_well_formed():
    big = doc(9000)
    kept, dropped = B.fit_context([big, doc(50)], budget_tokens=200)
    assert len(kept) == 1 and dropped == 1
    assert kept[0].startswith("<doc>") and kept[0].endswith("</doc>")
    assert B.estimate_tokens(kept[0]) <= 200


def test_the_budget_shrinks_with_history_question_and_answer_room():
    plain = B.context_budget(8192, 1024, SYSTEM, [], "short question")
    with_history = B.context_budget(8192, 1024, SYSTEM, [{"role": "user", "content": "x" * 3000}], "short question")
    bigger_answer = B.context_budget(8192, 2048, SYSTEM, [], "short question")
    assert with_history < plain and bigger_answer < plain
    assert plain < 8192 - 1024                                               # the answer's room is always reserved


def test_the_budget_never_drops_below_the_floor():
    assert B.context_budget(2048, 1024, SYSTEM, [{"role": "user", "content": "x" * 60000}], "q" * 8000) == B.MIN_CONTEXT_TOKENS


def test_the_assembled_prompt_never_exceeds_the_window():
    """End to end on the numbers: system + history + question + fitted context + answer room <= model window."""
    window, answer = 8192, 1024
    history = [{"role": "user", "content": "earlier question " * 20}, {"role": "assistant", "content": "earlier answer " * 40}]
    question = "What is the plan for the northern warehouse?"
    items = [doc(2500) for _ in range(20)] + ["<kg_fact>" + "f" * 400 + "</kg_fact>"] * 5
    kept, dropped = B.fit_context(items, B.context_budget(window, answer, SYSTEM, history, question))
    total = (B.estimate_tokens(SYSTEM) + sum(B.estimate_tokens(m["content"]) + B.PER_MESSAGE_OVERHEAD for m in history)
             + B.estimate_tokens(question) + sum(B.estimate_tokens(k) + 2 for k in kept) + answer)
    assert dropped > 0 and total <= window


def test_a_numbered_passage_is_truncated_without_losing_its_tags():
    numbered = '<doc id="3">' + "x" * 9000 + "</doc>"
    kept, _ = B.fit_context([numbered], budget_tokens=200)
    assert kept[0].startswith('<doc id="3">') and kept[0].endswith("</doc>") and B.estimate_tokens(kept[0]) <= 200
