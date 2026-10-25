"""The generation/summary half of the harness, exercised with a fake LLM (no vLLM needed)."""
import asyncio

import pytest

pytest.importorskip("openai")

from evaluation import generate as G  # noqa: E402
from llm import LLMError  # noqa: E402


class FakeRag:
    """stream_answer yields canned tokens keyed by question."""
    def __init__(self, answers):
        self.answers = answers

    async def stream_answer(self, query, context):
        for tok in self.answers[query]:
            if isinstance(tok, Exception):
                raise tok
            yield tok


def _rec(qid, question, contexts, gold="white"):
    return {"id": qid, "question": question, "ground_truth": gold, "contexts": contexts, "n_gold": 1,
            "type": None, "graph_facts": [], "graph_pool_size": 0, "timings_ms": {"dense_and_rerank": 10.0},
            "n_context_items": len(contexts), "context_words": sum(len(c.split()) for c in contexts),
            "final_recall@3": 1.0, "final_mrr": 1.0, "pool_recall@20": 1.0}


def test_generate_all_scores_failures_and_empty_context_separately():
    records = [
        _rec("a", "q-ok", ["<doc>x</doc>"]),
        _rec("b", "q-err", ["<doc>x</doc>"]),
        _rec("c", "q-empty", []),
    ]
    rag = FakeRag({"q-ok": ["The jersey was ", "white."], "q-err": ["partial ", LLMError("boom")], "q-empty": []})
    asyncio.run(G.generate_all(rag, records, concurrency=2))
    ok, err, empty = records

    assert ok["answer"] == "The jersey was white." and not ok["failed"] and ok["ans_contains_gold"] == 1.0
    assert err["failed"] and "ans_contains_gold" not in err   # an error string is never scored as an answer
    assert empty["no_context"] and not empty["failed"] and empty["answer"] == ""

    summary = G.summarize_run(records, generated=True)
    gen = summary["generation"]
    assert gen["n_failed"] == 1 and gen["n_no_context"] == 1
    assert gen["answers"]["ans_contains_gold"]["n"] == 1      # only the genuinely answered question counts


def test_summary_without_generation_reports_graph_coverage():
    recs = [_rec("a", "q", ["<doc>x</doc>"]), _rec("b", "q2", ["<doc>y</doc>"])]
    recs[0]["graph_facts"], recs[0]["graph_pool_size"] = ["A uses B"], 4
    summary = G.summarize_run(recs, generated=False)
    assert summary["graph"]["questions_with_graph_facts"] == 0.5
    assert summary["graph"]["questions_with_raw_graph_pool"] == 0.5
    assert "generation" not in summary
    assert summary["retrieval"]["final_recall@3"]["mean"] == 1.0


# ------------------------------------------------------------------ abstention and citations in the harness (GEN-02, GEN-03)

def _arec(qid, question, answerable=True, contexts=("<doc id=\"1\">a</doc>", "<doc id=\"2\">b</doc>"), gated=False, covers=((0,), ())):
    r = _rec(qid, question, list(contexts))
    r.update(answerable=answerable, gated=gated, n_passages=len(contexts), final_covers=[list(c) for c in covers])
    if not answerable:
        for k in [k for k in r if k.startswith(("final_", "pool_"))]:
            del r[k]
    return r


def test_abstention_and_citation_quality_are_measured_separately_from_answers():
    from answers import ABSTAIN_MESSAGE
    records = [
        _arec("a1", "q-cited"),                                   # answerable, answers and cites the right passage [1]
        _arec("a2", "q-refused"),                                 # answerable, but the model refuses: a false refusal
        _arec("u1", "q-idk", answerable=False),                   # unanswerable, the model says I don't know: correct
        _arec("u2", "q-guess", answerable=False),                 # unanswerable, the model answers anyway: a hallucination
        _arec("u3", "q-gated", answerable=False, gated=True),     # unanswerable, refused by the gate without calling the model
    ]
    rag = FakeRag({"q-cited": ["It was white [1] and [2]."], "q-refused": [ABSTAIN_MESSAGE], "q-idk": [ABSTAIN_MESSAGE],
                   "q-guess": ["It was white [9]."], "q-gated": ["never called"]})
    asyncio.run(G.generate_all(rag, records, concurrency=2))
    assert records[4]["answer"] == ABSTAIN_MESSAGE and records[4]["abstained"]               # the gate answered, not the model
    summary = G.summarize_run(records, generated=True)

    ab = summary["abstention"]
    assert ab["n_unanswerable"] == 3 and ab["n_answerable"] == 2
    assert ab["abstain_rate_on_unanswerable"]["mean"] == pytest.approx(2 / 3)              # u1 and u3 refused, u2 hallucinated
    assert ab["false_refusal_rate_on_answerable"]["mean"] == 0.5                           # a2 refused
    assert ab["answered_by_gate_share"] == pytest.approx(1 / 3)

    c = summary["citations"]
    assert c["n_answers"] == 1 and c["answers_with_a_valid_citation"]["mean"] == 1.0       # only a1 is an answerable, non-refused answer
    assert c["citation_precision"] == 0.5                                                  # [1] holds the gold passage, [2] does not
    assert summary["n_answerable"] == 2 and summary["generation"]["answers"]["ans_contains_gold"]["n"] == 2   # unanswerable ones are not scored as answers


def test_a_run_without_unanswerable_questions_has_no_abstention_block():
    recs = [_arec("a", "q-ok")]
    asyncio.run(G.generate_all(FakeRag({"q-ok": ["white [1]"]}), recs, concurrency=1))
    assert "abstention" not in G.summarize_run(recs, generated=True)
