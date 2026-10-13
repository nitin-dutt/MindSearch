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
