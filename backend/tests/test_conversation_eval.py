"""The multi-turn evaluation harness (CONV-03): its strategies change only how the follow-up is rewritten, and a chat is scored
from what chat.ChatService really did. Uses fakes; the real run needs the model and the evaluation index."""
import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("anyio")

from evaluation import conversation_eval as ce  # noqa: E402
from evaluation import prepare_conversations as pc  # noqa: E402

GOLD = "Paris is the capital of France."
USER = SimpleNamespace(id="eval", is_super=True)


class FakeRewriter:
    def __init__(self):
        self.calls = []

    async def rewrite(self, history, question, force=False):
        self.calls.append(force)
        return ("What is the capital of France?", True) if force or question.startswith("And") else (question, False)


class Rag:
    def __init__(self, chunks, weak=False, tokens=("Paris [1].",)):
        self.chunks, self.weak, self.tokens = chunks, weak, tokens
        self.registry = SimpleNamespace(filenames=lambda ids: {})

    def retrieve(self, query, user, doc_ids=None):
        found = [{"chunk": c} for c in self.chunks] if "France" in query else []
        ctx = [c["chunk"] for c in found]
        return SimpleNamespace(chunks=found, to_context=lambda: ctx,
                               sources=lambda: [{"n": 1, "doc_id": "d", "chunk_index": 0, "page": None}] if ctx else [])

    def should_abstain(self, result):
        return self.weak

    async def stream_answer(self, query, context, history=None, today=None):
        for t in self.tokens:
            yield t


def test_always_forces_every_rewrite_and_noretry_refuses_the_forced_one():
    inner = FakeRewriter()
    assert asyncio.run(ce.AlwaysRewrite(inner).rewrite([], "Tell me more.")) == ("What is the capital of France?", True)
    assert inner.calls == [True]
    assert asyncio.run(ce.NoRetry(inner).rewrite([], "Tell me more.", force=True)) == ("Tell me more.", False)
    assert inner.calls == [True]                                   # the retry never reached the real rewriter


CHAT = {"id": "c1", "category": "pronoun", "expect": "answer", "gold_passages": [GOLD],
        "history": [{"role": "user", "content": "Tell me about France."}, {"role": "assistant", "content": "A country."}],
        "question": "And its capital?", "standalone": "What is the capital of France?"}


def run_chat(strategy, rag, generate, chat=CHAT):
    from chat import ChatService
    service = ChatService(rag, ce.RecordingAudit(), rewriter=FakeRewriter() if strategy != "raw" else None, workers=1)
    if strategy == "standalone":
        service.rewriter = None
    return asyncio.run(ce.one_chat(service, rag, chat, strategy, generate, USER))


@pytest.mark.parametrize("generate", [False, True])
def test_a_rewritten_follow_up_finds_the_gold_passage_and_counts_as_answered(generate):
    row = run_chat("needed", Rag([GOLD, "Other text."]), generate)
    assert row["rewritten"] is True and row["gold@1"] == 1.0 and row["refused"] is False and row["right_outcome"] == 1.0
    assert ("answer" in row) is generate


def test_raw_searches_as_typed_and_standalone_ignores_the_history():
    raw = run_chat("raw", Rag([GOLD]), False)
    assert raw["query"] == "And its capital?" and raw["refused"] is True and raw["right_outcome"] == 0.0
    standalone = run_chat("standalone", Rag([GOLD]), False)
    assert standalone["query"] == "What is the capital of France?" and standalone["gold@1"] == 1.0


def test_an_unanswerable_chat_is_right_only_when_refused():
    chat = {**CHAT, "expect": "refuse", "gold_passages": []}
    assert run_chat("needed", Rag([GOLD], weak=True), False, chat)["right_outcome"] == 1.0       # the gate refused
    refused = run_chat("needed", Rag([GOLD], tokens=("I don't know based on the documents available to you.",)), True, chat)
    assert refused["refused"] and refused["right_outcome"] == 1.0                                     # the model refused
    row = run_chat("needed", Rag([GOLD]), True, chat)
    assert row["right_outcome"] == 0.0 and "gold@1" not in row


def test_the_hand_written_set_has_the_planned_size_and_no_repeated_last_question():
    sizes = {"pronoun": len(pc.PRONOUN), "topic_switch": len(pc.TOPIC_SWITCH), "vague": len(pc.VAGUE),
             "refusal_mid": len(pc.REFUSAL_MID), "unanswerable": len(pc.UNANSWERABLE)}
    assert sizes == {"pronoun": 35, "topic_switch": 20, "vague": 15, "refusal_mid": 15, "unanswerable": 15}
    last = [row[-1] for row in pc.PRONOUN + pc.REFUSAL_MID + pc.UNANSWERABLE] + [row[2] for row in pc.VAGUE]
    assert len(last) == len(set(last))
