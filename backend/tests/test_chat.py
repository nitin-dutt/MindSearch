"""One chat turn driven directly through chat.ChatService, with no HTTP: the same interface the API uses, so anything else (the
evaluation harness, a script) can run the whole path too. The HTTP view of the same behaviour is in test_api.py / test_agent.py."""
import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("anyio")

import chat  # noqa: E402
from llm import LLMError  # noqa: E402

USER = SimpleNamespace(id="staff", is_super=False)


class Audit:
    def __init__(self):
        self.rows = []

    def log(self, event, **fields):
        self.rows.append({"event": event, **fields})


class Rag:
    def __init__(self, found=True, weak=False, tokens=("Paris ", "[1]."), fail=False):
        self.found, self.weak, self.tokens, self.fail, self.queries = found, weak, tokens, fail, []
        self.registry = SimpleNamespace(filenames=lambda ids: {i: "guide.txt" for i in ids})

    def retrieve(self, query, user, doc_ids=None):
        self.queries.append((query, user.id, doc_ids))
        ctx = ['<doc id="1">Paris is the capital.</doc>'] if self.found else []
        return SimpleNamespace(to_context=lambda: ctx,
                               sources=lambda: [{"n": 1, "doc_id": "d1", "chunk_index": 0, "page": None}] if ctx else [])

    def should_abstain(self, result):
        return self.weak

    async def stream_answer(self, query, context, history=None, today=None):
        if self.fail:
            raise LLMError("down")
        for t in self.tokens:
            yield t


def run(rag, question="What is the capital?", history=(), **kw):
    async def go():
        service = chat.ChatService(rag, Audit(), workers=2)
        turn = await service.begin(list(history), question, USER, **kw)
        return turn, [e async for e in turn.events()]
    return asyncio.run(go())


def test_an_answer_is_tokens_then_the_outcome_then_the_sources_with_what_was_cited():
    turn, events = run(Rag())
    assert [e["type"] for e in events] == ["token", "token", "outcome", "sources"]
    assert events[2] == {"type": "outcome", "outcome": "answered"}
    assert events[3]["sources"] == [{"n": 1, "doc_id": "d1", "chunk_index": 0, "page": None, "filename": "guide.txt", "cited": True}]
    assert turn.outcome is None and turn.query == "What is the capital?" and turn.rewritten is False


def test_nothing_found_and_a_weak_match_are_decided_before_the_model_is_asked():
    turn, events = run(Rag(found=False, fail=True))             # the model would fail, but it is never called
    assert turn.outcome == "none_found"
    assert events == [{"type": "outcome", "outcome": "none_found"}, {"type": "token", "text": chat.NO_CONTEXT_MESSAGE}]
    turn, events = run(Rag(weak=True, fail=True))
    assert turn.outcome == "refused" and events[1]["text"] == chat.ABSTAIN_MESSAGE


def test_the_models_own_i_dont_know_is_reported_as_a_refusal_and_lists_no_sources():
    _, events = run(Rag(tokens=("I don't know ", "based on the documents available to you.")))
    assert [e["type"] for e in events] == ["outcome", "token", "token"]
    assert events[0]["outcome"] == "refused"


def test_a_model_that_cannot_start_raises_instead_of_answering():
    with pytest.raises(LLMError):
        run(Rag(fail=True))


def test_every_search_runs_as_the_caller_within_the_callers_scope():
    rag = Rag()
    run(rag, doc_ids=["a" * 32])
    assert rag.queries == [("What is the capital?", "staff", ["a" * 32])]
