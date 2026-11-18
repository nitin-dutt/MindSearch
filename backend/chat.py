"""One chat turn, from the user's message to the last event of the answer.

    turn = await chat.begin(history, question, user, doc_ids)     # searches, decides, asks the model for its first word
    async for event in turn.events(): ...                         # typed events, in the order the client should see them

Everything between those two lines lives here: the search plan (agent.py) or the follow-up rewrite (conversation.py), the retry with a
rewritten question, both ways of refusing without the model (nothing found, best passage too weak), naming the sources, the citations,
the metrics and the audit rows. The HTTP layer (api.py) only turns the events into Server-Sent Events; the evaluation harness can drive
the very same path without HTTP.

Events (plain dicts, always with a "type"):
    {"type": "standalone", "question": str}     the follow-up as it was actually searched for (only when it was changed)
    {"type": "searches",   "queries": [str]}    the separate searches the agent strategy ran (only when more than one)
    {"type": "outcome",    "outcome": str}      what kind of reply this is: "answered", "refused" (an "I don't know", from the gate
                                                or from the model) or "none_found" (nothing this user may read matched). The client
                                                styles the reply from THIS, never by reading the answer text.
    {"type": "token",      "text": str}         a piece of the answer
    {"type": "sources",    "sources": [...]}    the numbered passages the model was shown, each with a `cited` flag
    {"type": "error",      "message": str}      the model died mid-answer; whatever text came before is incomplete

Error mode: `begin` raises llm.LLMError when the model cannot produce a first word, so the caller can answer with a real failure
(HTTP 503) instead of a stream whose "answer" is an error message. After `begin` returned, `events()` never raises.

Security: every search runs as `user` (taken from the caller's token by the API) with the caller's document scope. Neither the chat
history nor anything the model writes can change who is searching.
"""
import asyncio
import datetime
import functools
import time
from typing import List, Optional

import anyio

import agent
import conversation as conv
import metrics
from answers import ABSTAIN_MESSAGE, is_abstention, mark_cited
from audit import text_fingerprint
from llm import LLMError
from settings import get_settings

NO_CONTEXT_MESSAGE = (
    "I couldn't find any relevant information in the documents available to you. "
    "If you expected an answer, check that the document has finished processing and that "
    "your access level covers it."
)

ANSWERED, REFUSED, NONE_FOUND = "answered", "refused", "none_found"


class ChatTurn:
    """What `ChatService.begin` hands back: the facts about the search, and the events of the reply."""

    def __init__(self, question, query, rewritten, searches, result, retrieval_ms, events, outcome=None):
        self.question = question            # the user's own words
        self.query = query                  # the standalone question the answer is written for
        self.rewritten = rewritten          # True when `query` differs from `question`
        self.searches = searches            # every search query that ran, in order
        self.result = result                # the pipeline's RetrievalResult (after merging / the retry)
        self.retrieval_ms = retrieval_ms
        self.outcome = outcome              # known before the model is asked only for "refused" (the gate) and "none_found"
        self._events = events

    def events(self):
        """The reply as an async iterator of events (see the module docstring). Consume it once."""
        return self._events


class ChatService:
    """`rag` is the pipeline (retrieve, merge_results, should_abstain, stream_answer, registry). `planner` (agent.SearchPlanner) and
    `rewriter` (conversation.QueryRewriter) are optional: without a planner, or with a disabled one, the rewrite strategy runs; without
    a rewriter the question is searched as typed."""

    def __init__(self, rag, audit, planner=None, rewriter=None, workers=None, max_passages=None):
        self.rag, self.audit, self.planner, self.rewriter = rag, audit, planner, rewriter
        cfg = get_settings()
        # Retrieval is CPU-heavy (embedding, cross-encoder, graph lookup). It runs in worker threads so the event loop stays free for
        # streaming and health checks, and the pool is capped so a burst of users cannot oversubscribe the CPU (extra requests wait).
        self._workers = int(workers or cfg.retrieval_workers)
        self._max_passages = int(max_passages or cfg.agent_max_passages)
        self._limiter = None

    # ------------------------------------------------------------------ searching

    async def _search(self, query, user, doc_ids):
        if self._limiter is None:                       # created lazily, inside the running event loop
            self._limiter = anyio.CapacityLimiter(self._workers)
        scope = {"doc_ids": doc_ids} if doc_ids else {}
        return await anyio.to_thread.run_sync(functools.partial(self.rag.retrieve, query, user, **scope), limiter=self._limiter)

    async def _rewrite(self, history, question, force=False):
        if self.rewriter is None:
            return question, False
        return await self.rewriter.rewrite(history, question, force=force)

    def _usable(self, result) -> bool:
        """Something was found and the refusal gate would let it through."""
        return bool(result.to_context()) and not self.rag.should_abstain(result)

    async def _plan(self, history, question, user) -> Optional[agent.SearchPlan]:
        """Agent strategy (CONV-06): the model plans the searches by calling the `search_documents` tool. Its arguments are only search
        words. None means the agent strategy is off (no planner, or CHAT_STRATEGY=rewrite). A plan that fails (timeout, model error,
        unusable arguments) raises LLMError: there is deliberately NO fallback to the rewrite strategy, because a quiet fallback hid
        for weeks that the model server had tool calling switched off and the agent never once ran (found 2026-11-17)."""
        if self.planner is None or not self.planner.enabled:
            return None
        with metrics.timed("plan"):
            outcome = await self.planner.plan(history, question)
        plan = outcome.plan
        metrics.AGENT_PLANS.labels("used" if plan else outcome.error).inc()
        if plan is None:
            self.audit.log("tool_call_failed", user=user.id, tool=agent.TOOL_NAME, reason=outcome.error)
            raise LLMError(f"planning the searches failed ({outcome.error})")
        metrics.AGENT_QUERIES.inc(len(plan.queries))
        self.audit.log("tool_call", user=user.id, tool=agent.TOOL_NAME, queries=[text_fingerprint(q) for q in plan.queries])
        return plan

    async def _retrieve(self, history, question, user, doc_ids):
        """(query the answer is written for, whether it differs from the question, the searches that ran, result, seconds, strategy)."""
        plan = await self._plan(history, question, user)
        if plan is not None:
            t0 = time.perf_counter()
            results = await asyncio.gather(*(self._search(q, user, doc_ids) for q in plan.queries))
            result = self.rag.merge_results(list(results), self._max_passages)
            rewritten = plan.question.strip().lower() != question.strip().lower()
            return plan.question, rewritten, list(plan.queries), result, time.perf_counter() - t0, "agent"

        # A follow-up ("what about its price?") is rewritten into a standalone question; retrieval searches with that.
        with metrics.timed("rewrite"):
            query, rewritten = await self._rewrite(history, question)
        t0 = time.perf_counter()
        searches = [query]
        result = await self._search(query, user, doc_ids)
        # A question that looked complete is searched as typed. If that finds nothing usable ("give a detailed list" means nothing
        # without the chat) and there is history, rewrite it from the history and search once more, then use whichever did better.
        if history and not rewritten and not self._usable(result):
            retry_query, changed = await self._rewrite(history, question, force=True)
            if changed:
                retry = await self._search(retry_query, user, doc_ids)
                searches.append(retry_query)
                helped = self._usable(retry)
                metrics.REWRITE_RETRIES.labels("helped" if helped else "no_help").inc()
                if helped:
                    query, rewritten, result = retry_query, True, retry
        return query, rewritten, searches, result, time.perf_counter() - t0, "rewrite"

    # ------------------------------------------------------------------ the turn

    async def begin(self, history: List[dict], question: str, user, doc_ids=None, today=None) -> ChatTurn:
        """Search and decide how this turn is answered. `history` is the earlier turns ([{"role", "content"}], see
        conversation.split_conversation); it is trimmed here. Raises LLMError if the model cannot start an answer."""
        t_request = time.perf_counter()
        history = conv.trim_history(history)
        try:
            query, rewritten, searches, result, seconds, strategy = await self._retrieve(history, question, user, doc_ids)
        except LLMError as e:                           # the rewrite or the search plan needed the model and it failed
            self.audit.log("chat_failed", user=user.id, reason="language_model_unavailable", stage="search")
            metrics.chat_outcome("failed")
            raise LLMError(str(e)) from e
        context = result.to_context()
        metrics.STAGE_SECONDS.labels("retrieve_total").observe(seconds)
        self.audit.log("chat", user=user.id, super_role=user.is_super, query=text_fingerprint(question), rewritten=rewritten,
                       history_messages=len(history), context_items=len(context), scoped_docs=len(doc_ids or []),
                       strategy=strategy, searches=len(searches) if strategy == "agent" else 1)

        def turn(events, outcome=None):
            return ChatTurn(question, query, rewritten, searches, result, seconds * 1000, events, outcome)

        # Nothing retrieved (empty index, or nothing this user may see): answer deterministically without calling the model, so it
        # cannot answer from thin air.
        if not context:
            metrics.chat_outcome("no_context")
            return turn(_fixed_reply(NONE_FOUND, NO_CONTEXT_MESSAGE), NONE_FOUND)

        # The refusal gate: something was retrieved, but the best passage is too weakly related to the question to answer from.
        # Say so deterministically instead of letting the model improvise from loosely related text.
        if self.rag.should_abstain(result):
            self.audit.log("chat_abstained", user=user.id, reason="low_relevance")
            metrics.chat_outcome("refused_gate")
            return turn(_fixed_reply(REFUSED, ABSTAIN_MESSAGE), REFUSED)

        # The passages the model is shown, numbered like the citations in its answer (filenames come from the registry; the chunk
        # filter already guaranteed this user may read them).
        sources = result.sources()
        if sources:
            names = await anyio.to_thread.run_sync(self.rag.registry.filenames, list({s["doc_id"] for s in sources}))
            sources = [{**s, "filename": names.get(s["doc_id"], "(unknown)")} for s in sources]

        # Ask for the first token BEFORE anything is sent: if the model is down the caller gets an exception (HTTP 503), not an
        # "answer" that is an error message.
        t_generation = time.perf_counter()
        answer = self.rag.stream_answer(query, context, conv.for_generation(history), today=today or datetime.date.today())
        try:
            first = await answer.__anext__()
        except StopAsyncIteration:
            first = None
        except LLMError:
            self.audit.log("chat_failed", user=user.id, reason="language_model_unavailable")
            metrics.chat_outcome("failed")
            raise
        metrics.TTFT_SECONDS.observe(time.perf_counter() - t_request)

        shown_searches = searches if strategy == "agent" and len(searches) > 1 else []
        return turn(self._answer_events(user, query if rewritten else None, shown_searches, first, answer, sources, t_generation))

    async def _answer_events(self, user, standalone, searches, first, answer, sources, t_generation):
        if standalone:
            yield {"type": "standalone", "question": standalone}
        if searches:                                    # the agent strategy ran several searches: show the user what was searched
            yield {"type": "searches", "queries": searches}
        failed, refused, parts = False, False, []

        async def tokens():
            if first is not None:
                yield first
            async for tok in answer:
                yield tok
        try:
            async for tok in tokens():
                parts.append(tok)
                # An "I don't know" is recognisable from its first words: tell the client as soon as it is, so it never has to guess
                # from the text. (Once true it stays true: more text cannot turn a refusal back into an answer.)
                if not refused and is_abstention("".join(parts)):
                    refused = True
                    yield {"type": "outcome", "outcome": REFUSED}
                yield {"type": "token", "text": tok}
        except Exception as e:  # noqa: BLE001 - the reply has started, so the failure travels as an event
            failed = True
            self.audit.log("chat_failed", user=user.id, reason=type(e).__name__)
            metrics.chat_outcome("interrupted")
            yield {"type": "error", "message": "The answer was interrupted. Please try again."}
        full = "".join(parts)
        metrics.STAGE_SECONDS.labels("generation").observe(time.perf_counter() - t_generation)
        if failed:
            return
        metrics.chat_outcome("abstained_model" if refused else "answered")
        if refused:                                     # an 'I don't know' cites nothing
            return
        yield {"type": "outcome", "outcome": ANSWERED}
        if sources:
            marked, info = mark_cited(sources, full)    # `cited` = the [n] numbers the answer actually used
            if info["invalid"]:
                self.audit.log("citation_invalid", user=user.id, numbers=info["invalid"])
                metrics.INVALID_CITATIONS.inc(len(info["invalid"]))
            yield {"type": "sources", "sources": marked}


async def _fixed_reply(outcome, text):
    yield {"type": "outcome", "outcome": outcome}
    yield {"type": "token", "text": text}


def model_helpers(model) -> dict:
    """The planner and the rewriter as the running application uses them (they talk to the configured language model `model`):
    `ChatService(rag, audit, **model_helpers(model))`."""
    from llm import complete_chat, complete_tool_call
    cfg = get_settings()
    return {"planner": agent.SearchPlanner(model, complete_tool_call, enabled=cfg.chat_strategy == "agent",
                                           timeout=cfg.agent_timeout_s, max_queries=cfg.agent_max_queries),
            "rewriter": conv.QueryRewriter(model, complete_chat, enabled=cfg.rewrite_enabled, timeout=cfg.rewrite_timeout_s)}
