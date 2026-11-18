"""Search as a tool: the model decides WHAT to search for (alignment_plan CONV-06).

The default strategy (conversation.QueryRewriter) rewrites a follow-up into one standalone question and searches once. That cannot
help a message that asks about two things ("compare the refund policy with the warranty terms"): one search, one ranking, and the
passages about the second thing lose to the passages about the first.

With CHAT_STRATEGY=agent the language model is given one tool, `search_documents`, and must call it. Its arguments are
  * `question`: the user's latest message, made understandable without the chat (what the rewriter produced), and
  * `queries`:  1 to 3 search queries, one per thing the message asks about.
The SERVER then runs each query and merges the passages (rag_pipeline.RetrievalResult.merge), and the answer is written from them
as before. This is one planning step, not a loop: the model does not see the results and search again (each extra round costs a
model call on a GPU that serves one model; see the plan for why that is left out).

Security: the tool's arguments are text the model wrote, so they are untrusted. They are validated here (type, count, length), and
they can only ever be SEARCH WORDS. Who is searching is taken from the caller's token by the API, never from the arguments, so
nothing the model (or a document that tricks the model) writes can widen what is searched. Every call is audit-logged.

`SearchPlanner.plan` never raises; it returns the reason when there is no usable plan (timeout, model error, unusable arguments), and
chat.ChatService turns that into a visible failure (HTTP 503). There is no fallback to the rewrite strategy: one hid for weeks that
the model server had tool calling switched off. With CHAT_STRATEGY=agent the API also checks at startup that tool calling works.
"""
import asyncio
import json
from dataclasses import dataclass, field
from typing import List, Optional

from conversation import MAX_REWRITE_CHARS, _transcript

TOOL_NAME = "search_documents"
MAX_QUERY_CHARS = 300

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Search the company's documents. Returns the passages most relevant to each query.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string",
                             "description": "The user's latest message, rewritten so it can be understood without the conversation. "
                                            "Unchanged if it already can."},
                "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3,
                            "description": "1 to 3 short search queries. One query for a simple question; one query per topic when "
                                           "the message asks about several separate things."},
            },
            "required": ["question", "queries"],
            "additionalProperties": False,
        },
    },
}

PLANNER_SYSTEM = (
    "You plan document searches for an enterprise assistant. You never answer the user; you only call the search_documents tool.\n"
    "Fill its arguments like this:\n"
    "- question: the user's latest message rewritten so it can be understood without the earlier conversation. Replace pronouns and "
    "references (it, they, that, 'what about...') with the things they refer to, keeping the user's own words. If the message already "
    "stands on its own, copy it unchanged.\n"
    "- queries: what to search for. A simple question needs ONE query (usually the rewritten question itself). Use two or three queries "
    "only when the message asks about several separate things or compares things: then write one query per thing, each complete on its "
    "own. Never write queries about topics the user did not ask about, and never put facts or guesses of your own into a query."
)


@dataclass
class SearchPlan:
    question: str                       # the standalone question the answer is written for
    queries: List[str] = field(default_factory=list)


@dataclass
class PlanOutcome:
    plan: Optional[SearchPlan] = None
    error: Optional[str] = None         # why there is no plan: 'timeout', 'model_error' or 'invalid_arguments'


def parse_plan(raw: str, original: str, max_queries: int) -> Optional[SearchPlan]:
    """Validate the tool arguments the model wrote. Returns None if they are not a usable plan.
    Accepted: a JSON object with a list of query strings. Queries are trimmed, empty and over-long ones dropped, repeats removed, and
    the list is cut to `max_queries`. A missing or unusable `question` makes the whole plan unusable (no quiet substitute)."""
    try:
        args = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(args, dict) or not isinstance(args.get("queries"), list):
        return None
    queries, seen = [], set()
    for q in args["queries"]:
        if not isinstance(q, str):
            continue
        q = " ".join(q.split())
        if not q or len(q) > MAX_QUERY_CHARS or q.casefold() in seen:
            continue
        seen.add(q.casefold())
        queries.append(q)
    if not queries:
        return None
    question = args.get("question")
    question = " ".join(question.split()) if isinstance(question, str) else ""
    if not question or len(question) > MAX_REWRITE_CHARS:
        return None
    return SearchPlan(question=question, queries=queries[:max_queries])


class SearchPlanner:
    """Asks the model for a search plan. `call_tool(model, messages, tool, max_tokens)` returns the raw arguments string of the tool
    call (llm.complete_tool_call in production, a fake in tests)."""

    def __init__(self, model, call_tool, enabled=True, timeout=8.0, max_queries=3):
        self.model, self.call_tool, self.enabled, self.timeout, self.max_queries = model, call_tool, enabled, timeout, max_queries

    async def plan(self, history: List[dict], question: str) -> PlanOutcome:
        shown = f"Conversation so far:\n{_transcript(history)}\n\n" if history else ""
        messages = [{"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": f"{shown}Latest message: {question}"}]
        try:
            raw = await asyncio.wait_for(self.call_tool(self.model, messages, SEARCH_TOOL, max_tokens=250), self.timeout)
        except asyncio.TimeoutError:
            return PlanOutcome(error="timeout")
        except Exception:  # noqa: BLE001 - model down, bad response: reported as the reason; ChatService fails the turn
            return PlanOutcome(error="model_error")
        plan = parse_plan(raw, question, self.max_queries)
        return PlanOutcome(plan=plan) if plan else PlanOutcome(error="invalid_arguments")


PROBE_HISTORY = [{"role": "user", "content": "Who wrote the refund policy?"},
                 {"role": "assistant", "content": "The finance team wrote it."}]
PROBE_QUESTION = "When was it last updated?"


def check_tool_calling(planner: SearchPlanner) -> None:
    """Startup check for CHAT_STRATEGY=agent: ask for one real search plan and raise RuntimeError if none comes back, so a model
    server without tool calling (vLLM needs --enable-auto-tool-choice --tool-call-parser) stops the API at startup with the reason,
    instead of failing every chat turn later. Runs its own event loop in a thread, so it works whether or not a loop is running."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        outcome = pool.submit(asyncio.run, planner.plan(PROBE_HISTORY, PROBE_QUESTION)).result()
    if outcome.plan is None:
        raise RuntimeError(f"CHAT_STRATEGY=agent but the language model did not return a usable search plan ({outcome.error}). "
                           "Check that vLLM runs with --enable-auto-tool-choice --tool-call-parser, or set CHAT_STRATEGY=rewrite.")
