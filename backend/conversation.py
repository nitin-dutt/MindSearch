"""Multi-turn chat support (alignment_plan INF-06).

Problem: "What about its price?" means nothing to a search engine that never sees the chat. Two things fix it:
  1. QueryRewriter turns a follow-up into a standalone question using the recent history, and retrieval searches with that.
  2. The answer-writing model gets a short, size-limited slice of the conversation for continuity.

Limits are by SIZE as well as count (a 7B model on an 8 GB GPU has little spare memory per request, and long old chat
distracts it). The SERVER enforces them, never the client. History is untrusted input: only user/assistant roles are kept,
and it never influences access control (retrieval still runs with the caller's own permissions).
"""
import asyncio
import re
from typing import List, Tuple

from settings import get_settings

_cfg = get_settings()
MAX_HISTORY_MESSAGES = _cfg.history_max_messages                  # default 6 = 3 question/answer turns
GENERATION_HISTORY_MESSAGES = _cfg.history_generation_messages    # default 4 = 2 turns for the answer writer
HISTORY_TOKEN_BUDGET = _cfg.history_token_budget
PER_MESSAGE_CHARS = _cfg.history_per_message_chars
REWRITE_TIMEOUT_S = _cfg.rewrite_timeout_s
MAX_QUESTION_CHARS = 8000
MAX_REWRITE_CHARS = 400


class ConversationError(ValueError):
    pass


def estimate_tokens(text: str) -> int:
    """Rough size: about 4 characters per token for English. Good enough for a budget; avoids loading a tokenizer."""
    return max(1, len(text) // 4)


def split_conversation(messages) -> Tuple[str, List[dict]]:
    """(current question, earlier turns). Only user/assistant messages count; anything else (e.g. a client-supplied
    'system' message) is dropped. The last message must be a user message."""
    usable = [m for m in messages if m.role in ("user", "assistant") and m.content.strip()]
    if not usable or usable[-1].role != "user":
        raise ConversationError("The last message must be a non-empty message from the user.")
    question = usable[-1].content.strip()
    if len(question) > MAX_QUESTION_CHARS:
        raise ConversationError(f"The question is longer than {MAX_QUESTION_CHARS} characters.")
    return question, [{"role": m.role, "content": m.content.strip()} for m in usable[:-1]]


def trim_history(history: List[dict], max_messages: int = None, token_budget: int = None,
                 per_message_chars: int = None) -> List[dict]:
    """The most recent messages that fit the limits, oldest dropped first. Long messages are shortened (old answers are
    the usual culprit). The result always starts with a user message, as chat models expect."""
    max_messages = MAX_HISTORY_MESSAGES if max_messages is None else max_messages
    token_budget = HISTORY_TOKEN_BUDGET if token_budget is None else token_budget
    per_message_chars = PER_MESSAGE_CHARS if per_message_chars is None else per_message_chars
    kept = [{"role": m["role"], "content": m["content"] if len(m["content"]) <= per_message_chars
             else m["content"][:per_message_chars].rstrip() + "…"} for m in history[-max_messages:]] if max_messages > 0 else []
    while kept and (sum(estimate_tokens(m["content"]) for m in kept) > token_budget or kept[0]["role"] != "user"):
        kept.pop(0)
    return kept


def for_generation(history: List[dict]) -> List[dict]:
    """The even smaller slice shown to the answer-writing model (the rewrite already carried the meaning forward)."""
    return trim_history(history, max_messages=GENERATION_HISTORY_MESSAGES)


REWRITE_SYSTEM = (
    "You rewrite a user's follow-up question so it can be understood without the earlier conversation. "
    "Replace pronouns and references (it, they, that, the first one, 'what about...') with the specific things they refer to. "
    "Keep the user's meaning and wording as far as possible. Do not answer the question and do not add facts. "
    "If the question already stands on its own, return it unchanged. Output only the rewritten question, nothing else."
)


def _transcript(history: List[dict]) -> str:
    return "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in history)


def clean_rewrite(text: str, original: str) -> str:
    """Take the model's output, keep only a sane single-line question, or fall back to the original."""
    line = (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""
    line = re.sub(r"^(standalone question|rewritten question|question)\s*:\s*", "", line, flags=re.I).strip().strip('"').strip("'").strip()
    if not line or len(line) > MAX_REWRITE_CHARS:
        return original
    return line


class QueryRewriter:
    """Turns a follow-up into a standalone question. Never raises: on any problem the original question is used."""

    def __init__(self, model, complete, enabled=True, timeout=REWRITE_TIMEOUT_S):
        self.model, self.complete, self.enabled, self.timeout = model, complete, enabled, timeout

    async def rewrite(self, history: List[dict], question: str) -> Tuple[str, bool]:
        """Returns (question to search with, whether it was changed)."""
        if not self.enabled or not history:
            return question, False
        messages = [{"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": f"Conversation so far:\n{_transcript(history)}\n\n"
                                                f"Follow-up question: {question}\n\nStandalone question:"}]
        try:
            raw = await asyncio.wait_for(self.complete(self.model, messages, max_tokens=100, temperature=0.0), self.timeout)
        except Exception:  # noqa: BLE001 - timeouts, model errors, anything: fall back silently (the answer still works)
            return question, False
        standalone = clean_rewrite(raw, question)
        return standalone, standalone.strip().lower() != question.strip().lower()
