"""Keep the prompt inside the model's context window (alignment_plan GEN-05).

A prompt = system instructions + recent chat + retrieved passages + the question, and the model also needs room to write its answer.
If the whole thing is longer than the model's window the request fails or the start is silently cut off. So the retrieved passages are
trimmed to whatever space is left, dropping the LOWEST-ranked ones first (the context list is best-first, with graph facts last).

Tokens are ESTIMATED from characters (3 characters per token). English averages about 4, so this over-counts on purpose: the budget errs
on the side of dropping a little too much rather than overflowing. Counting exactly would mean loading the language model's tokenizer in
the API process; `evaluation` checks the estimate against the real tokenizer instead.
"""
import math
import re

CHARS_PER_TOKEN = 3.0
PER_MESSAGE_OVERHEAD = 6          # role markers and separators around each chat message
TEMPLATE_OVERHEAD = 48            # "Context: ... Question: ... Answer:" wrapper and chat-template tokens
SAFETY_MARGIN = 64
MIN_CONTEXT_TOKENS = 256          # never starve the context completely, whatever the history looks like

_TAGGED = re.compile(r"^(<(\w+)>)(.*)(</\2>)$", re.S)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def context_budget(model_max_len, answer_max_tokens, system_prompt, history, question) -> int:
    """Tokens left for the retrieved passages."""
    used = (answer_max_tokens + estimate_tokens(system_prompt) + estimate_tokens(question) + TEMPLATE_OVERHEAD + SAFETY_MARGIN
            + sum(estimate_tokens(m["content"]) + PER_MESSAGE_OVERHEAD for m in (history or [])))
    return max(MIN_CONTEXT_TOKENS, model_max_len - used)


def _truncate(item: str, budget_tokens: int) -> str:
    """Shorten one passage to the budget, keeping its <doc>...</doc> tags balanced."""
    max_chars = max(0, int(budget_tokens * CHARS_PER_TOKEN))
    m = _TAGGED.match(item)
    if not m:
        return item[:max_chars]
    open_tag, _, inner, close_tag = m.groups()
    room = max(0, max_chars - len(open_tag) - len(close_tag))
    return open_tag + inner[:room] + close_tag


def fit_context(items, budget_tokens):
    """(kept items, number dropped). Items arrive best-first; whole items are kept until the next one would not fit, then the rest are
    dropped, so the kept list is always a prefix (no gaps in the ranking). If even the best item alone is too long it is truncated."""
    kept, used = [], 0
    for index, item in enumerate(items):
        cost = estimate_tokens(item) + 2
        if used + cost <= budget_tokens:
            kept.append(item)
            used += cost
        elif not kept:
            kept.append(_truncate(item, budget_tokens - 2))
            return kept, len(items) - 1
        else:
            return kept, len(items) - index
    return kept, 0
