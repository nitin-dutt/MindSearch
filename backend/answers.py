"""Helpers for the shape of an answer: the "I don't know" sentence, and the [n] citations (alignment_plan GEN-02 / GEN-03).

The language model is told to cite the numbered passages it used as [1], [2][3] ... and, when the passages do not contain the answer,
to reply with exactly ABSTAIN_MESSAGE. These functions read that back out of the answer text so the API can mark which sources were
actually cited and the evaluation can measure citation quality and abstention. They never change the answer.
"""
import re

ABSTAIN_MESSAGE = "I don't know based on the documents available to you."

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_QUOTES = str.maketrans({"’": "'", "‘": "'"})


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.translate(_QUOTES).lower()).strip()


def is_abstention(answer: str) -> bool:
    """True if the answer is the model's (or the gate's) 'I don't know'. Deliberately strict: the exact sentence anywhere, or an answer that
    STARTS with 'I don't know' / 'I do not know'. A real answer that merely mentions what the documents lack does not count."""
    norm = _normalize(answer)
    return _normalize(ABSTAIN_MESSAGE) in norm or norm.startswith(("i dont know", "i do not know"))


def extract_citations(answer: str, n_sources: int) -> dict:
    """Which numbered passages the answer cites. Returns {"cited": sorted valid numbers, "invalid": numbers that point at no passage,
    "mentions": how many [n] markers there are}. A number is valid if 1 <= n <= n_sources."""
    numbers = [int(x) for group in _CITATION.findall(answer) for x in re.split(r"\s*,\s*", group)]
    valid = sorted({n for n in numbers if 1 <= n <= n_sources})
    invalid = sorted({n for n in numbers if not 1 <= n <= n_sources})
    return {"cited": valid, "invalid": invalid, "mentions": len(numbers)}


def mark_cited(sources: list, answer: str) -> tuple:
    """(sources with a `cited` flag, extract_citations result). `sources` entries carry their citation number as `n`."""
    info = extract_citations(answer, len(sources))
    return [{**s, "cited": s.get("n") in info["cited"]} for s in sources], info
