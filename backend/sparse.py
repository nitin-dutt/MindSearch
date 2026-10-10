"""Keyword (BM25-style) search vectors for hybrid retrieval (alignment_plan RET-02).

Dense embeddings capture meaning but blur exact words: names, ids, numbers and rare terms. Keyword search is the
opposite. Hybrid search runs both and fuses the two rankings (Reciprocal Rank Fusion), which beat the dense-only
embedder clearly on a 10,000-passage benchmark (see README, Results).

How it works. Each chunk becomes a *sparse vector*: a short list of (term id, weight) pairs, one per distinct word.
The weight is BM25's term-frequency part (a word repeated many times counts more, with diminishing returns, and long
chunks are damped). Qdrant multiplies by the inverse document frequency (rare words matter more) on the server, because
the collection is created with the IDF modifier. A query is a sparse vector with weight 1 per distinct word.

Privacy. The term ids are NOT plain word hashes: they are an HMAC keyed with a sub-key of the data-encryption key, so
someone who can read the vector database cannot hash a dictionary of words and see which terms occur in which chunk.
They still reveal how many distinct terms a chunk has and which chunks share terms. Like the dense vectors this is
searchable-index leakage, documented in the README's security model.
"""
import hashlib
import hmac
import re

K1 = 1.2                     # BM25 term-frequency saturation
B = 0.75                     # BM25 length normalisation
AVG_LEN = 120.0              # assumed average chunk length in terms (a fixed constant keeps ingestion independent per document)
MAX_TERM_ID = 2 ** 31 - 1

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the of in on at to for and or is are was were be by with as from that this it its which who what when where how why did do does".split())


def tokenize(text: str):
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class SparseEncoder:
    """Turns text into (indices, values) sparse vectors. `key` is the secret that keys the term ids."""

    def __init__(self, key: bytes):
        if not key:
            raise ValueError("SparseEncoder needs a key")
        self._key = key
        self._ids = {}                    # token -> id cache (a vocabulary is small, HMAC is the slow part)

    def term_id(self, token: str) -> int:
        tid = self._ids.get(token)
        if tid is None:
            digest = hmac.new(self._key, token.encode("utf-8"), hashlib.sha256).digest()
            tid = self._ids[token] = int.from_bytes(digest[:4], "big") & MAX_TERM_ID
        return tid

    def document(self, text: str):
        """BM25 term-frequency weights for one chunk."""
        tokens = tokenize(text)
        if not tokens:
            return [], []
        counts = {}
        for t in tokens:
            tid = self.term_id(t)
            counts[tid] = counts.get(tid, 0) + 1
        norm = K1 * (1 - B + B * len(tokens) / AVG_LEN)
        indices = sorted(counts)
        return indices, [counts[i] * (K1 + 1) / (counts[i] + norm) for i in indices]

    def query(self, text: str):
        """Weight 1 for every distinct term of the question."""
        indices = sorted({self.term_id(t) for t in tokenize(text)})
        return indices, [1.0] * len(indices)
