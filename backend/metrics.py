"""Prometheus metrics for the request pipeline and ingestion (alignment_plan OBS-01).

The API already exposes the standard HTTP metrics (request count and latency per route). These add what those cannot show: WHERE a chat
request spends its time (embedding the question, searching, decrypting, re-ranking, the graph hop, the model), how requests end, and how
ingestion is going. Everything is exposed on the same `/metrics` endpoint (the API) or on the worker's own metrics port, scraped by
Prometheus and drawn by the provisioned Grafana dashboard.

Design notes:
  * Labels are a small FIXED set (stage names, outcomes): a metric must never be labelled with a user, a query or a document id, or the
    number of time series would grow without bound (and the labels would leak information).
  * If `prometheus_client` is not installed (a minimal local environment) every metric is a no-op, so importing this module never breaks
    the pipeline.
"""
import time
from contextlib import contextmanager

try:
    from prometheus_client import Counter, Histogram
    ENABLED = True
except ImportError:                                     # pragma: no cover - exercised only without the library
    ENABLED = False

    class _NoOp:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    def Counter(*args, **kwargs):                       # noqa: N802 - mirrors the real class name
        return _NoOp()

    def Histogram(*args, **kwargs):                     # noqa: N802
        return _NoOp()

# seconds: from 5 ms (a vector search) to 60 s (a very slow model)
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)

# The pipeline stages of one chat request, in the order they happen.
STAGES = ("rewrite", "embed_query", "vector_search", "decrypt", "rerank", "graph", "retrieve_total", "generation")

STAGE_SECONDS = Histogram("mindsearch_stage_seconds", "Time spent in one stage of a chat request", ["stage"], buckets=_BUCKETS)
TTFT_SECONDS = Histogram("mindsearch_time_to_first_token_seconds",
                         "From the chat request arriving to the first word of the answer being ready", buckets=_BUCKETS)

# How chat requests end. answered: the model answered; abstained_model: the model said "I don't know"; refused_gate: the relevance gate
# refused before the model was called; no_context: nothing retrievable; failed: the model was unavailable (HTTP 503);
# interrupted: the model died mid-answer.
CHAT_OUTCOMES = ("answered", "abstained_model", "refused_gate", "no_context", "failed", "interrupted")
CHAT_REQUESTS = Counter("mindsearch_chat_requests_total", "Chat requests by how they ended", ["outcome"])

DECRYPT_FAILURES = Counter("mindsearch_decrypt_failures_total",
                           "Stored chunks that failed their integrity check and were skipped (possible tampering)")
INVALID_CITATIONS = Counter("mindsearch_invalid_citations_total", "Citation numbers in answers that point at no shown passage")

INGEST_DOCUMENTS = Counter("mindsearch_ingest_documents_total", "Documents processed by the worker", ["status"])   # done, failed, retried
INGEST_CHUNKS = Counter("mindsearch_ingest_chunks_total", "Chunks written by the worker")
INGEST_SECONDS = Histogram("mindsearch_ingest_seconds", "Time to ingest one document", buckets=(0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600))
KG_CHUNKS_FAILED = Counter("mindsearch_kg_chunks_failed_total", "Chunks whose graph extraction failed after retries")


@contextmanager
def timed(stage: str):
    """`with timed("rerank"): ...` records how long the block took under that stage name."""
    start = time.perf_counter()
    try:
        yield
    finally:
        STAGE_SECONDS.labels(stage).observe(time.perf_counter() - start)


REWRITE_RETRIES = Counter("mindsearch_rewrite_retries_total",
                          "Searches repeated with a rewritten question after the first search found nothing usable", ["result"])


def chat_outcome(outcome: str):
    CHAT_REQUESTS.labels(outcome).inc()
