"""PERF-01: with graph hops on, the search pool is scored by the cross-encoder ONCE per question, not twice.
Real embedder and embedded Qdrant; the graph and the cross-encoder are fakes so the test counts exactly what was scored."""
import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("qdrant_client")

from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402

from access_policy import DocACL, get_default_policy  # noqa: E402
from vector_db import close_qdrant_client  # noqa: E402
from encryptor import Encryptor  # noqa: E402
from pipeline_config import get_preset  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402
from retriever import retrieve_detailed  # noqa: E402

POLICY = get_default_policy()
ADMIN = POLICY.make_user("admin", "Employee", roles=["Admin"])
LINES = [f"Passage number {i} talks about topic {i} " + "word " * i for i in range(1, 5)]


class CountingReranker:
    """Scores a passage by its length (deterministic) and remembers every (question, passage) pair it was asked to score."""
    def __init__(self):
        self.pairs = []

    def rerank(self, query, chunks, top_k=5):
        self.pairs += [(query, c) for c in chunks]
        rows = [{"chunk": c, "score": float(len(c)), "original_index": i} for i, c in enumerate(chunks)]
        return sorted(rows, key=lambda r: -r["score"])[:top_k]


class NoGraph:
    """A graph that knows no entities: the hop step runs, adds nothing, and must not cost a second scoring pass."""
    def entities_in_chunks(self, refs, user):
        return []

    def query_entity_keys(self, query):
        return []

    def candidate_chunks(self, keys, user, hub_limit):
        return {}

    def entities_by_chunk(self, refs, user):
        return {}


@pytest.fixture
def rag(tmp_path):
    cfg = get_preset("hybrid_graph_chunks").with_overrides(qdrant_path=str(tmp_path / "q"), collection="t", use_graph=False,
                                                           use_reranker=False, chunk_size=12, chunk_overlap=0, final_k=2)
    pipeline = RAGPipeline(cfg, encryptor=Encryptor(key=AESGCM.generate_key(bit_length=256)), policy=POLICY)
    f = tmp_path / "d.md"
    f.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    pipeline.ingest_document(str(f), DocACL(min_rank=0))
    pipeline.kg_builder, pipeline.reranker = NoGraph(), CountingReranker()
    yield pipeline
    close_qdrant_client(cfg.qdrant_path)


def test_each_pool_passage_is_scored_exactly_once_per_question(rag):
    result = rag.retrieve("topic", ADMIN)
    scored = [c for _, c in rag.reranker.pairs]
    assert len(result.dense_pool) >= len(LINES)
    assert sorted(scored) == sorted(result.dense_pool)            # once each: no passage appears twice
    assert len(result.chunks) == 2


def test_reusing_the_scores_gives_the_same_passages_as_scoring_again(rag):
    """The speed-up must not change what is retrieved: the hop step fed with the saved scores picks the same passages, with the same
    cross-encoder scores, as the hop step that scores the pool itself (the behaviour before PERF-01)."""
    cfg = rag.config
    detail = retrieve_detailed("topic", rag.store, ADMIN, candidate_k=cfg.candidate_k, final_k=cfg.final_k, reranker=rag.reranker,
                               hybrid=cfg.hybrid)
    assert [r["original_index"] for r in detail["scored"]] == sorted(range(len(detail["pool"])), key=lambda i: -len(detail["pool"][i]))
    reused, _ = rag.graph_candidates("topic", ADMIN, detail)
    again, _ = rag.graph_candidates("topic", ADMIN, {**detail, "scored": []})
    assert [(c["chunk"], c["ce"]) for c in reused] == [(c["chunk"], c["ce"]) for c in again]
    assert [c["chunk"] for c in rag.graph_select(reused)] == [c["chunk"] for c in rag.graph_select(again)]


def test_without_a_reranker_there_are_no_scores_and_no_hop_step(rag):
    cfg = rag.config
    detail = retrieve_detailed("topic", rag.store, ADMIN, candidate_k=cfg.candidate_k, final_k=cfg.final_k, reranker=None)
    assert detail["scored"] == [] and len(detail["results"]) == 2


def test_rerank_once_off_brings_back_the_second_pass_with_the_same_passages(rag, monkeypatch):
    """RERANK_ONCE=0 exists only for the before/after load test: it must cost the old second scoring pass and nothing else."""
    once = [c for c in rag.retrieve("topic", ADMIN).chunks]
    rag.reranker.pairs.clear()
    monkeypatch.setenv("RERANK_ONCE", "0")
    result = rag.retrieve("topic", ADMIN)
    scored = [c for _, c in rag.reranker.pairs]
    assert sorted(scored) == sorted(result.dense_pool * 2)          # every pool passage scored twice
    assert [c["chunk"] for c in result.chunks] == [c["chunk"] for c in once]
