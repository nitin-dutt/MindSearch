"""Pipeline metrics (OBS-01): stages are timed, outcomes are counted, and no label can grow without bound."""
import pytest

pytest.importorskip("prometheus_client")

from prometheus_client import REGISTRY  # noqa: E402

import metrics as MX  # noqa: E402


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_timed_records_one_observation_for_the_named_stage():
    before = sample("mindsearch_stage_seconds_count", stage="rerank")
    with MX.timed("rerank"):
        pass
    assert sample("mindsearch_stage_seconds_count", stage="rerank") == before + 1


def test_timed_still_records_when_the_stage_fails():
    before = sample("mindsearch_stage_seconds_count", stage="decrypt")
    with pytest.raises(ValueError):
        with MX.timed("decrypt"):
            raise ValueError("boom")
    assert sample("mindsearch_stage_seconds_count", stage="decrypt") == before + 1


def test_outcomes_are_counted_per_kind():
    before = sample("mindsearch_chat_requests_total", outcome="refused_gate")
    MX.chat_outcome("refused_gate")
    assert sample("mindsearch_chat_requests_total", outcome="refused_gate") == before + 1


def test_the_known_stages_and_outcomes_are_a_small_fixed_set():
    """Labels must never carry a user, a query or a document id: the set of values is fixed and small."""
    assert len(MX.STAGES) <= 10 and len(MX.CHAT_OUTCOMES) <= 8


def test_a_retrieval_times_the_search_stages(tmp_path):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("qdrant_client")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    import embedder
    from access_policy import DocACL, get_default_policy
    from encryptor import Encryptor
    from pipeline_config import get_preset
    from rag_pipeline import RAGPipeline

    cfg = get_preset("dense_rerank").with_overrides(qdrant_path=str(tmp_path / "q"), collection="t")
    rag = RAGPipeline(cfg, encryptor=Encryptor(key=AESGCM.generate_key(bit_length=256)))
    f = tmp_path / "d.md"
    f.write_text("The cafeteria opens at eight in the morning.\n", encoding="utf-8")
    names = ("embed_query", "vector_search", "decrypt", "rerank")
    before = {s: sample("mindsearch_stage_seconds_count", stage=s) for s in names}
    try:
        rag.ingest_document(str(f), DocACL(min_rank=0))
        rag.retrieve("When does the cafeteria open?", get_default_policy().make_user("emp", "Employee"))
    finally:
        embedder.close_qdrant_client(cfg.qdrant_path)
    for s in names:
        assert sample("mindsearch_stage_seconds_count", stage=s) > before[s], s


def test_a_tampered_chunk_is_counted(tmp_path):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    from encryptor import Encryptor
    from retriever import decrypt_hits

    enc = Encryptor(key=AESGCM.generate_key(bit_length=256))
    good = enc.encrypt("fine", aad=b"a")
    before = sample("mindsearch_decrypt_failures_total")
    texts, _, _ = decrypt_hits([good, good], [b"a", b"WRONG"], [1.0, 0.5], [{}, {}], enc)
    assert texts == ["fine"] and sample("mindsearch_decrypt_failures_total") == before + 1
