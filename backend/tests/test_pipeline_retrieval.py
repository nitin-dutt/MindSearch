"""Retrieval behaviour on a tiny real index (real embedder + reranker, ephemeral encryption key).
Skipped automatically when the model libraries are not installed."""
import os

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("qdrant_client")

from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402

from access_policy import AccessPolicy, DocACL, get_default_policy  # noqa: E402
from embedder import close_qdrant_client  # noqa: E402
from encryptor import Encryptor  # noqa: E402
from pipeline_config import PRESETS, PipelineConfig, get_preset  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = get_default_policy()                       # Employee(1) < Manager(2); super role: Admin
ADMIN = POLICY.make_user("admin", "Employee", roles=["Admin"])
EMPLOYEE = POLICY.make_user("emp", "Employee")
MANAGER = POLICY.make_user("mgr", "Manager")
EVERYONE = DocACL(min_rank=0)

CORPUS = (
    "# Tiny corpus\n\n"
    "The Eiffel Tower is located in Paris and was completed in 1889 for the World's Fair.\n\n"
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen in plants.\n\n"
    "The Treaty of Westphalia was signed in 1648 and ended the Thirty Years' War in Europe.\n\n"
)


@pytest.fixture
def pipeline_factory(tmp_path):
    made = []

    def make(config_name="dense_only", policy=None, **overrides):
        cfg = get_preset(config_name).with_overrides(qdrant_path=str(tmp_path / "q"), collection="t", **overrides)
        made.append(cfg.qdrant_path)
        return RAGPipeline(cfg, encryptor=Encryptor(key=AESGCM.generate_key(bit_length=256)), policy=policy)

    yield make
    for path in made:
        close_qdrant_client(path)


@pytest.fixture
def corpus_file(tmp_path):
    p = tmp_path / "corpus.md"
    p.write_text(CORPUS, encoding="utf-8")
    return str(p)


def test_empty_index_returns_no_context_not_placeholder_text(pipeline_factory):
    rag = pipeline_factory()
    result = rag.retrieve("Where is the Eiffel Tower?", ADMIN)
    assert result.chunks == [] and result.dense_pool == []
    assert rag.retrieve_context("Where is the Eiffel Tower?", ADMIN) == []


def test_dense_only_finds_relevant_chunk_and_wraps_in_doc_tags(pipeline_factory, corpus_file):
    rag = pipeline_factory()
    rag.ingest_document(corpus_file, EVERYONE)
    ctx = rag.retrieve_context("When was the Eiffel Tower completed?", EMPLOYEE)
    assert ctx and all(c.startswith("<doc>") and c.endswith("</doc>") for c in ctx)
    assert "1889" in ctx[0]


def test_final_k_is_honored(pipeline_factory, corpus_file):
    rag = pipeline_factory(chunk_size=15, chunk_overlap=0, final_k=2)
    rag.ingest_document(corpus_file, EVERYONE)
    result = rag.retrieve("treaty", EMPLOYEE)
    assert len(result.chunks) == 2
    assert len(result.dense_pool) >= 3


def test_clearance_levels_are_hierarchical(pipeline_factory, corpus_file):
    rag = pipeline_factory()
    rag.ingest_document(corpus_file, POLICY.make_acl("Manager"))     # manager-level documents
    assert rag.retrieve("Eiffel Tower", EMPLOYEE).chunks == []        # lower rank: nothing
    assert rag.retrieve("Eiffel Tower", MANAGER).chunks != []         # rank equal: allowed
    assert rag.retrieve("Eiffel Tower", ADMIN).chunks != []           # super role: allowed


def test_manager_can_read_employee_documents(pipeline_factory, corpus_file):
    """The old equality check ('Manager' != 'Employee') hid these; the hierarchy must not."""
    rag = pipeline_factory()
    rag.ingest_document(corpus_file, POLICY.make_acl("Employee"))
    assert rag.retrieve("Eiffel Tower", MANAGER).chunks != []


def test_department_walls_with_a_departmental_policy(pipeline_factory, corpus_file):
    policy = AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policies", "departmental.yaml"))
    rag = pipeline_factory(policy=policy)
    rag.ingest_document(corpus_file, policy.make_acl("Staff", groups=["HR"]))
    hr_staff = policy.make_user("a", "Staff", ["HR"])
    legal_exec = policy.make_user("b", "Executive", ["Legal"])          # higher rank, wrong department
    assert rag.retrieve("Eiffel Tower", hr_staff).chunks != []
    assert rag.retrieve("Eiffel Tower", legal_exec).chunks == []
    shared = policy.make_user("c", "Staff", ["Legal", "HR"])            # member of several departments
    assert rag.retrieve("Eiffel Tower", shared).chunks != []


def test_search_requires_an_explicit_user(pipeline_factory, corpus_file):
    rag = pipeline_factory()
    rag.ingest_document(corpus_file, EVERYONE)
    with pytest.raises(TypeError):
        rag.retrieve("Eiffel Tower")          # no silent "see everything" default


def test_dense_rerank_keeps_a_reranked_top_k(pipeline_factory, corpus_file):
    rag = pipeline_factory("dense_rerank", final_k=2)
    rag.ingest_document(corpus_file, EVERYONE)
    result = rag.retrieve("Which treaty ended the Thirty Years' War?", EMPLOYEE)
    assert "Westphalia" in result.chunks[0]["chunk"]
    assert len(result.chunks) <= 2


def test_dense_only_pipeline_has_no_graph_or_reranker(pipeline_factory):
    rag = pipeline_factory()
    assert rag.reranker is None and rag.kg_builder is None


def test_presets_are_the_three_ablation_configs():
    assert {"dense_only", "dense_rerank", "full"} <= set(PRESETS)
    assert PRESETS["full"].use_graph and PRESETS["full"].use_reranker
    assert not PRESETS["dense_only"].use_reranker and not PRESETS["dense_only"].use_graph
    with pytest.raises(ValueError):
        get_preset("nope")
    assert isinstance(PRESETS["full"], PipelineConfig)


class _FakeReranker:
    """Returns pre-set scores so the cutoff logic can be tested without a model."""
    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, chunks, top_k=5):
        rows = [{"chunk": c, "score": s, "original_index": i} for i, (c, s) in enumerate(zip(chunks, self.scores))]
        return sorted(rows, key=lambda r: -r["score"])[:top_k]


def _retrieve_with(scores, margin, pipeline_factory, corpus_file):
    from retriever import retrieve
    rag = pipeline_factory("dense_only", chunk_size=15, chunk_overlap=0)
    rag.ingest_document(corpus_file, EVERYONE)
    cfg = rag.config
    results, _ = retrieve("anything", cfg.qdrant_path, ADMIN, rag.policy, candidate_k=3, final_k=3,
                          reranker=_FakeReranker(scores), encryptor=rag.encryptor,
                          collection_name=cfg.collection, rerank_margin=margin)
    return results


def test_dynamic_cutoff_keeps_only_chunks_close_to_the_best(pipeline_factory, corpus_file):
    assert len(_retrieve_with([9.0, 8.0, -3.0], 4.0, pipeline_factory, corpus_file)) == 2   # drops the far one
    assert len(_retrieve_with([9.0, -3.0, -5.0], 4.0, pipeline_factory, corpus_file)) == 1  # keeps only the best


def test_cutoff_off_keeps_final_k_and_always_keeps_the_best(pipeline_factory, corpus_file):
    assert len(_retrieve_with([9.0, -3.0, -5.0], None, pipeline_factory, corpus_file)) == 3
    out = _retrieve_with([-8.0, -20.0, -30.0], 4.0, pipeline_factory, corpus_file)
    assert len(out) == 1   # even a poor best chunk is kept; the rule is relative, not absolute


def test_dynamic_presets_exist_and_use_a_margin():
    assert PRESETS["dense_rerank_dynamic"].rerank_margin == 4.0
    assert PRESETS["dense_rerank_fast"].candidate_k == 10


# ------------------------------------------------ tamper resistance (SEC-07)

def _all_points(rag):
    from embedder import get_qdrant_client
    client = get_qdrant_client(rag.config.qdrant_path)
    points, _ = client.scroll(rag.config.collection, limit=100, with_payload=True)
    return client, points


def test_lowering_a_documents_access_level_in_the_database_does_not_expose_it(pipeline_factory, corpus_file):
    """Someone with database write access edits min_rank from 2 to 0. The ACL filter would now pass the
    chunk for an Employee, but the ciphertext was bound to the old rule, so it cannot be decrypted."""
    rag = pipeline_factory()
    rag.ingest_document(corpus_file, POLICY.make_acl("Manager"))
    client, points = _all_points(rag)
    client.set_payload(rag.config.collection, payload={"min_rank": 0}, points=[p.id for p in points])
    assert rag.retrieve("Eiffel Tower", EMPLOYEE).chunks == []
    assert rag.retrieve("Eiffel Tower", ADMIN).chunks == []        # even for a super role: tampered = not served


def test_copying_restricted_ciphertext_onto_a_public_record_does_not_leak_it(pipeline_factory, tmp_path):
    secret = tmp_path / "secret.md"
    secret.write_text("The merger with Contoso will be announced on Friday.\n", encoding="utf-8")
    public = tmp_path / "public.md"
    public.write_text("The cafeteria opens at eight in the morning.\n", encoding="utf-8")
    rag = pipeline_factory()
    rag.ingest_document(str(secret), POLICY.make_acl("Manager"))
    rag.ingest_document(str(public), EVERYONE)
    client, points = _all_points(rag)
    restricted = next(p for p in points if p.payload["min_rank"] == 2)
    open_pt = next(p for p in points if p.payload["min_rank"] == 0)
    # attacker moves the restricted ciphertext into the employee-visible record
    client.set_payload(rag.config.collection, payload={"chunk_data": restricted.payload["chunk_data"]},
                       points=[open_pt.id])
    leaked = [c["chunk"] for c in rag.retrieve("When is the merger announced?", EMPLOYEE).chunks]
    assert not any("Contoso" in c for c in leaked)


def test_untampered_data_still_decrypts_after_the_integrity_checks(pipeline_factory, corpus_file):
    rag = pipeline_factory()
    rag.ingest_document(corpus_file, POLICY.make_acl("Manager"))
    assert any("1889" in c["chunk"] for c in rag.retrieve("When was the Eiffel Tower completed?", MANAGER).chunks)


# ------------------------------------------------------ idempotent ingestion (INF-03)

def _count(rag):
    from embedder import get_qdrant_client
    return get_qdrant_client(rag.config.qdrant_path).count(rag.config.collection).count


def test_ingesting_the_same_file_twice_does_not_duplicate_chunks(pipeline_factory, corpus_file):
    rag = pipeline_factory(chunk_size=15, chunk_overlap=0)
    id1 = rag.ingest_document(corpus_file, EVERYONE)["doc_id"]
    n = _count(rag)
    id2 = rag.ingest_document(corpus_file, EVERYONE)["doc_id"]        # a re-upload, or a Celery retry
    assert id1 == id2 and _count(rag) == n and n >= 3
    texts = [c["chunk"] for c in rag.retrieve("Eiffel Tower", EMPLOYEE).chunks]
    assert len(texts) == len(set(texts))                     # no duplicate chunks reach the prompt


def test_a_retry_after_a_partial_failure_leaves_one_copy(pipeline_factory, corpus_file):
    """Vectors written, then the graph step 'fails' and the whole task is retried."""
    rag = pipeline_factory(chunk_size=15, chunk_overlap=0)
    doc_id = rag.ingest_document(corpus_file, EVERYONE)["doc_id"]
    n = _count(rag)
    rag.ingest_vectors(corpus_file, EVERYONE)                # retry re-runs the vector stage
    assert _count(rag) == n and doc_id


def test_same_file_with_a_different_access_rule_is_a_separate_document(pipeline_factory, corpus_file):
    rag = pipeline_factory(chunk_size=15, chunk_overlap=0)
    a = rag.ingest_document(corpus_file, EVERYONE)["doc_id"]
    n = _count(rag)
    b = rag.ingest_document(corpus_file, POLICY.make_acl("Manager"))["doc_id"]
    assert a != b and _count(rag) == 2 * n                   # permissions are never merged or overwritten


def test_points_carry_their_document_identity(pipeline_factory, corpus_file):
    from embedder import get_qdrant_client
    rag = pipeline_factory(chunk_size=15, chunk_overlap=0)
    doc_id = rag.ingest_document(corpus_file, EVERYONE)["doc_id"]
    points, _ = get_qdrant_client(rag.config.qdrant_path).scroll(rag.config.collection, limit=50, with_payload=True)
    assert {p.payload["doc_id"] for p in points} == {doc_id}
    assert sorted(p.payload["chunk_index"] for p in points) == list(range(len(points)))


# ------------------------------------------------ source links and deletion (INF-04)

def test_retrieved_chunks_point_back_to_their_document(pipeline_factory, corpus_file):
    rag = pipeline_factory(chunk_size=15, chunk_overlap=0)
    info = rag.ingest_document(corpus_file, EVERYONE)
    result = rag.retrieve("When was the Eiffel Tower completed?", EMPLOYEE)
    assert result.chunks and all(c["doc_id"] == info["doc_id"] and isinstance(c["chunk_index"], int) for c in result.chunks)
    assert result.sources() == [{"doc_id": info["doc_id"], "chunk_index": result.chunks[0]["chunk_index"], "page": None}]   # a .md file has no pages


def test_ingest_reports_how_many_chunks_were_written(pipeline_factory, corpus_file):
    rag = pipeline_factory(chunk_size=15, chunk_overlap=0)
    info = rag.ingest_document(corpus_file, EVERYONE)
    assert info["chunks"] == _count(rag) >= 3


def test_deleting_a_document_removes_its_chunks_and_registry_entry(pipeline_factory, corpus_file, tmp_path):
    other = tmp_path / "other.md"
    other.write_text("The Great Wall of China was built over many centuries to protect northern borders.\n", encoding="utf-8")
    rag = pipeline_factory(chunk_size=15, chunk_overlap=0)
    first = rag.ingest_document(corpus_file, EVERYONE)
    second = rag.ingest_document(str(other), EVERYONE)
    rag.registry.register(first["doc_id"], EVERYONE, "corpus.md", "admin", 10)
    rag.registry.register(second["doc_id"], EVERYONE, "other.md", "admin", 10)

    report = rag.delete_document(first["doc_id"])
    assert report["chunks_deleted"] == first["chunks"]
    assert rag.retrieve("Eiffel Tower", EMPLOYEE).chunks == [] or all(
        c["doc_id"] == second["doc_id"] for c in rag.retrieve("Eiffel Tower", EMPLOYEE).chunks)
    assert rag.registry.get(first["doc_id"]) is None
    assert rag.registry.get(second["doc_id"]) is not None                       # the other document is untouched
    assert any("Great Wall" in c["chunk"] for c in rag.retrieve("Great Wall of China", EMPLOYEE).chunks)
    assert _count(rag) == second["chunks"]


def test_deleting_twice_is_harmless(pipeline_factory, corpus_file):
    rag = pipeline_factory(chunk_size=15, chunk_overlap=0)
    info = rag.ingest_document(corpus_file, EVERYONE)
    rag.registry.register(info["doc_id"], EVERYONE, "c.md", "admin", 1)
    rag.delete_document(info["doc_id"])
    again = rag.delete_document(info["doc_id"])
    assert again["chunks_deleted"] == 0 and _count(rag) == 0
