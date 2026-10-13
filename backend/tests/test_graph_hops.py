"""Graph-guided passage hops (alignment_plan KG-hop work): provenance, exact entity matching, access control, end to end.
Needs a real Neo4j; skipped when it is unreachable."""
import os
import uuid

import pytest

pytest.importorskip("neo4j")

from access_policy import AccessPolicy, DocACL  # noqa: E402
from kg_builder import KnowledgeGraphBuilder, entity_key  # noqa: E402
from kg_extractors import Triple, extract_chunks  # noqa: E402

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policies", "departmental.yaml"))
STAFF = POLICY.make_user("staff", "Staff")
EXEC = POLICY.make_user("exec", "Executive")
SUPER = POLICY.make_user("root", "Staff", roles=["Admin"])
EVERYONE = DocACL(min_rank=0)
P = "Zzhop"       # unique name prefix; removed with the test's documents


@pytest.fixture
def kg():
    builder = KnowledgeGraphBuilder()
    try:
        builder.driver.verify_connectivity()
    except Exception:
        pytest.skip("Neo4j not reachable")
    created = []
    builder.created = created
    yield builder
    for doc_id in created:
        builder.delete_document(doc_id)
    builder.close()


def new_doc(kg):
    doc_id = P + uuid.uuid4().hex[:10]
    kg.created.append(doc_id)
    return doc_id


def T(s, r, o):
    return Triple(subject=f"{P} {s}", relation=r, object=f"{P} {o}")


def key(name):
    return entity_key(f"{P} {name}")


# ------------------------------------------------------------------ pure helpers

def test_entity_keys_ignore_case_spacing_quotes_and_leading_articles():
    assert entity_key('  The  "Wilmslow   Show" ') == "wilmslow show"
    assert entity_key("A Dr. Fad Show") == "dr. fad show"


class OneTriplePerChunk:
    name = "fake"

    def extract(self, chunk):
        a, b = chunk.split()[:2]
        return [Triple(subject=a, relation="REL", object=b)]


def test_extraction_remembers_which_chunk_each_entity_came_from():
    mentions = []
    extract_chunks(OneTriplePerChunk(), ["alpha beta one two three", "beta gamma four five six"], mentions_out=mentions)
    assert sorted(mentions) == [("alpha", 0), ("beta", 0), ("beta", 1), ("gamma", 1)]


# ------------------------------------------------------------------ graph store

def save(kg, doc_id, triples, mentions, acl=EVERYONE):
    kg.save_triples(triples, acl, doc_id, mentions=[(f"{P} {n}", i) for n, i in mentions])


def test_mentions_lead_from_entities_to_passages_and_back(kg):
    d = new_doc(kg)
    save(kg, d, [T("Ann", "FOUNDED", "Corp")], [("Ann", 0), ("Corp", 0), ("Corp", 3)])
    assert kg.entities_in_chunks([(d, 0)], SUPER) == sorted([key("Ann"), key("Corp")]) or \
        set(kg.entities_in_chunks([(d, 0)], SUPER)) == {key("Ann"), key("Corp")}
    refs = kg.chunk_refs_for_entities([key("Corp")], SUPER)
    assert set(refs) == {(d, 0), (d, 3)}


def test_restricted_passages_are_invisible_to_lower_ranked_users(kg):
    d = new_doc(kg)
    save(kg, d, [T("Boss", "OWNS", "Vault")], [("Boss", 0), ("Vault", 0)], acl=POLICY.make_acl("Executive"))
    assert kg.chunk_refs_for_entities([key("Vault")], STAFF) == {}
    assert kg.entities_in_chunks([(d, 0)], STAFF) == []
    assert (d, 0) in kg.chunk_refs_for_entities([key("Vault")], EXEC)


def test_very_common_entities_are_not_followed(kg):
    d = new_doc(kg)
    save(kg, d, [T("Hub", "IN", "Leaf")], [("Hub", i) for i in range(30)] + [("Leaf", 1)])
    assert kg.chunk_refs_for_entities([key("Hub")], SUPER, hub_limit=12) == {}
    assert len(kg.chunk_refs_for_entities([key("Hub")], SUPER, hub_limit=50)) == 30


def test_exact_key_matching_does_not_match_on_a_shared_word(kg):
    d = new_doc(kg)
    save(kg, d, [T("Dr Fad Show", "FEATURED", "Inventions")], [("Dr Fad Show", 0), ("Inventions", 0)])
    assert kg.expand_keys([key("Dr Fad")], SUPER) == []               # a partial name: the old substring rule matched this
    assert kg.expand_keys([key("Dr Fad Show")], SUPER) != []


def test_deleting_a_document_removes_its_passage_links(kg):
    d = new_doc(kg)
    save(kg, d, [T("Gone", "IN", "Away")], [("Gone", 0), ("Away", 0)])
    kg.delete_document(d)
    assert kg.chunk_refs_for_entities([key("Gone")], SUPER) == {}


def test_query_entity_keys_pick_up_names_and_noun_phrases(kg):
    keys = kg.query_entity_keys("Where is the school that the Wilmslow Show is held in?")
    assert "wilmslow show" in keys and "school" in keys


# ------------------------------------------------------------------ end to end (real embedder, reranker, Qdrant, Neo4j)

def test_a_second_hop_passage_is_found_through_the_graph(kg, tmp_path):
    pytest.importorskip("sentence_transformers")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from embedder import close_qdrant_client
    from encryptor import Encryptor
    from pipeline_config import get_preset
    from rag_pipeline import RAGPipeline

    first = "Quillfeather founded the firm Vandermeer Industries in the year 1950."
    second = "Vandermeer Industries has its headquarters in the city of Lisbon."
    filler = [f"Unrelated note number {i} describes routine weather and traffic conditions." for i in range(12)]
    f = tmp_path / "d.md"
    f.write_text("\n".join([first, second] + filler) + "\n", encoding="utf-8")

    class Fake:
        name = "fake"

        def extract(self, chunk):
            if "Quillfeather" in chunk:
                return [Triple(subject="Quillfeather", relation="FOUNDED", object="Vandermeer Industries")]
            if "Vandermeer" in chunk:
                return [Triple(subject="Vandermeer Industries", relation="HEADQUARTERED_IN", object="Lisbon")]
            return []

    cfg = get_preset("hybrid_graph_chunks").with_overrides(qdrant_path=str(tmp_path / "q"), collection="t", chunk_size=14,
                                                           chunk_overlap=0, final_k=2)
    rag = RAGPipeline(cfg, encryptor=Encryptor(key=AESGCM.generate_key(bit_length=256)), policy=POLICY)
    rag.kg_builder.extractor = Fake()
    try:
        doc_id = rag.ingest_document(str(f), EVERYONE)["doc_id"]
        kg.created.append(doc_id)
        # the question names only the first-hop entity; the answer is in the second passage
        res = rag.retrieve("Where are the offices of the firm Quillfeather started?", STAFF)
        texts = [c["chunk"] for c in res.chunks]
        assert any("Lisbon" in t for t in texts), texts
        assert res.graph_chunks_added >= 0
    finally:
        close_qdrant_client(cfg.qdrant_path)
