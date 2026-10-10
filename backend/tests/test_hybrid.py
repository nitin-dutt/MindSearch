"""Keyword vectors and hybrid retrieval (alignment_plan RET-02)."""

import pytest

import sparse as SP

pytest.importorskip("sentence_transformers")
pytest.importorskip("qdrant_client")

from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402
from qdrant_client.models import Distance, VectorParams  # noqa: E402

import embedder  # noqa: E402
from access_policy import DocACL, get_default_policy  # noqa: E402
from encryptor import Encryptor  # noqa: E402
from pipeline_config import get_preset  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402

POLICY = get_default_policy()
EMPLOYEE = POLICY.make_user("emp", "Employee")
MANAGER = POLICY.make_user("mgr", "Manager")
EVERYONE = DocACL(min_rank=0)


# ------------------------------------------------------------------ the keyword encoder

def encoder(key=b"k" * 32):
    return SP.SparseEncoder(key)


def test_stopwords_are_dropped_and_case_is_ignored():
    assert SP.tokenize("The Eiffel Tower IS in Paris") == ["eiffel", "tower", "paris"]


def test_term_ids_are_stable_and_depend_on_the_key():
    a1, a2, b = encoder(b"a" * 32), encoder(b"a" * 32), encoder(b"b" * 32)
    assert a1.term_id("eiffel") == a2.term_id("eiffel")
    assert a1.term_id("eiffel") != b.term_id("eiffel")     # without the key the ids cannot be recomputed from a word list


def test_a_repeated_word_weighs_more_but_with_diminishing_returns():
    enc = encoder()
    idx1, w1 = enc.document("tower " + "filler " * 5)
    idx3, w3 = enc.document("tower tower tower " + "filler " * 5)
    t = enc.term_id("tower")
    one, three = w1[idx1.index(t)], w3[idx3.index(t)]
    assert one < three < 3 * one


def test_longer_chunks_are_damped():
    enc = encoder()
    short_i, short_w = enc.document("tower paris")
    long_i, long_w = enc.document("tower paris " + "word " * 300)
    t = enc.term_id("tower")
    assert short_w[short_i.index(t)] > long_w[long_i.index(t)]


def test_query_vector_has_weight_one_per_distinct_term_and_empty_text_is_empty():
    idx, w = encoder().query("Eiffel eiffel tower")
    assert len(idx) == 2 and set(w) == {1.0}
    assert encoder().query("the of and") == ([], [])
    assert encoder().document("") == ([], [])


def test_the_encryptor_derives_independent_subkeys():
    enc = Encryptor(key=AESGCM.generate_key(bit_length=256))
    assert enc.subkey("keyword-terms") == enc.subkey("keyword-terms")
    assert enc.subkey("keyword-terms") != enc.subkey("something-else") and len(enc.subkey("x")) == 32


# ------------------------------------------------------------------ hybrid retrieval end to end

@pytest.fixture
def make(tmp_path):
    made = []

    def build(preset="hybrid_rerank", **overrides):
        cfg = get_preset(preset).with_overrides(qdrant_path=str(tmp_path / "q"), collection="t", **overrides)
        made.append(cfg.qdrant_path)
        return RAGPipeline(cfg, encryptor=Encryptor(key=AESGCM.generate_key(bit_length=256)))

    yield build
    for path in made:
        embedder.close_qdrant_client(path)


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


FILLER = "\n".join(f"Quarterly note {i}: the committee reviewed routine operational matters and approved the usual budget." for i in range(30))


def test_hybrid_finds_an_exact_identifier_among_many_similar_passages(make, tmp_path):
    f = write(tmp_path, "d.md", FILLER + "\nPart ZX4471 is stored in warehouse seven.\n")
    rag = make("hybrid_rerank", chunk_size=10, chunk_overlap=0, final_k=1)
    rag.ingest_vectors(f, EVERYONE)
    top = rag.retrieve("Where is ZX4471 kept?", EMPLOYEE).chunks[0]["chunk"]
    assert "ZX4471" in top


def test_the_access_filter_applies_to_the_keyword_branch_too(make, tmp_path):
    """A restricted chunk that matches the question's exact words must not reach a lower-ranked user."""
    secret = write(tmp_path, "s.md", "Project Bluebird acquisition price is forty million.\n")
    public = write(tmp_path, "p.md", "The cafeteria opens at eight in the morning.\n")
    rag = make("hybrid_rerank")
    rag.ingest_document(secret, POLICY.make_acl("Manager"))
    rag.ingest_document(public, EVERYONE)
    leaked = [c["chunk"] for c in rag.retrieve("Bluebird acquisition price", EMPLOYEE).chunks]
    assert not any("Bluebird" in c for c in leaked)
    assert any("Bluebird" in c["chunk"] for c in rag.retrieve("Bluebird acquisition price", MANAGER).chunks)


def test_hybrid_and_dense_agree_when_the_question_has_no_keywords(make, tmp_path):
    f = write(tmp_path, "d.md", "The Eiffel Tower is located in Paris.\n")
    rag = make("hybrid_rerank")
    rag.ingest_vectors(f, EVERYONE)
    assert rag.retrieve("what is it?", EMPLOYEE).chunks            # all stop words: falls back to dense, still answers


def test_hybrid_search_needs_the_key_for_the_keyword_index(make, tmp_path):
    f = write(tmp_path, "d.md", "The Eiffel Tower is located in Paris.\n")
    rag = make("hybrid_rerank")
    rag.ingest_vectors(f, EVERYONE)
    with pytest.raises(ValueError):
        embedder.search_embeddings("Eiffel", EMPLOYEE, POLICY, rag.config.qdrant_path, "t", hybrid=True, encryptor=None)


def test_the_stored_keyword_vector_does_not_contain_plain_words(make, tmp_path):
    f = write(tmp_path, "d.md", "Project Bluebird acquisition price is forty million.\n")
    rag = make("hybrid_rerank")
    rag.ingest_vectors(f, EVERYONE)
    client = embedder.get_qdrant_client(rag.config.qdrant_path)
    pts, _ = client.scroll("t", limit=5, with_vectors=True)
    sparse_vec = pts[0].vector[embedder.KEYWORDS]
    plain = SP.SparseEncoder(b"\x00" * 32)                 # an attacker guessing the key / hashing words without it
    assert not set(sparse_vec.indices) & {plain.term_id(t) for t in SP.tokenize("bluebird acquisition price forty million")}


def test_an_index_from_before_hybrid_search_fails_with_a_clear_message(make, tmp_path):
    path = str(tmp_path / "q")
    client = embedder.get_qdrant_client(path)
    client.create_collection("t", vectors_config=VectorParams(size=embedder.EMBEDDING_DIM, distance=Distance.COSINE))
    rag = make("hybrid_rerank")
    f = write(tmp_path, "d.md", "The Eiffel Tower is located in Paris.\n")
    with pytest.raises(embedder.IndexSchemaError, match="Re-index"):
        rag.ingest_vectors(f, EVERYONE)
