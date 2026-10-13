"""Page numbers travel from a PDF through chunking, storage and retrieval to the sources list."""
import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("qdrant_client")

from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402

import embedder  # noqa: E402
from access_policy import DocACL, get_default_policy  # noqa: E402
from encryptor import Encryptor  # noqa: E402
from pipeline_config import get_preset  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402
from tests.test_chunker import make_pdf, wrap  # noqa: E402

POLICY = get_default_policy()
EMPLOYEE = POLICY.make_user("emp", "Employee")
EVERYONE = DocACL(min_rank=0)


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


def test_a_pdf_answer_cites_the_right_page(make, tmp_path):
    pdf = tmp_path / "report.pdf"
    make_pdf(pdf, [wrap("The cafeteria serves breakfast from seven to nine every morning in the main hall."),
                   wrap("The warehouse inventory was audited in March and the totals matched the ledger."),
                   wrap("Parking permits are renewed every January at the front desk by the security team.")])
    rag = make(chunk_size=20, chunk_overlap=0, final_k=1)
    rag.ingest_document(str(pdf), EVERYONE)
    src = rag.retrieve("When are the parking permits renewed?", EMPLOYEE).sources()
    assert src and src[0]["page"] == 3


def test_text_files_have_no_page_in_their_sources(make, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("The cafeteria opens at eight in the morning.\n", encoding="utf-8")
    rag = make(final_k=1)
    rag.ingest_document(str(f), EVERYONE)
    assert rag.retrieve("When does the cafeteria open?", EMPLOYEE).sources()[0]["page"] is None
