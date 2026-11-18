"""No silent fallbacks (2026-11-17): a file that cannot be read, or a graph that could not be built, fails the document with the real
reason instead of being stored as empty, half-decoded or graph-less while showing 'done'."""
from types import SimpleNamespace

import pytest

pytest.importorskip("pdfplumber")

import chunker  # noqa: E402
from kg_extractors import ExtractionStats  # noqa: E402
from rag_pipeline import GraphExtractionError, RAGPipeline  # noqa: E402


def test_a_damaged_word_file_is_an_error_with_a_reason_not_an_empty_document(tmp_path):
    f = tmp_path / "report.docx"
    f.write_bytes(b"this is not a zip archive, so not a real .docx")
    with pytest.raises(chunker.DocumentReadError, match="DOCX file could not be read"):
        chunker.extract_document(str(f))


def test_a_damaged_pdf_is_an_error_with_a_reason(tmp_path):
    f = tmp_path / "scan.pdf"
    f.write_bytes(b"%PDF-1.7 truncated garbage")
    with pytest.raises(chunker.DocumentReadError, match="PDF file could not be read"):
        chunker.extract_document(str(f))


def test_a_confidently_detected_encoding_is_decoded_exactly(tmp_path):
    f = tmp_path / "notes.txt"
    text = "Café prices in München rose by 3 € this year, said Zoë. " * 20
    f.write_bytes(text.encode("utf-16"))                         # a byte-order mark: detected with full confidence
    assert chunker.extract_text_from_txt(str(f)) == text


def test_a_low_confidence_encoding_guess_is_refused_not_used(tmp_path):
    """Measured: this Windows-1252 text is guessed as cp850 at 11% confidence, which decodes without error into wrong letters."""
    f = tmp_path / "notes.txt"
    f.write_bytes(("Café prices in München rose by 3 € this year, said Zoë. " * 20).encode("cp1252"))
    with pytest.raises(chunker.DocumentReadError, match="could not be detected reliably"):
        chunker.extract_text_from_txt(str(f))


def test_a_latin_1_guess_is_read_as_windows_1252(tmp_path, monkeypatch):
    f = tmp_path / "notes.txt"
    f.write_bytes("Price: 3 € – “final”".encode("cp1252"))
    monkeypatch.setattr(chunker.chardet, "detect", lambda raw: {"encoding": "ISO-8859-1", "confidence": 0.73})
    assert chunker.extract_text_from_txt(str(f)) == "Price: 3 € – “final”"


def test_text_that_cannot_be_decoded_is_an_error_never_read_with_characters_dropped(tmp_path, monkeypatch):
    f = tmp_path / "notes.txt"
    f.write_bytes(b"\xff\xfe\xfa" * 50)
    monkeypatch.setattr(chunker.chardet, "detect", lambda raw: {"encoding": None})
    with pytest.raises(chunker.DocumentReadError, match="encoding"):
        chunker.extract_text_from_txt(str(f))


def fake_pipeline(stats):
    return SimpleNamespace(ingest_vectors=lambda path, acl, doc_id=None, report=None: ["passage one", "passage two"],
                           ingest_graph=lambda chunks, acl, doc_id=None: stats)


def test_too_many_failed_graph_passages_raise_with_the_counts_and_the_first_error(monkeypatch):
    monkeypatch.setenv("KG_MAX_FAILURE_RATE", "0.25")
    stats = ExtractionStats(extractor="ollama:qwen2.5:14b-instruct", chunks_ok=1, chunks_failed=3, first_error="ConnectError")
    with pytest.raises(GraphExtractionError, match=r"3 of 4 passages.*ConnectError.*ollama"):
        RAGPipeline.ingest_document(fake_pipeline(stats), "f.txt", acl=None, doc_id="d1")


def test_a_few_failed_graph_passages_are_kept_and_reported(monkeypatch):
    monkeypatch.setenv("KG_MAX_FAILURE_RATE", "0.25")
    stats = ExtractionStats(extractor="vllm:m", chunks_ok=9, chunks_failed=1, first_error="ExtractionError")
    out = RAGPipeline.ingest_document(fake_pipeline(stats), "f.txt", acl=None, doc_id="d1")
    assert out["graph"]["chunks_failed"] == 1 and out["graph"]["first_error"] == "ExtractionError"


def test_without_a_graph_there_is_nothing_to_check():
    out = RAGPipeline.ingest_document(fake_pipeline(None), "f.txt", acl=None, doc_id="d1")
    assert out["graph"] is None and out["chunks"] == 2
