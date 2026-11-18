"""The worker's half of a document's life (ingestion.Ingestion.run), with a fake pipeline: no models, no databases, no Celery.
The API's half (`submit`: duplicates, a queue that is down) is covered through the HTTP tests in test_api.py."""
import pytest

from ingestion import Ingestion


class FakeRegistry:
    def __init__(self):
        self.history = []                       # every status the document went through, with its fields

    def set_status(self, doc_id, status, **fields):
        self.history.append((status, fields))


class FakeRag:
    def __init__(self, chunks=3, error=None):
        self.registry, self.chunks, self.error = FakeRegistry(), chunks, error
        self.removed = []

    def ingest_document(self, path, acl, doc_id=None):
        if self.error:
            raise self.error
        return {"doc_id": doc_id, "chunks": self.chunks, "extraction": {}, "graph": None}

    def remove_partial(self, doc_id):
        self.removed.append(doc_id)


@pytest.fixture
def upload(tmp_path):
    f = tmp_path / "upload.txt"
    f.write_text("plain text waiting to be ingested", encoding="utf-8")
    return f


def counted(status):
    prometheus = pytest.importorskip("prometheus_client")
    return prometheus.REGISTRY.get_sample_value("mindsearch_ingest_documents_total", {"status": status}) or 0.0


def test_a_processed_document_ends_as_done_and_its_plaintext_file_is_removed(upload):
    rag = FakeRag(chunks=3)
    result = Ingestion(rag).run(str(upload), acl=None, doc_id="d1")
    assert result == {"status": "success", "file": "upload.txt", "doc_id": "d1", "chunks": 3}
    assert [s for s, _ in rag.registry.history] == ["processing", "done"]
    assert rag.registry.history[-1][1]["chunks"] == 3
    assert not upload.exists()


def test_a_failure_that_will_be_retried_keeps_the_file_and_is_counted_as_a_retry_only(upload):
    rag = FakeRag(error=ConnectionError("graph database glitch"))
    before = {s: counted(s) for s in ("retried", "failed")}
    with pytest.raises(ConnectionError):
        Ingestion(rag).run(str(upload), acl=None, doc_id="d1", last_attempt=False)
    assert [s for s, _ in rag.registry.history] == ["processing"]         # still in flight: the retry will finish it
    assert upload.exists()                                                # the retry needs the file
    assert (counted("retried") - before["retried"], counted("failed") - before["failed"]) == (1.0, 0.0)


def test_the_last_failed_attempt_marks_the_document_failed_and_is_not_also_counted_as_a_retry(upload):
    rag = FakeRag(error=ConnectionError("graph database glitch"))
    before = {s: counted(s) for s in ("retried", "failed")}
    with pytest.raises(ConnectionError):
        Ingestion(rag).run(str(upload), acl=None, doc_id="d1", last_attempt=True)
    status, fields = rag.registry.history[-1]
    assert status == "failed" and "ConnectionError" in fields["error"] and "glitch" not in fields["error"]   # the type, not the details
    assert not upload.exists()                                            # no plaintext left behind by a failure
    assert rag.removed == ["d1"]                                          # and nothing half-built left searchable
    assert (counted("retried") - before["retried"], counted("failed") - before["failed"]) == (0.0, 1.0)


def test_an_earlier_failed_attempt_keeps_the_data_for_the_retry(upload):
    rag = FakeRag(error=ConnectionError("graph database glitch"))
    with pytest.raises(ConnectionError):
        Ingestion(rag).run(str(upload), acl=None, doc_id="d1", last_attempt=False)
    assert rag.removed == [] and upload.exists()                          # the retry overwrites its own records


def test_too_many_graph_failures_fail_the_document_with_the_reason_shown(upload):
    from rag_pipeline import GraphExtractionError
    reason = "Knowledge-graph extraction failed for 3 of 3 passages (more than 25%; first error: ConnectError)."
    rag = FakeRag(error=GraphExtractionError(reason))
    with pytest.raises(GraphExtractionError):
        Ingestion(rag).run(str(upload), acl=None, doc_id="d1", last_attempt=True)
    status, fields = rag.registry.history[-1]
    assert status == "failed" and fields["error"] == reason and rag.removed == ["d1"]


def test_an_unreadable_file_fails_at_once_with_its_reason_and_is_not_retried(upload):
    from chunker import DocumentReadError
    rag = FakeRag(error=DocumentReadError("The DOCX file could not be read (BadZipFile); it may be damaged."))
    out = Ingestion(rag).run(str(upload), acl=None, doc_id="d1", last_attempt=False)    # no exception: nothing to retry
    status, fields = rag.registry.history[-1]
    assert out["status"] == "failed" and status == "failed" and "BadZipFile" in fields["error"]
    assert not upload.exists()


def test_a_finished_document_keeps_its_graph_extraction_counts(upload):
    rag = FakeRag()
    rag.ingest_document = lambda path, acl, doc_id=None: {"doc_id": doc_id, "chunks": 3, "extraction": {},
                                                          "graph": {"chunks_ok": 3, "chunks_failed": 0}}
    Ingestion(rag).run(str(upload), acl=None, doc_id="d1", last_attempt=True)
    status, fields = rag.registry.history[-1]
    assert status == "done" and fields["graph"] == {"chunks_ok": 3, "chunks_failed": 0}


def test_a_file_with_no_readable_text_fails_at_once_with_a_reason(upload):
    pytest.importorskip("pdfplumber")                                     # the reason text comes from chunker.py
    rag = FakeRag(chunks=0)
    result = Ingestion(rag).run(str(upload), acl=None, doc_id="d1", last_attempt=False)
    assert result["status"] == "failed" and result["chunks"] == 0         # returned, not raised: retrying cannot help
    status, fields = rag.registry.history[-1]
    assert status == "failed" and fields["error"]
    assert not upload.exists()


def test_a_document_ingested_without_a_registry_entry_updates_no_status(upload):
    rag = FakeRag()
    assert Ingestion(rag).run(str(upload), acl=None)["status"] == "success"
    assert rag.registry.history == []
