"""A document's life from upload to searchable, in one place.

    queued ──> processing ──> done
       │            └───────> failed   (unreadable, or still failing after the retries)
       └────────────────────> failed   (the job queue was down)

Two halves, run by two processes, sharing the status in the document registry (doc_registry.py):

    Ingestion(rag, dispatch).submit(path, data, acl, filename, uploader)    # the API: save, de-duplicate, register, queue
    Ingestion(rag).run(path, acl, doc_id, last_attempt)                     # the worker: chunk, embed, encrypt, index, graph

Rules both halves keep:
  * The status in the registry is always the truth the UI shows. Every way out of `run` leaves it at done or failed, except an
    attempt that will be retried (it stays 'processing').
  * The uploaded file is plaintext on disk, so it is removed as soon as it is no longer needed: when ingestion succeeds, when it
    can never succeed, and when the job could not be queued. After that the only stored copy of the content is the encrypted
    passages. (KEEP_UPLOADS turns the removal off for debugging.)
  * Ingestion is idempotent: the same bytes under the same access rule are the same document (doc_identity.compute_doc_id), so a
    re-upload is reported as a duplicate and a retry overwrites its own records.

Who may upload what is decided before this module is reached (api.py); validating the bytes is uploads.py.
"""
import logging
import os
import time

import metrics
from doc_identity import compute_doc_id
from settings import get_settings

log = logging.getLogger("mindsearch.ingestion")

IN_FLIGHT_OR_DONE = ("queued", "processing", "done")


class QueueUnavailable(RuntimeError):
    """The document was registered but the job could not be queued; it is marked failed and its file is gone."""

    def __init__(self, doc_id, cause):
        super().__init__(f"could not queue the ingestion job ({cause})")
        self.doc_id, self.cause = doc_id, cause


class Ingestion:
    def __init__(self, rag, dispatch=None):
        """`dispatch(path, acl_payload, doc_id) -> task id` hands the job to the worker (Celery in production). Only `submit` needs
        it; without one, documents are registered but nothing processes them (some tests)."""
        self.rag, self.dispatch = rag, dispatch

    # ------------------------------------------------------------------ the API's half

    def submit(self, path, data: bytes, acl, filename, uploader) -> dict:
        """Save already-validated bytes at `path` and queue them for processing. Returns the receipt shown to the uploader:
        {"filename", "doc_id", "status", "duplicate", "task_id"}. Raises QueueUnavailable if the job cannot be queued."""
        with open(path, "wb") as out:
            out.write(data)
        doc_id = compute_doc_id(path, acl)
        existing = self.rag.registry.get(doc_id)
        if existing and existing.status in IN_FLIGHT_OR_DONE:
            os.remove(path)                    # same content + same access rule: nothing new to ingest
            return {"filename": filename, "doc_id": doc_id, "status": existing.status, "duplicate": True, "task_id": None}
        self.rag.registry.register(doc_id, acl, filename, uploader, len(data))
        try:
            task_id = self.dispatch(path, acl.to_payload(), doc_id) if self.dispatch else None
        except Exception as e:  # noqa: BLE001 - e.g. the job queue is down
            self.rag.registry.set_status(doc_id, "failed", error="could not queue the job")
            os.remove(path)
            raise QueueUnavailable(doc_id, type(e).__name__) from e
        return {"filename": filename, "doc_id": doc_id, "status": "queued", "duplicate": False, "task_id": task_id}

    # ------------------------------------------------------------------ the worker's half

    def run(self, path, acl, doc_id=None, last_attempt=True) -> dict:
        """Process one queued document. Returns {"status": "success" | "failed", "file", "doc_id", "chunks"}.

        If processing raises (Neo4j or the model server glitched), the exception is re-raised so the caller can retry: that is safe
        because ingestion is idempotent. On the `last_attempt` the document is first marked failed and its file removed."""
        from chunker import DocumentReadError         # imported here: the API process uses `submit` only and needs no file parsers
        from rag_pipeline import GraphExtractionError
        started = time.perf_counter()
        name = os.path.basename(path)
        self._status(doc_id, "processing")
        try:
            result = self.rag.ingest_document(path, acl, doc_id=doc_id)
        except DocumentReadError as e:
            # The file itself cannot be read (damaged, mislabelled, undecodable): the same bytes fail the same way, so no retry.
            log.error("Document %s could not be read: %s", name, e)
            metrics.INGEST_DOCUMENTS.labels("failed").inc()
            _remove_upload(path)
            self._status(doc_id, "failed", error=str(e))
            return {"status": "failed", "file": name, "doc_id": doc_id, "chunks": 0}
        except Exception as e:
            log.error("Error processing document %s: %s", name, e)
            if last_attempt:
                metrics.INGEST_DOCUMENTS.labels("failed").inc()
                _remove_upload(path)
                if doc_id:
                    self.rag.remove_partial(doc_id)    # a failed document is never searchable half-built
                reason = str(e) if isinstance(e, GraphExtractionError) else f"{type(e).__name__}: ingestion failed after retries"
                self._status(doc_id, "failed", error=reason)
            else:
                metrics.INGEST_DOCUMENTS.labels("retried").inc()
            raise
        scan = result.get("extraction") or {}
        _remove_upload(path)
        if result["chunks"] == 0:
            from chunker import no_text_message      # imported here: the API process uses `submit` only and needs no file parsers
            # Nothing to index: an empty file, or a scan the OCR engine could not read. Not retried (the file will not get better)
            # and not shown as 'done': a document that can never be answered from must say so.
            metrics.INGEST_DOCUMENTS.labels("failed").inc()
            self._status(doc_id, "failed", error=no_text_message(scan), scan=scan)
            return {"status": "failed", "file": name, "doc_id": result["doc_id"], "chunks": 0}
        self._status(doc_id, "done", chunks=result["chunks"], scan=scan, graph=result.get("graph"))
        metrics.INGEST_DOCUMENTS.labels("done").inc()
        metrics.INGEST_CHUNKS.inc(result["chunks"])
        metrics.INGEST_SECONDS.observe(time.perf_counter() - started)
        return {"status": "success", "file": name, "doc_id": result["doc_id"], "chunks": result["chunks"]}

    def _status(self, doc_id, status, **fields):
        if doc_id:                              # a document ingested without the API (a script) has no registry entry to update
            self.rag.registry.set_status(doc_id, status, **fields)


def _remove_upload(path):
    if get_settings().keep_uploads:
        return
    try:
        os.remove(path)
    except OSError:
        pass
