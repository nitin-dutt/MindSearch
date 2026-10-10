from celery import Celery
from access_policy import DocACL
from rag_pipeline import RAGPipeline
import logging
import os

# Initialize Celery application
# Connects to a local Redis instance by default
celery_app = Celery(
    "mindsearch_worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

# Initialize the RAG Pipeline once per worker process
# This is safe because RAGPipeline no longer stores state in RAM
rag = RAGPipeline()
log = logging.getLogger("mindsearch.worker")

@celery_app.task(name="process_document", bind=True, max_retries=3)
def process_document_task(self, file_path: str, acl_payload: dict, doc_id: str = None):
    """
    Background task to process, embed, encrypt, and extract Knowledge Graph from a document.
    `acl_payload` is DocACL.to_payload(): who may read the document (min_rank, groups, users).
    Progress is written to the document registry (queued -> processing -> done/failed) so the UI can show it.
    The uploaded plaintext file is deleted once ingestion succeeds (or retries are exhausted), so the
    only stored copy of the content is the encrypted chunks.
    """
    acl = DocACL(min_rank=int(acl_payload["min_rank"]), groups=tuple(acl_payload.get("groups", ())),
                 users=tuple(acl_payload.get("users", ())))
    if doc_id:
        rag.registry.set_status(doc_id, "processing")
    try:
        result = rag.ingest_document(file_path, acl, doc_id=doc_id)
    except Exception as e:
        log.error("Error processing document %s: %s", os.path.basename(file_path), e)
        if self.request.retries >= self.max_retries:
            _remove_upload(file_path)
            if doc_id:
                rag.registry.set_status(doc_id, "failed", error=f"{type(e).__name__}: ingestion failed after retries")
        # Retry the task if it fails (e.g., Neo4j or Ollama connection glitch). Safe because ingestion is idempotent.
        raise self.retry(exc=e, countdown=10)
    if doc_id:
        rag.registry.set_status(doc_id, "done", chunks=result["chunks"])
    _remove_upload(file_path)
    return {"status": "success", "file": os.path.basename(file_path), "doc_id": result["doc_id"], "chunks": result["chunks"]}


def _remove_upload(path):
    if os.getenv("KEEP_UPLOADS") == "1":
        return
    try:
        os.remove(path)
    except OSError:
        pass
