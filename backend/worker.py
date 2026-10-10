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
def process_document_task(self, file_path: str, acl_payload: dict):
    """
    Background task to process, embed, encrypt, and extract Knowledge Graph from a document.
    `acl_payload` is DocACL.to_payload(): who may read the document (min_rank, groups, users).
    The uploaded plaintext file is deleted once ingestion succeeds (or retries are exhausted), so the
    only stored copy of the content is the encrypted chunks.
    """
    acl = DocACL(min_rank=int(acl_payload["min_rank"]), groups=tuple(acl_payload.get("groups", ())),
                 users=tuple(acl_payload.get("users", ())))
    try:
        rag.ingest_document(file_path, acl)
    except Exception as e:
        log.error("Error processing document %s: %s", os.path.basename(file_path), e)
        if self.request.retries >= self.max_retries:
            _remove_upload(file_path)
        # Retry the task if it fails (e.g., Neo4j or Ollama connection glitch)
        raise self.retry(exc=e, countdown=10)
    _remove_upload(file_path)
    return {"status": "success", "file": os.path.basename(file_path)}


def _remove_upload(path):
    if os.getenv("KEEP_UPLOADS") == "1":
        return
    try:
        os.remove(path)
    except OSError:
        pass
