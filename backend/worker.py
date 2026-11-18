"""The background worker: Celery glue around ingestion.Ingestion.run (which owns what happens to a document)."""
import logging

from celery import Celery
from celery.signals import worker_ready

from access_policy import DocACL
from ingestion import Ingestion
from rag_pipeline import RAGPipeline
from settings import get_settings

_cfg = get_settings()

celery_app = Celery("mindsearch_worker", broker=_cfg.celery_broker_url, backend=_cfg.celery_result_backend)

# One pipeline per worker process (the models are loaded once). Safe to share: it keeps no per-document state in memory.
rag = RAGPipeline()
ingestion = Ingestion(rag)
log = logging.getLogger("mindsearch.worker")


@celery_app.task(name="process_document", bind=True, max_retries=3)
def process_document_task(self, file_path: str, acl_payload: dict, doc_id: str = None):
    """Process, embed, encrypt and graph-extract one uploaded document. `acl_payload` is DocACL.to_payload(): who may read the
    document (min_rank, groups, users). A failure is retried up to `max_retries` times, 10 seconds apart (e.g. a Neo4j or model
    server glitch); ingestion is idempotent, so a retry is safe."""
    acl = DocACL(min_rank=int(acl_payload["min_rank"]), groups=tuple(acl_payload.get("groups", ())),
                 users=tuple(acl_payload.get("users", ())))
    try:
        return ingestion.run(file_path, acl, doc_id, last_attempt=self.request.retries >= self.max_retries)
    except Exception as e:
        raise self.retry(exc=e, countdown=10)     # on the last attempt this re-raises `e` and the task ends as failed


@worker_ready.connect
def _start_metrics_server(**_):
    """Expose this worker's metrics for Prometheus once the worker is really running (not when the module is merely imported)."""
    port = get_settings().worker_metrics_port
    if port:
        from prometheus_client import start_http_server
        start_http_server(port)
        log.info("worker metrics on port %d", port)
