"""
Utility script to reset the Qdrant collection so fresh chunks get re-indexed.
Run with: docker compose exec celery_worker python reset_qdrant.py
"""
from qdrant_client import QdrantClient

from settings import get_settings

_cfg = get_settings()
QDRANT_URL = _cfg.qdrant_url
QDRANT_PATH = _cfg.qdrant_local_path
COLLECTION_NAME = _cfg.qdrant_collection

client = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(path=QDRANT_PATH)

existing = client.get_collections().collections
names = [c.name for c in existing]

if COLLECTION_NAME in names:
    client.delete_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' deleted successfully.")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist — nothing to delete.")

print("Qdrant is clean and ready for re-ingestion.")
