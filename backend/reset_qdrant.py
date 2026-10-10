"""
Utility script to reset the Qdrant collection so fresh chunks get re-indexed.
Run with: docker compose exec celery_worker python reset_qdrant.py
"""
import os

from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "mindsearch_docs"

client = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(path=QDRANT_PATH)

existing = client.get_collections().collections
names = [c.name for c in existing]

if COLLECTION_NAME in names:
    client.delete_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' deleted successfully.")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist — nothing to delete.")

print("Qdrant is clean and ready for re-ingestion.")
