"""The embedding model: turns text into the vector that similarity search compares (EMBEDDING_MODEL in settings.py).

Loaded once per process, on the CPU. Storing and searching the vectors is chunk_store.py's job.
"""
from sentence_transformers import SentenceTransformer

from settings import get_settings

EMBEDDING_MODEL_NAME = get_settings().embedding_model

model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
# Taken from the model itself, so switching EMBEDDING_MODEL needs no code change (the collection must be re-indexed).
EMBEDDING_DIM = (getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension)()
