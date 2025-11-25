from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_embeddings(chunks, index_path):
    embeddings = model.encode(chunks, convert_to_numpy=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    return True

def search_embeddings(query, index_path, k=5):
    index = faiss.read_index(index_path)
    q = model.encode([query], convert_to_numpy=True)
    scores, idxs = index.search(q, k)
    return idxs[0]
