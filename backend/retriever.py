from embedder import search_embeddings
import os

def retrieve(query, index_path, chunks):
    # Check if index exists
    if not os.path.exists(index_path):
        return ["No documents ingested yet. Please upload documents first."]
    
    if not chunks:
        return ["No documents ingested yet. Please upload documents first."]
    
    idxs = search_embeddings(query, index_path, k=5)
    return [chunks[i] for i in idxs]
