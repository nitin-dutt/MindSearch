from embedder import search_embeddings
import os
import time

from bm25_retriever import BM25Retriever

def retrieve(query, index_path, bm25_path, chunks, k=5, reranker=None, encryptor=None):
    """
    Hybrid retrieval using RRF (Reciprocal Rank Fusion).
    """
    # 1. FAISS Search (Dense)
    if not os.path.exists(index_path) or not chunks:
         return [{"chunk": "Hello there no currently no documents that are uploaded, please upload some documents to proceed", "score": 1.0}], [], []
    
    start_time = time.time()
    dense_scores, dense_idxs = search_embeddings(query, index_path, k=k*2) # Get more candidates for fusion
    
    # 2. BM25 Search (Sparse)
    bm25 = BM25Retriever()
    if os.path.exists(bm25_path):
        bm25.load_index(bm25_path)
        sparse_results = bm25.search(query, k=k*2)
    else:
        sparse_results = []
        
    # 3. RRF Fusion
    # Score = 1 / (k + rank)
    k_const = 60
    doc_scores = {}
    
    # Process Dense Results
    for rank, idx in enumerate(dense_idxs):
        if idx == -1: continue # Invalid index from FAISS
        if idx not in doc_scores:
            doc_scores[idx] = 0.0
        doc_scores[idx] += 1.0 / (k_const + rank + 1)
        
    # Process Sparse Results
    for rank, item in enumerate(sparse_results):
        idx = item['index']
        if idx not in doc_scores:
            doc_scores[idx] = 0.0
        doc_scores[idx] += 1.0 / (k_const + rank + 1)
        
    # Sort by RRF score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N candidates for re-ranking (e.g., top 20 or all if less)
    # If no re-ranker, just take top k
    top_n_candidates = sorted_docs[:k*4] if reranker else sorted_docs[:k]
    
    candidate_chunks = []
    for idx, score in top_n_candidates:
        if idx < len(chunks):
             chunk_data = chunks[idx]
             # Decrypt if encryptor is provided and data is bytes
             if encryptor and isinstance(chunk_data, bytes):
                 try:
                     candidate_chunks.append(encryptor.decrypt(chunk_data))
                 except Exception:
                     # On fail, maybe just pass raw or empty?
                     candidate_chunks.append("")
             elif isinstance(chunk_data, bytes):
                 # If no encryptor but bytes, we can't re-rank efficiently (model needs text)
                 # We skip re-ranking or pass empty string?
                 # ideally we shouldn't be here.
                 candidate_chunks.append("") 
             else:
                 candidate_chunks.append(chunk_data)
             
    if reranker:
        # Re-rank the candidates
        # ReRanker needs plaintext
        final_results = reranker.rerank(query, candidate_chunks, top_k=k)
        
        # We need to map back to original encrypted chunks for return?
        # The reranker returns {'chunk': text, ...}. 
        # But our pipeline expects `retrieve` to return the objects from `chunks` list (which are encrypted).
        # We need to preserve the link to original data.
        
        # Let's adjust ReRanker to return indices or we map back.
        # Reranker returns 'original_index' relative to the list passed to it.
        # We need to map that back to global index.
        
        mapped_results = []
        for res in final_results:
             local_idx = res['original_index']
             global_idx_pair = top_n_candidates[local_idx] # (idx, score)
             global_idx = global_idx_pair[0]
             
             mapped_results.append({
                 "chunk": chunks[global_idx], # Return the encrypted chunk
                 "score": res['score']
             })
        final_results = mapped_results
        
    else:
        # Format as before for consistency
        final_results = []
        for i, (idx, score) in enumerate(top_n_candidates):
            if idx < len(chunks):
                final_results.append({
                    "chunk": chunks[idx],
                    "score": score
                })
            
    return final_results, dense_idxs, sparse_results # Return debug info if needed

