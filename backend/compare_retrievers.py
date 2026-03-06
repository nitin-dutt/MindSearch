import os
import sys
from rag_pipeline import RAGPipeline
from bm25_retriever import BM25Retriever
from embedder import search_embeddings
from retriever import retrieve
import time
from kg_builder import KnowledgeGraphBuilder

def compare_retrievers():
    # Setup paths
    faiss_path = "./faiss.index"
    bm25_path = "./bm25.index"
    
    # Initialize Pipeline (just to get chunks if already loaded, or we reload them)
    # Note: RAGPipeline loads chunks in memory after ingest. 
    # For this script we need to manually load chunks from the BM25 pickle since RAGPipeline doesn't persist chunks independently on disk in this simple version
    
    if not os.path.exists(bm25_path):
        print("BM25 index not found. Ingesting sample document (README.md)...")
        rag = RAGPipeline()
        # Ingest README.md from parent directory
        readme_path = os.path.abspath("../README.md")
        if os.path.exists(readme_path):
            rag.ingest_document(readme_path)
            print("Ingestion complete.")
        else:
            print(f"README not found at {readme_path}. Please ingest a document manually.")
            return

    # Load resources
    print("Loading resources...")
    bm25_ret = BM25Retriever()
    bm25_ret.load_index(bm25_path)
    chunks = bm25_ret.chunks
    print(f"Loaded {len(chunks)} chunks.")
    
    print("Initializing Knowledge Graph Builder...")
    kg_builder = KnowledgeGraphBuilder(uri="neo4j://127.0.0.1:7687", user="neo4j", password="24112003")
    
    # Define Test Queries
    queries = [
        "What is RAG?",
        "How is FAISS used?",
        "Does it support PDF?",
        "explain the architecture",
        "requirements for installation"
    ]
    
    output_file = "retrieval_comparison.md"
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Retrieval Comparison: BM25 vs FAISS vs Hybrid\n\n")
            
            for i, q in enumerate(queries):
                print(f"Processing query {i+1}: {q}")
                f.write(f"## Query: {q}\n\n")
                
                # 1. BM25
                f.write("### BM25 Only\n")
                bm25_results = bm25_ret.search(q, k=3)
                print(f"  BM25 found {len(bm25_results)} results")
                for res in bm25_results:
                    f.write(f"- [Score: {res['score']:.4f}] {res['chunk'][:300]}...\n")
                f.write("\n")
                
            # 2. FAISS (Dense)
            f.write("### FAISS Only\n")
            scores, idxs = search_embeddings(q, faiss_path, k=3)
            for score, idx in zip(scores, idxs):
                if idx < len(chunks):
                    f.write(f"- [Score: {score:.4f}] {chunks[idx][:300]}...\n")
            f.write("\n")
            
            # 3. Hybrid (RRF)
            f.write("### Hybrid (RRF)\n")
            # retrieve returns (chunks, dense_debug, sparse_debug)
            # We need to compute RRF scores here to display them, or modify retrieve to return them.
            # For now, let's just use the order which implies rank.
            # Actually, let's modify retrieve to return the detailed list with scores for debugging.
            
            # Since I can't easily change the retrieve signature without breaking other things potentially (though I just wrote it),
            # I will trust the rank order but print more text.
            
            hybrid_results, _, _ = retrieve(q, faiss_path, bm25_path, chunks, k=3)
            for i, res in enumerate(hybrid_results):
                f.write(f"- [Rank {i+1} | Score: {res['score']:.4f}] {res['chunk'][:300]}...\n")
            f.write("\n")
            
            # 4. Hybrid + Re-ranking
            f.write("### Hybrid + Re-ranking\n")
            # We need to initialize ReRanker once outside loop usually, but for script simplicity we can do here or top level
            # Check if we have reranker initialized
            if 'reranker' not in locals():
                print("Initializing ReRanker...")
                from reranker import ReRanker
                reranker = ReRanker()
                
            reranked_results, _, _ = retrieve(q, faiss_path, bm25_path, chunks, k=3, reranker=reranker)
            for i, res in enumerate(reranked_results):
                f.write(f"- [Rank {i+1} | Score: {res['score']:.4f}] {res['chunk'][:300]}...\n")
            f.write("\n")
            
            # 5. Graph-Enhanced Hybrid (GraphRAG)
            f.write("### Graph-Enhanced Hybrid RAG\n")
            f.write("**Graph Context Extracted:**\n")
            graph_context = kg_builder.get_graph_context(q)
            if graph_context:
                for gc in graph_context:
                    f.write(f"- {gc}\n")
            else:
                f.write("- No graph context found for the query entities.\n")
                
            f.write("\n**Combined Context (Graph + Hybrid RRF):**\n")
            # We prepend the graph context to the top hybrid results
            combined = graph_context + [res['chunk'] for res in reranked_results]
            for i, chunk in enumerate(combined[:5]): # Show top 5 combined
                 # Truncate for display
                 display_chunk = chunk if isinstance(chunk, str) else str(chunk)
                 f.write(f"- [Result {i+1}] {display_chunk[:300]}...\n")
                 
            f.write("\n")
            
            f.write("---\n\n")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    print(f"Comparison saved to {output_file}")

if __name__ == "__main__":
    compare_retrievers()
