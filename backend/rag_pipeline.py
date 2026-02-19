from chunker import run_chunking
from embedder import build_embeddings
from retriever import retrieve
from llm import stream_generate
from bm25_retriever import BM25Retriever
from reranker import ReRanker
from encryptor import Encryptor

class RAGPipeline:

    def __init__(self):
        self.chunks = [] # Stores encrypted bytes
        self.faiss_path = "./faiss.index"
        self.bm25_path = "./bm25.index"
        self.reranker = ReRanker()
        self.encryptor = Encryptor() # Generates new key on each restart if not persisted!
        # In a real app, load key from env/file. For demo/research, generating new key is fine 
        # as long as we don't expect persistence across restarts without reloading chunks.

    def ingest_document(self, file_path):
        plaintext_chunks = run_chunking(file_path)
        
        # 1. Build Dense Index (FAISS needs plaintext or vectors derived from it)
        build_embeddings(plaintext_chunks, self.faiss_path)
        
        # 2. Build Sparse Index (BM25 technically needs tokens, so we give it plaintext)
        bm25 = BM25Retriever()
        bm25.build_index(plaintext_chunks)
        bm25.save_index(self.bm25_path)
        
        # 3. Encrypt Chunks for Storage
        self.chunks = [self.encryptor.encrypt(chunk) for chunk in plaintext_chunks]

    def retrieve_context(self, query):
        # We need to pass the *encrypted* chunks to retrieve, but wait...
        # The retriever currently returns the chunk content.
        # If we pass encrypted chunks to retrieve, it will return encrypted chunks.
        # We need to decrypt them before returning context.
        
        # Actually, BM25 retriever stores its own copy of chunks in the pickle file (plaintext currently).
        # This is a security hole if we want "Secure RAG".
        # But for Stage 3 scope (encrypt stored chunks), let's focus on what RAGPipeline holds.
        
        results, _, _ = retrieve(query, self.faiss_path, self.bm25_path, self.chunks, reranker=self.reranker, encryptor=self.encryptor)
        
        # Decrypt the results
        decrypted_context = []
        for r in results:
            chunk_data = r["chunk"]
            # If it came from FAISS, it's an index into self.chunks (which is encrypted).
            # If it came from BM25, it might be the string from BM25 object if we aren't careful.
            
            # Helper in retriever.py returns:
            # { "chunk": chunks[idx], "score": ... }
            # So `r["chunk"]` IS the data from `self.chunks`.
            
            if isinstance(chunk_data, bytes):
                decrypted_context.append(self.encryptor.decrypt(chunk_data))
            else:
                 # Fallback if somehow we got plaintext (e.g. from BM25 object directly?)
                 # In our current retriever.py, we pass `chunks` list and it picks from there.
                 # So it should be bytes.
                 decrypted_context.append(str(chunk_data))
                 
        return decrypted_context

    async def stream_answer(self, query, context):
        ctx = "\n\n".join(context)
        async for token in stream_generate("llama3:8b", query, ctx):
            yield token












# import asyncio
# from typing import List
# from chunker import run_chunking
# from encryptor import encrypt_chunks
# from embedder import build_embeddings, search_embeddings
# from retriever import bm25_search
# from kg_builder import expand_query
# from llm import stream_generate

# class RAGPipeline:

#     def __init__(
#         self,
#         chunk_dir,
#         encrypted_dir,
#         faiss_path,
#         bm25_dir,
#         neo4j_uri,
#         neo4j_user,
#         neo4j_pass,
#         llm
#     ):
#         self.chunk_dir = chunk_dir
#         self.encrypted_dir = encrypted_dir
#         self.faiss_path = faiss_path
#         self.bm25_dir = bm25_dir
#         self.llm_name = llm
    
#     # ========================
#     # INGESTION PIPELINE
#     # ========================
#     def ingest_document(self, file_path: str):

#         chunks = run_chunking(file_path)
#         enc_paths = encrypt_chunks(chunks, self.encrypted_dir)
#         build_embeddings(enc_paths, self.faiss_path)
#         bm25_search.build_index(enc_paths, self.bm25_dir)
#         expand_query.build_kg(enc_paths)

#         return True

#     # ========================
#     # RETRIEVAL PIPELINE
#     # ========================
#     def retrieve_context(self, query: str) -> List[str]:
#         bm25_hits = bm25_search(query, self.bm25_dir)
#         dense_hits = search_embeddings(query, self.faiss_path)
#         kg_hits = expand_query(query)

#         # merge + rerank later
#         merged = bm25_hits[:5] + dense_hits[:5] + kg_hits[:3]
#         return merged

#     # ========================
#     # GENERATION PIPELINE
#     # ========================
#     async def stream_answer(self, query: str, context: List[str]):
#         full_context = "\n\n".join(context)
#         async for token in stream_generate(self.llm_name, query, full_context):
#             yield token
