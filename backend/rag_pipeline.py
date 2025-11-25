from chunker import run_chunking
from embedder import build_embeddings
from retriever import retrieve
from llm import stream_generate

class RAGPipeline:

    def __init__(self):
        self.chunks = []

    def ingest_document(self, file_path):
        chunks = run_chunking(file_path)
        build_embeddings(chunks, "./faiss.index")
        self.chunks = chunks

    def retrieve_context(self, query):
        return retrieve(query, "./faiss.index", self.chunks)

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
