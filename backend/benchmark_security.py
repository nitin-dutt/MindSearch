import time
from encryptor import Encryptor
from rag_pipeline import RAGPipeline
import os

def benchmark_security():
    print("=== Secure RAG Benchmark ===")
    
    # 1. Micro-benchmark: Encryption Overhead
    encryptor = Encryptor()
    sample_text = "This is a sample sentence to test AES-256 encryption latency." * 10 # ~600 chars
    
    iterations = 1000
    start = time.time()
    for _ in range(iterations):
        enc = encryptor.encrypt(sample_text)
    enc_time = (time.time() - start) / iterations * 1000 # ms
    
    start = time.time()
    for _ in range(iterations):
        dec = encryptor.decrypt(enc)
    dec_time = (time.time() - start) / iterations * 1000 # ms
    
    print(f"Encryption Latency: {enc_time:.4f} ms/chunk")
    print(f"Decryption Latency: {dec_time:.4f} ms/chunk")
    print(f"Throughput: {1000/enc_time:.2f} chunks/sec (Encrypt)")

    # 2. Pipeline Integration Test
    print("\nTesting Pipeline with Encryption...")
    rag = RAGPipeline()
    
    # Ingest logic (mocking file read to isolate pipeline logic)
    # We will just verify that self.chunks holds bytes
    readme_path = os.path.abspath("../README.md")
    if os.path.exists(readme_path):
        rag.ingest_document(readme_path)
        
        # Check storage type
        if rag.chunks and isinstance(rag.chunks[0], bytes):
            print("✅ Verified: Chunks are stored as BYTES (Encrypted)")
        else:
            print("❌ Failed: Chunks are NOT bytes")
            
        # Check Retrieval (Decryption)
        query = "What is MindSearch?"
        start = time.time()
        context = rag.retrieve_context(query)
        retrieval_time = (time.time() - start) * 1000 # ms
        
        if context and isinstance(context[0], str):
             print(f"✅ Verified: Retrieved context is STRING (Decrypted)")
             print(f"Retrieval Latency (Search + ReRank + Decrypt): {retrieval_time:.2f} ms")
        else:
             print("❌ Failed: Retrieved context is not string")

    else:
        print("Skipping Pipeline test (README.md not found)")

if __name__ == "__main__":
    benchmark_security()
