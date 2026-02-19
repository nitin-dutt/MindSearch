from rag_pipeline import RAGPipeline

print("Testing retrieval with no documents (expecting graceful fallback)...")
try:
    rag = RAGPipeline()
    # verify chunks is empty
    assert not rag.chunks
    
    context = rag.retrieve_context("Hello")
    print("Context returned:", context)
    
    if len(context) == 1 and "No documents" in context[0]:
        print("✅ SUCCESS: Handled empty state correctly.")
    else:
        print("❌ FAILURE: Unexpected context.")
        
except Exception as e:
    print(f"❌ CRASHED: {e}")
    import traceback
    traceback.print_exc()
