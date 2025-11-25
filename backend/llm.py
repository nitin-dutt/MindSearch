import asyncio
import ollama
import queue
import threading

async def stream_generate(model, query, context):
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    
    try:
        # Use a thread-safe queue for communication between threads
        token_queue = queue.Queue()
        
        def generate_sync():
            try:
                response = ollama.generate(model=model, prompt=prompt, stream=True)
                for chunk in response:
                    token = chunk.get("response", "")
                    if token:
                        token_queue.put(("token", token))
                token_queue.put(("done", None))
            except Exception as e:
                token_queue.put(("error", str(e)))
                token_queue.put(("done", None))
        
        # Start the sync generation in a thread
        thread = threading.Thread(target=generate_sync, daemon=True)
        thread.start()
        
        # Yield tokens as they arrive
        while True:
            msg_type, value = token_queue.get()
            if msg_type == "done":
                break
            elif msg_type == "error":
                yield f"Error: {value}"
                break
            else:
                yield value
        
        thread.join(timeout=5)
        
    except Exception as e:
        yield f"Error generating response: {str(e)}"
