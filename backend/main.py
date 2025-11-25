from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import asyncio
import os

from rag_pipeline import RAGPipeline

app = FastAPI()

# Simple base-paper RAG pipeline (no BM25, no KG, no encryption)
rag = RAGPipeline()

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "rag-model"
    messages: List[ChatMessage]
    stream: bool = True
    session_id: Optional[str] = None

@app.post("/v1/chat/completions")
async def chat_endpoint(req: ChatRequest):

    user_query = req.messages[-1].content

    # 1. Retrieve context
    context = rag.retrieve_context(user_query)

    # 2. Stream LLM answer
    async def event_stream():
        try:
            async for token in rag.stream_answer(user_query, context):
                yield f"data: {json.dumps({'choices':[{'delta':{'content':token}}]})}\n\n"
                await asyncio.sleep(0.01)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/v1/ingest")
async def ingest_endpoint(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    processed = 0

    for f in files:
        raw = await f.read()

        # Save file
        file_path = f"./uploads/{f.filename}"
        with open(file_path, "wb") as out:
            out.write(raw)

        # Ingest into RAG
        rag.ingest_document(file_path)

        processed += 1

    return {"status": "success", "processed": processed}












# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from typing import List, Optional
# import uvicorn
# import json
# import asyncio

# # ==== IMPORT YOUR RAG PIPELINE ====
# from rag_pipeline import RAGPipeline

# app = FastAPI()

# rag = RAGPipeline(
#     chunk_dir="./chunks",
#     encrypted_dir="./encrypted",
#     faiss_path="./faiss.index",
#     bm25_dir="./bm25_index",
#     neo4j_uri="bolt://localhost:7687",
#     neo4j_user="neo4j",
#     neo4j_pass="password",
#     llm="llama3:8b"
# )

# # ========================
# # 1) CHAT COMPLETION API
# # ========================

# class ChatMessage(BaseModel):
#     role: str
#     content: str

# class ChatRequest(BaseModel):
#     model: str = "rag-model"
#     messages: List[ChatMessage]
#     stream: bool = True
#     session_id: Optional[str] = None


# @app.post("/v1/chat/completions")
# async def chat_endpoint(req: ChatRequest):

#     user_query = req.messages[-1].content

#     # 1. Retrieve context (BM25 + FAISS + KG)
#     context = rag.retrieve_context(user_query)

#     # 2. Generate answer using LLM (streaming)
#     async def event_stream():
#         async for chunk in rag.stream_answer(user_query, context):
#             yield f"data: {json.dumps({'choices':[{'delta':{'content':chunk}}]})}\n\n"
#             await asyncio.sleep(0.01)

#         yield "data: [DONE]\n\n"

#     return StreamingResponse(event_stream(), media_type="text/event-stream")


# # ========================
# # 2) DOCUMENT INGESTION
# # ========================

# @app.post("/v1/ingest")
# async def ingest_endpoint(
#     session_id: str = Form(...),
#     files: List[UploadFile] = File(...)
# ):
#     processed = 0

#     for f in files:
#         raw = await f.read()

#         # Step 1: Save file to disk
#         file_path = f"./uploads/{f.filename}"
#         with open(file_path, "wb") as out:
#             out.write(raw)

#         # Step 2: Chunk → Encrypt → Store → Build KG → Index (all automatic)
#         rag.ingest_document(file_path)

#         processed += 1

#     return {"status": "success", "processed": processed}
