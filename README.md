# MindSearch

A modern **Retrieval-Augmented Generation (RAG)** system that enables intelligent document chat using semantic search, knowledge graphs, and LLM integration.

## Features

- 📄 **Multi-format Document Support**: Ingest PDF, DOCX, and TXT files
- 🔍 **Hybrid Retrieval**: Combines Dense (FAISS) and Sparse (BM25) search using Reciprocal Rank Fusion (RRF)
- 🕸️ **Knowledge Graph Integration**: Extracts entities and relationships via LLM (Ollama) & SpaCy, storing them in a Neo4j graph database to retrieve 1-2 hop contextual subgraphs
- 🎯 **Advanced Re-Ranking**: Uses Cross-Encoders (MS-MARCO) to re-rank retrieved context for higher accuracy
- 🔒 **Secure Data Storage**: AES-256-GCM encryption for all document chunks stored on disk
- 💬 **AI-Powered Chat**: Stream responses from Ollama LLM models (`llama3:8b`)
- 🚀 **Production-Ready**: FastAPI backend with Streamlit frontend
- ⚡ **Async Streaming**: Server-Sent Events (SSE) for real-time responses

## Architecture

```
Frontend (Streamlit)
      ↓
FastAPI Backend
      ├── Document Ingestion → Chunking
      │     ├── Knowledge Graph Extraction (Ollama LLM → Neo4j)
      │     └── AES-256 Encryption & Embedding
      ├── Hybrid Retrieval (FAISS + BM25) & Cross-Encoder Re-Ranking
      ├── Graph Context Retrieval (Neo4j 1-2 hop relationships)
      └── LLM Integration (Ollama)
```

## Prerequisites

- Python 3.9+
- Ollama with `llama3:8b` model installed ([ollama.ai](https://ollama.ai))
- Neo4j Database running locally (default URI: `neo4j://127.0.0.1:7687`, user: `neo4j`, password: `24112003`)
- pip

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/nitin-dutt/MindSearch.git
cd MindSearch
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r backend/requirements.txt
```
*(Note: SpaCy model `en_core_web_sm` is automatically downloaded on first run via `kg_builder.py`)*

## Usage

### 1. Start External Services
```bash
# Start Ollama Model Service
ollama serve
# Ensure llama3:8b is pulled
ollama pull llama3:8b

# Ensure your local Neo4j desktop or Docker instance is running
```

### 2. Start Backend (FastAPI)
```bash
cd backend
python -m uvicorn main:app --reload
# Server runs on http://localhost:8000
```

### 3. Start Frontend (Streamlit)
```bash
cd frontend
streamlit run app.py
# Opens on http://localhost:8501
```

## API Endpoints

### Chat Completion (Streaming)
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "rag-model",
  "messages": [{"role": "user", "content": "What is in the documents?"}],
  "stream": true,
  "session_id": "session-123"
}
```

### Document Ingestion
```bash
POST /v1/ingest
Content-Type: multipart/form-data

session_id: session-123
files: [file1.pdf, file2.txt, ...]
```

## Project Structure

```
MindSearch/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── rag_pipeline.py      # Core RAG pipeline integrating all components
│   ├── chunker.py           # Document processing
│   ├── embedder.py          # Vector embeddings (FAISS)
│   ├── bm25_retriever.py    # Sparse retrieval index (BM25)
│   ├── retriever.py         # Semantic search & Hybrid RRF fusion
│   ├── reranker.py          # Cross-encoder re-ranking
│   ├── encryptor.py         # AES-256 chunk encryption
│   ├── kg_builder.py        # Entity & Relationship extraction (Neo4j, SpaCy, Ollama)
│   ├── llm.py               # LLM streaming
│   ├── compare_retrievers.py# Evaluation script
│   ├── requirements.txt
│   └── uploads/             # Uploaded documents
│
├── frontend/
│   ├── app.py               # Streamlit UI
│   └── requirements.txt
```

## Key Components

### Knowledge Graph Pipeline (`kg_builder.py`)
- Leverages LLM (`llama3:8b`) to extract Subject-Verb-Object relationships directly from chunks
- Built with Neo4j to store `Entity` nodes and directional relationships
- Uses SpaCy for query entity extraction to pull context-rich 1-hop and 2-hop graph subgraphs during retrieval

### Document Chunking (`chunker.py`)
- Supports PDF, DOCX, TXT files
- Sentence-based chunking with configurable size
- Automatic encoding detection

### Data Security (`encryptor.py`)
- AES-GCM 256-bit encryption for all stored chunks
- Decryption on-the-fly during retrieval

### Hybrid Retrieval & Re-Ranking (`retriever.py`, `reranker.py`)
- Dense search (FAISS + all-MiniLM-L6-v2)
- Sparse search (BM25)
- Reciprocal Rank Fusion (RRF) combines both results
- Cross-Encoder re-ranking (`ms-marco-MiniLM-L-6-v2`) for top candidates

### LLM Integration (`llm.py`)
- Streaming responses via Ollama
- Async/thread-safe implementation
- Error handling and fallbacks

## Configuration

Edit backend files to customize:
- **Neo4j Credentials**: `rag_pipeline.py` & `kg_builder.py`
- **Chunk size**: `chunker.py` → `chunk_size` parameter
- **Model name**: `llm.py` → `stream_generate()` model parameter
- **Search results**: `retriever.py` → `k` parameter
- **API base URL**: `frontend/app.py` → `API_BASE_URL`

## Dependencies

**Backend:**
- FastAPI, Uvicorn
- sentence-transformers, FAISS (Dense Search)
- rank_bm25 (Sparse Search)
- cryptography (AES-GCM Encryption)
- pdfplumber, python-docx
- Ollama client
- NLTK, Spacy
- neo4j

**Frontend:**
- Streamlit
- requests

## Troubleshooting

### "Neo4j connection error"
- Ensure Neo4j desktop or Docker container is running locally.
- Validate the URI, User, and Password match the defaults inside `kg_builder.py`.

### "Unable to connect to RAG API"
- Check if backend is running on `http://localhost:8000`
- Verify firewall settings

### "No documents ingested yet"
- Upload documents first using the frontend
- Check `uploads/` folder for saved files

### "Ollama connection error"
- Ensure Ollama service is running (`ollama serve`)
- Check if model is installed (`ollama list`)

### NLTK tokenizer errors
- Automatically downloaded on first use
- Manual download: `python -m nltk.downloader punkt_tab`

## Performance Tips

- Larger chunk sizes → fewer but longer context windows
- Smaller embedding models → faster but less accurate search
- Adjust `k` (search results) based on context length needs

## License

MIT

## Author

Nitin Dutt - [GitHub](https://github.com/nitin-dutt)
