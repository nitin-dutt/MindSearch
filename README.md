# MindSearch

A modern **Retrieval-Augmented Generation (RAG)** system that enables intelligent document chat using semantic search and LLM integration.

## Features

- 📄 **Multi-format Document Support**: Ingest PDF, DOCX, and TXT files
- 🔍 **Semantic Search**: Uses sentence transformers and FAISS for efficient vector search
- 💬 **AI-Powered Chat**: Stream responses from Ollama LLM models
- 🚀 **Production-Ready**: FastAPI backend with Streamlit frontend
- ⚡ **Async Streaming**: Server-Sent Events (SSE) for real-time responses

## Architecture

```
Frontend (Streamlit)
      ↓
FastAPI Backend
      ├── Document Ingestion → Chunking → Embedding
      ├── Vector Search (FAISS)
      └── LLM Integration (Ollama)
```

## Prerequisites

- Python 3.9+
- Ollama with `llama3:8b` model installed ([ollama.ai](https://ollama.ai))
- pip

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/MindSearch.git
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

## Usage

### 1. Start Ollama Service
```bash
ollama serve
# In another terminal, pull the model if needed:
ollama pull llama3:8b
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
│   ├── rag_pipeline.py      # Core RAG logic
│   ├── chunker.py           # Document processing
│   ├── embedder.py          # Vector embeddings (FAISS)
│   ├── retriever.py         # Semantic search
│   ├── llm.py               # LLM streaming
│   ├── requirements.txt
│   └── uploads/             # Uploaded documents
│
├── frontend/
│   ├── app.py               # Streamlit UI
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

## Key Components

### Document Chunking (`chunker.py`)
- Supports PDF, DOCX, TXT files
- Sentence-based chunking with configurable size
- Automatic encoding detection

### Embeddings (`embedder.py`)
- Uses `sentence-transformers` (all-MiniLM-L6-v2)
- FAISS for efficient similarity search
- Indexes stored for fast retrieval

### LLM Integration (`llm.py`)
- Streaming responses via Ollama
- Async/thread-safe implementation
- Error handling and fallbacks

## Configuration

Edit backend files to customize:
- **Chunk size**: `chunker.py` → `chunk_size` parameter
- **Model name**: `llm.py` → `stream_generate()` model parameter
- **Search results**: `retriever.py` → `k` parameter
- **API base URL**: `frontend/app.py` → `API_BASE_URL`

## Dependencies

**Backend:**
- FastAPI, Uvicorn
- sentence-transformers, FAISS
- pdfplumber, python-docx
- Ollama client
- NLTK

**Frontend:**
- Streamlit
- requests

## Troubleshooting

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

## Future Enhancements

- [ ] Multi-model support (GPT-4, Claude)
- [ ] Web search integration
- [ ] Query expansion with knowledge graphs
- [ ] Document metadata filtering
- [ ] User authentication & persistence
- [ ] Docker deployment
- [ ] Evaluation metrics dashboard

## License

MIT

## Author

Nitin Dutt - [GitHub](https://github.com/nitin-dutt)

