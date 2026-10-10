FROM python:3.12-slim

# Install system dependencies for PyMuPDF/Tesseract if needed later
# (Currently just basic python deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better Docker layer caching
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Download spacy model (used by kg_builder.py)
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command (can be overridden by docker-compose for Celery)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
