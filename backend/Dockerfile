FROM python:3.12-slim

# System packages needed to compile a few Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, for Docker layer caching. Every version is pinned: requirements*.txt name what we use,
# constraints.txt fixes the exact version of every package (including indirect ones) so a build is reproducible.
# WITH_EVAL=1 (set for the Celery worker in docker-compose.yml) also installs the evaluation libraries.
ARG WITH_EVAL=0
COPY requirements.txt requirements-eval.txt constraints.txt ./
# CPU-only torch first (the embedder and cross-encoder run on CPU; the language model runs in the vLLM container). The default
# PyPI build drags in about 3 GB of CUDA libraries this image never uses. Same version as the lock, so the next step keeps it.
RUN pip install --default-timeout=1000 --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
RUN if [ "$WITH_EVAL" = "1" ]; then REQ=requirements-eval.txt; else REQ=requirements.txt; fi \
    && pip install --default-timeout=1000 --no-cache-dir -r "$REQ" -c constraints.txt

# spaCy model used by kg_builder.py, pinned to the version that matches the pinned spaCy
RUN pip install --no-cache-dir \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# Copy source code
COPY . .

EXPOSE 8000

# Default command (docker-compose overrides it for Celery)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
