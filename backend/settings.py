"""Central configuration (alignment_plan INF-09).

Every model name, service URL, path and tuning knob the application uses is defined HERE, once, with its default and the
environment variable that overrides it. The API, the Celery worker and the evaluation harness all read the same object,
so changing a model or an address is a one-line change (or one environment variable) instead of an edit in several files.

    from settings import get_settings
    cfg = get_settings()
    cfg.embedding_model, cfg.vllm_api_base, ...

`get_settings()` reads the environment on every call (it is cheap), so tests and scripts can change a variable and see
the effect. `describe()` returns a secret-free view for logs and for the evaluation run metadata.

Secrets (NEO4J_PASSWORD, MINDSEARCH_JWT_SECRET, MINDSEARCH_DATA_KEY and its key file) are deliberately NOT part of this
object: they are read only by the modules that need them (kg_builder, auth, encryptor), so they can never end up in a
log line or a results file by dumping the configuration.
"""
import os
from dataclasses import asdict, dataclass, fields

_DEFAULT_POLICY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "org_policy.yaml")


class ConfigError(ValueError):
    pass


def _int(env, name, default):
    raw = env.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, got '{raw}'") from None


def _float(env, name, default):
    raw = env.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got '{raw}'") from None


def _str(env, name, default):
    return env.get(name) or default


@dataclass(frozen=True)
class Settings:
    # ---- models
    generator_model: str          # GENERATOR_MODEL: the model vLLM serves for answers (and follow-up rewriting)
    embedding_model: str          # EMBEDDING_MODEL: sentence-transformers model for dense retrieval (CPU)
    reranker_model: str           # RERANKER_MODEL: cross-encoder that re-scores the candidates (CPU)
    spacy_model: str              # SPACY_MODEL: entity recogniser used when querying the graph
    judge_embeddings_model: str   # JUDGE_EMBEDDINGS_MODEL: embeddings RAGAS uses for answer relevancy
    # ---- model servers
    vllm_api_base: str            # VLLM_API_BASE: OpenAI-compatible endpoint of the generator
    judge_api_base: str           # JUDGE_API_BASE: endpoint of the judge model (defaults to VLLM_API_BASE)
    ollama_host: str              # OLLAMA_HOST: Ollama server (optional graph extractor); empty = the client's default
    # ---- vector store and graph
    qdrant_url: str               # QDRANT_URL: Qdrant server; empty = embedded folder qdrant_local_path
    qdrant_local_path: str        # QDRANT_LOCAL_PATH: embedded Qdrant folder (tests, evaluation indexes)
    qdrant_collection: str        # QDRANT_COLLECTION
    neo4j_uri: str                # NEO4J_URI
    neo4j_user: str               # NEO4J_USER
    # ---- graph extraction
    kg_extractor: str             # KG_EXTRACTOR: ollama (14B on CPU, best accuracy) or vllm (GPU, fast)
    kg_extract_model: str         # KG_EXTRACT_MODEL: the Ollama model
    kg_vllm_model: str            # KG_VLLM_MODEL: model for the vllm extractor (empty = the generator model)
    kg_extract_num_ctx: int       # KG_EXTRACT_NUM_CTX
    kg_extract_num_gpu: int       # KG_EXTRACT_NUM_GPU: 0 = CPU
    kg_extract_workers: int       # KG_EXTRACT_WORKERS: 0 = automatic (8 for vllm, 1 for ollama)
    # ---- queue
    celery_broker_url: str        # CELERY_BROKER_URL
    celery_result_backend: str    # CELERY_RESULT_BACKEND
    # ---- API behaviour
    pipeline_preset: str          # PIPELINE_PRESET: retrieval strategy (see pipeline_config.PRESETS)
    retrieval_workers: int        # RETRIEVAL_WORKERS: concurrent retrievals
    torch_threads: int            # TORCH_THREADS: 0 = CPU cores divided by retrieval_workers
    rewrite_enabled: bool         # REWRITE_ENABLED: follow-up question rewriting
    rewrite_timeout_s: float      # REWRITE_TIMEOUT_S
    model_max_len: int            # MODEL_MAX_LEN: the context window the language model is served with (match vLLM --max-model-len)
    answer_max_tokens: int        # ANSWER_MAX_TOKENS: longest answer the model may write; the prompt budget leaves room for it
    history_max_messages: int     # HISTORY_MAX_MESSAGES: what the rewriter sees
    history_generation_messages: int  # HISTORY_GENERATION_MESSAGES: what the answer writer sees
    history_token_budget: int     # HISTORY_TOKEN_BUDGET
    history_per_message_chars: int    # HISTORY_PER_MESSAGE_CHARS
    max_upload_mb: int            # MAX_UPLOAD_MB
    keep_uploads: bool            # KEEP_UPLOADS=1 keeps the plaintext upload after ingestion (debugging only)
    # ---- files
    users_file: str               # USERS_FILE
    org_policy_path: str          # ORG_POLICY_PATH
    audit_log_path: str           # AUDIT_LOG_PATH

    @property
    def qdrant_location(self) -> str:
        """What the Qdrant client connects to: the server URL if set, else the embedded folder."""
        return self.qdrant_url or self.qdrant_local_path

    @property
    def effective_kg_vllm_model(self) -> str:
        return self.kg_vllm_model or self.generator_model

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        vllm_base = _str(env, "VLLM_API_BASE", "http://localhost:8001/v1")
        s = cls(
            generator_model=_str(env, "GENERATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
            embedding_model=_str(env, "EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            reranker_model=_str(env, "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            spacy_model=_str(env, "SPACY_MODEL", "en_core_web_sm"),
            judge_embeddings_model=_str(env, "JUDGE_EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"),
            vllm_api_base=vllm_base,
            judge_api_base=_str(env, "JUDGE_API_BASE", vllm_base),
            ollama_host=_str(env, "OLLAMA_HOST", ""),
            qdrant_url=_str(env, "QDRANT_URL", ""),
            qdrant_local_path=_str(env, "QDRANT_LOCAL_PATH", "./qdrant_db"),
            qdrant_collection=_str(env, "QDRANT_COLLECTION", "mindsearch_docs"),
            neo4j_uri=_str(env, "NEO4J_URI", "neo4j://127.0.0.1:7687"),
            neo4j_user=_str(env, "NEO4J_USER", "neo4j"),
            kg_extractor=_str(env, "KG_EXTRACTOR", "ollama").lower(),
            kg_extract_model=_str(env, "KG_EXTRACT_MODEL", "qwen2.5:14b-instruct"),
            kg_vllm_model=_str(env, "KG_VLLM_MODEL", ""),
            kg_extract_num_ctx=_int(env, "KG_EXTRACT_NUM_CTX", 4096),
            kg_extract_num_gpu=_int(env, "KG_EXTRACT_NUM_GPU", 0),
            kg_extract_workers=_int(env, "KG_EXTRACT_WORKERS", 0),
            celery_broker_url=_str(env, "CELERY_BROKER_URL", "redis://localhost:6379/0"),
            celery_result_backend=_str(env, "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
            pipeline_preset=_str(env, "PIPELINE_PRESET", "hybrid_graph_chunks"),
            retrieval_workers=_int(env, "RETRIEVAL_WORKERS", 4),
            torch_threads=_int(env, "TORCH_THREADS", 0),
            rewrite_enabled=_str(env, "REWRITE_ENABLED", "1") == "1",
            rewrite_timeout_s=_float(env, "REWRITE_TIMEOUT_S", 6.0),
            model_max_len=_int(env, "MODEL_MAX_LEN", 8192),
            answer_max_tokens=_int(env, "ANSWER_MAX_TOKENS", 1024),
            history_max_messages=_int(env, "HISTORY_MAX_MESSAGES", 6),
            history_generation_messages=_int(env, "HISTORY_GENERATION_MESSAGES", 4),
            history_token_budget=_int(env, "HISTORY_TOKEN_BUDGET", 700),
            history_per_message_chars=_int(env, "HISTORY_PER_MESSAGE_CHARS", 600),
            max_upload_mb=_int(env, "MAX_UPLOAD_MB", 50),
            keep_uploads=env.get("KEEP_UPLOADS") == "1",
            users_file=_str(env, "USERS_FILE", "users.yaml"),
            org_policy_path=_str(env, "ORG_POLICY_PATH", _DEFAULT_POLICY),
            audit_log_path=_str(env, "AUDIT_LOG_PATH", "./logs/audit.jsonl"),
        )
        s.validate()
        return s

    def validate(self):
        if self.kg_extractor not in ("ollama", "vllm"):
            raise ConfigError(f"KG_EXTRACTOR must be 'ollama' or 'vllm', got '{self.kg_extractor}'")
        for name in ("retrieval_workers", "max_upload_mb", "kg_extract_num_ctx", "history_token_budget"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name.upper()} must be at least 1")
        if self.model_max_len < self.answer_max_tokens + 1024:
            raise ConfigError("MODEL_MAX_LEN must leave at least 1024 tokens of room for the prompt beyond ANSWER_MAX_TOKENS")
        for name in ("history_max_messages", "history_generation_messages", "history_per_message_chars", "torch_threads",
                     "kg_extract_workers"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name.upper()} cannot be negative")

    def describe(self) -> dict:
        """Everything above as a plain dict (no secrets exist in this object), for logs and run metadata."""
        return asdict(self)


def get_settings() -> Settings:
    return Settings.from_env()


def field_names():
    return [f.name for f in fields(Settings)]
