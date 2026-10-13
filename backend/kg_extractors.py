"""Knowledge-graph triple extraction (alignment_plan KG-01).

Extraction runs per chunk, never on a whole document, so a prompt can never be silently truncated
and lose its instructions. Output is forced into a JSON schema and validated with pydantic.
Failures are counted and reported (ExtractionStats), not swallowed.

Backends (select with KG_EXTRACTOR):
  ollama (default)  qwen2.5:14b-instruct on CPU: best accuracy, slow. Env: KG_EXTRACT_MODEL,
                    KG_EXTRACT_NUM_GPU (0 = CPU), KG_EXTRACT_NUM_CTX, OLLAMA_HOST.
  vllm              the GPU vLLM server with guided JSON decoding: fast, lighter model.
                    Env: KG_VLLM_MODEL, VLLM_API_BASE.
An optional JSONL cache (CachedExtractor) makes long runs resumable and re-runs free.
"""
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import List

from pydantic import BaseModel, ValidationError

from settings import get_settings

log = logging.getLogger("mindsearch.kg")

PROMPT_VERSION = "v2"
MAX_NAME_CHARS = 80

SYSTEM_PROMPT = """You extract knowledge-graph facts from text.
Return JSON: {"triples": [{"subject": "...", "relation": "...", "object": "..."}]}

Rules:
- subject and object are concise names of real entities: people, organizations, places, products, works, concepts, dates, quantities.
- Use the entity's full name, never a pronoun like "he", "it" or "the company".
- relation is ONE uppercase word, or WORD_WORD with an underscore (for example FOUNDED_BY, LOCATED_IN, USES, BORN_IN).
- Include only facts the text states explicitly. Do not guess or add outside knowledge.
- If the text contains no clear facts, return {"triples": []}."""


class Triple(BaseModel):
    subject: str
    relation: str
    object: str


class Extraction(BaseModel):
    triples: List[Triple] = []


class ExtractionError(Exception):
    pass


# ----------------------------------------------------------------- cleaning

_REL_BAD = re.compile(r"[^A-Z0-9]+")


def normalize_relation(rel: str) -> str:
    cleaned = _REL_BAD.sub("_", rel.upper().strip()).strip("_")
    return cleaned or "RELATED_TO"


def clean_triples(triples):
    """Strip, normalize relations, drop empty / self-referential / oversized triples, de-duplicate."""
    seen, out = set(), []
    for t in triples:
        s, o = " ".join(t.subject.split()), " ".join(t.object.split())
        if not s or not o or s.casefold() == o.casefold():
            continue
        if len(s) > MAX_NAME_CHARS or len(o) > MAX_NAME_CHARS:
            continue
        rel = normalize_relation(t.relation)
        key = (s.casefold(), rel, o.casefold())
        if key not in seen:
            seen.add(key)
            out.append(Triple(subject=s, relation=rel, object=o))
    return out


# ---------------------------------------------------------------- backends

class OllamaExtractor:
    def __init__(self, model=None, num_ctx=None, num_gpu=None):
        import ollama
        cfg = get_settings()
        self.model = model or cfg.kg_extract_model
        self.num_ctx = int(num_ctx or cfg.kg_extract_num_ctx)
        self.num_gpu = int(num_gpu if num_gpu is not None else cfg.kg_extract_num_gpu)
        self.client = ollama.Client(host=cfg.ollama_host or None)  # None = the library default
        self.name = f"ollama:{self.model}"
        self.truncated = 0

    def extract(self, chunk: str):
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"Text:\n{chunk}"}],
            format=Extraction.model_json_schema(),
            options={"temperature": 0, "num_ctx": self.num_ctx, "num_gpu": self.num_gpu},
            keep_alive="30m",
        )
        used = getattr(resp, "prompt_eval_count", None)
        if used is not None and used >= self.num_ctx - 8:
            self.truncated += 1
            log.warning("Prompt hit the context limit (%s/%s tokens); output may be unreliable", used, self.num_ctx)
        return parse_extraction(resp.message.content)


class VLLMExtractor:
    def __init__(self, model=None, base_url=None):
        from openai import OpenAI
        cfg = get_settings()
        self.model = model or cfg.effective_kg_vllm_model
        self.client = OpenAI(api_key="EMPTY", base_url=base_url or cfg.vllm_api_base)
        self.name = f"vllm:{self.model}"
        self.truncated = 0

    def extract(self, chunk: str):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"Text:\n{chunk}"}],
            temperature=0, max_tokens=800,
            extra_body={"guided_json": Extraction.model_json_schema()},
        )
        return parse_extraction(resp.choices[0].message.content)


def parse_extraction(content: str):
    try:
        return Extraction.model_validate_json(content).triples
    except (ValidationError, ValueError) as e:
        raise ExtractionError(f"model output was not valid extraction JSON: {e}") from e


class CachedExtractor:
    """Append-only JSONL cache keyed by (prompt version, extractor, chunk). Failures are never cached."""

    def __init__(self, inner, path):
        self.inner, self.path = inner, path
        self.name = inner.name
        self.hits = 0
        self._lock = threading.Lock()
        self._store = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self._store[row["key"]] = row["triples"]

    def _key(self, chunk):
        return hashlib.sha256(f"{PROMPT_VERSION}|{self.name}|{chunk}".encode()).hexdigest()

    def extract(self, chunk):
        key = self._key(chunk)
        if key in self._store:
            self.hits += 1
            return [Triple(**t) for t in self._store[key]]
        triples = self.inner.extract(chunk)
        with self._lock:
            self._store[key] = [t.model_dump() for t in triples]
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "triples": self._store[key]}) + "\n")
        return triples

    def __getattr__(self, item):  # proxy counters like .truncated
        return getattr(self.inner, item)


def make_extractor_from_env():
    kind = get_settings().kg_extractor
    if kind == "ollama":
        return OllamaExtractor()
    if kind == "vllm":
        return VLLMExtractor()
    raise ValueError(f"KG_EXTRACTOR must be 'ollama' or 'vllm', got '{kind}'")


# ---------------------------------------------------------------- orchestration

@dataclass
class ExtractionStats:
    extractor: str = ""
    chunks_total: int = 0
    chunks_ok: int = 0
    chunks_skipped_short: int = 0
    chunks_failed: int = 0
    cache_hits: int = 0
    truncated_prompts: int = 0
    triples_raw: int = 0
    triples_kept: int = 0
    seconds: float = 0.0

    def to_dict(self):
        return asdict(self)

    @property
    def failure_rate(self):
        attempted = self.chunks_ok + self.chunks_failed
        return self.chunks_failed / attempted if attempted else 0.0


def extract_with_retry(extractor, chunk, attempts=2):
    last = None
    for _ in range(attempts):
        try:
            return extractor.extract(chunk)
        except Exception as e:  # noqa: BLE001 - any backend error counts as a failed attempt
            last = e
    raise ExtractionError(str(last)) from last


def extract_chunks(extractor, chunks, workers=1, min_words=5, attempts=2, progress_every=10, mentions_out=None):
    """Extract triples from every chunk. Returns (cleaned triples, ExtractionStats).

    If `mentions_out` (a list) is given, it is filled with (entity name, chunk index) for every entity that a chunk's
    facts mention. The graph stores this so a fact can lead back to the passage it came from (graph-guided retrieval)."""
    stats = ExtractionStats(extractor=extractor.name, chunks_total=len(chunks))
    started = time.time()
    done = 0
    lock = threading.Lock()

    def run(chunk):
        nonlocal done
        if len(chunk.split()) < min_words:
            result = ("skipped", [])
        else:
            try:
                result = ("ok", extract_with_retry(extractor, chunk, attempts))
            except ExtractionError as e:
                log.warning("Extraction failed for a chunk after %d attempts: %s", attempts, e)
                result = ("failed", [])
        with lock:
            done += 1
            if done % progress_every == 0 or done == len(chunks):
                log.info("KG extraction %d/%d chunks (%.0fs elapsed)", done, len(chunks), time.time() - started)
        return result

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(run, chunks))

    raw = []
    for index, (status, triples) in enumerate(results):
        if status == "ok":
            stats.chunks_ok += 1
            raw.extend(triples)
            if mentions_out is not None:
                names = {n for t in clean_triples(triples) for n in (t.subject, t.object)}
                mentions_out.extend((name, index) for name in sorted(names))
        elif status == "skipped":
            stats.chunks_skipped_short += 1
        else:
            stats.chunks_failed += 1
    cleaned = clean_triples(raw)
    stats.triples_raw, stats.triples_kept = len(raw), len(cleaned)
    stats.cache_hits = getattr(extractor, "hits", 0)
    stats.truncated_prompts = getattr(extractor, "truncated", 0)
    stats.seconds = round(time.time() - started, 1)
    return cleaned, stats
