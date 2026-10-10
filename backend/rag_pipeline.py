import time
from dataclasses import dataclass, field

from chunker import run_chunking
from embedder import build_embeddings
from retriever import retrieve
from llm import stream_generate
from encryptor import Encryptor
from pipeline_config import PipelineConfig
from access_policy import AccessPolicy, DocACL, User, get_default_policy
from doc_identity import compute_doc_id


@dataclass
class RetrievalResult:
    """Everything one retrieval produced, so callers (API, evaluation) can inspect each stage."""
    chunks: list = field(default_factory=list)       # [{"chunk", "score"}] final, best first
    graph_facts: list = field(default_factory=list)  # [{"chunk", "score"}] final, best first
    dense_pool: list = field(default_factory=list)   # decrypted dense candidates, pre-rerank order
    graph_pool: list = field(default_factory=list)   # raw graph facts before reranking
    timings_ms: dict = field(default_factory=dict)

    def to_context(self):
        """Prompt-ready context strings, wrapped in the XML tags the system prompt expects."""
        return ([f"<doc>{r['chunk']}</doc>" for r in self.chunks]
                + [f"<kg_fact>{r['chunk']}</kg_fact>" for r in self.graph_facts])


class RAGPipeline:
    def __init__(self, config: PipelineConfig = None, encryptor: Encryptor = None, policy: AccessPolicy = None):
        self.config = config or PipelineConfig()
        self.policy = policy or get_default_policy()
        # Models/clients are only created for the stages the config uses, so a dense-only
        # run needs neither Neo4j nor spaCy.
        self.reranker = None
        if self.config.use_reranker:
            from reranker import ReRanker
            self.reranker = ReRanker()
        self.encryptor = encryptor or Encryptor()
        self.kg_builder = None
        if self.config.use_graph:
            from kg_builder import KnowledgeGraphBuilder
            self.kg_builder = KnowledgeGraphBuilder()

    # ---------- ingestion ----------

    def ingest_vectors(self, file_path, acl: DocACL, doc_id=None):
        """Chunk -> embed -> encrypt (bound to record + ACL) -> index in Qdrant. Returns the plaintext chunks."""
        cfg = self.config
        doc_id = doc_id or compute_doc_id(file_path, acl)
        plaintext_chunks = run_chunking(file_path, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        build_embeddings(plaintext_chunks, self.encryptor, acl, cfg.qdrant_path,
                         collection_name=cfg.collection, doc_id=doc_id)
        return plaintext_chunks

    def ingest_graph(self, plaintext_chunks, acl: DocACL, doc_id=None):
        """Extract entities/relations per chunk and write them to Neo4j.
        Returns ExtractionStats (None when the graph is disabled)."""
        if self.kg_builder is None:
            return None
        return self.kg_builder.process_chunks(plaintext_chunks, acl, doc_id=doc_id)

    def ingest_document(self, file_path, acl: DocACL):
        """Idempotent: the same file with the same ACL always yields the same doc_id and point ids, so running
        this twice (or retrying after a partial failure) leaves exactly one copy."""
        doc_id = compute_doc_id(file_path, acl)
        plaintext_chunks = self.ingest_vectors(file_path, acl, doc_id=doc_id)
        self.ingest_graph(plaintext_chunks, acl, doc_id=doc_id)
        return doc_id

    # ---------- retrieval ----------

    def retrieve(self, query, user: User) -> RetrievalResult:
        cfg = self.config
        result = RetrievalResult()

        t0 = time.perf_counter()
        result.chunks, result.dense_pool = retrieve(
            query, cfg.qdrant_path, user, self.policy, candidate_k=cfg.candidate_k, final_k=cfg.final_k,
            reranker=self.reranker, encryptor=self.encryptor,
            collection_name=cfg.collection, rerank_margin=cfg.rerank_margin,
        )
        result.timings_ms["dense_and_rerank"] = (time.perf_counter() - t0) * 1000

        if self.kg_builder is not None:
            t0 = time.perf_counter()
            result.graph_pool = self.kg_builder.get_graph_context(query, user)
            if self.reranker is not None:
                result.graph_facts = self.reranker.rerank(query, result.graph_pool, top_k=cfg.graph_k)
            else:
                result.graph_facts = [{"chunk": f, "score": 0.0} for f in result.graph_pool[:cfg.graph_k]]
            result.timings_ms["graph"] = (time.perf_counter() - t0) * 1000

        return result

    def retrieve_context(self, query, user: User):
        return self.retrieve(query, user).to_context()

    # ---------- generation ----------

    async def stream_answer(self, query, context):
        ctx = "\n\n".join(context)
        async for token in stream_generate(self.config.generator_model, query, ctx):
            yield token
