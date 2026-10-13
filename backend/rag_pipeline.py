import time
from dataclasses import dataclass, field

from chunker import run_chunking_with_pages
from embedder import build_embeddings
from retriever import load_chunks, retrieve_detailed
import graph_ranker
from llm import stream_generate
from encryptor import Encryptor
from pipeline_config import PipelineConfig
from access_policy import AccessPolicy, DocACL, User, get_default_policy
from doc_identity import compute_doc_id
from doc_registry import DocumentRegistry


@dataclass
class RetrievalResult:
    """Everything one retrieval produced, so callers (API, evaluation) can inspect each stage."""
    chunks: list = field(default_factory=list)       # [{"chunk", "score"}] final, best first
    graph_facts: list = field(default_factory=list)  # [{"chunk", "score"}] final, best first
    dense_pool: list = field(default_factory=list)   # decrypted dense candidates, pre-rerank order
    graph_pool: list = field(default_factory=list)   # raw graph facts before reranking
    timings_ms: dict = field(default_factory=dict)
    graph_chunks_added: int = 0                       # final passages that came in through the graph, not the search

    def sources(self):
        """Which documents the final chunks came from, in rank order, one entry per document:
        [{"doc_id", "chunk_index", "page"}]. The UI turns these into citations; `page` (1-based, PDFs only) is the first page of the
        best-ranked chunk from that document."""
        seen, out = set(), []
        for r in self.chunks:
            if r.get("doc_id") and r["doc_id"] not in seen:
                seen.add(r["doc_id"])
                out.append({"doc_id": r["doc_id"], "chunk_index": r.get("chunk_index"), "page": r.get("page")})
        return out

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
        self._registry = None
        self.kg_builder = None
        if self.config.use_graph:
            from kg_builder import KnowledgeGraphBuilder
            self.kg_builder = KnowledgeGraphBuilder()

    @property
    def registry(self) -> DocumentRegistry:
        """Document-level metadata (filename, status, ...). Created on first use."""
        if self._registry is None:
            self._registry = DocumentRegistry(self.config.qdrant_path, self.encryptor, self.policy)
        return self._registry

    # ---------- ingestion ----------

    def ingest_vectors(self, file_path, acl: DocACL, doc_id=None):
        """Chunk -> embed -> encrypt (bound to record + ACL) -> index in Qdrant. Returns the plaintext chunks."""
        cfg = self.config
        doc_id = doc_id or compute_doc_id(file_path, acl)
        chunks = run_chunking_with_pages(file_path, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap,
                                         split_long=cfg.split_long_paragraphs)
        plaintext_chunks = [c.text for c in chunks]
        build_embeddings(plaintext_chunks, self.encryptor, acl, cfg.qdrant_path,
                         collection_name=cfg.collection, doc_id=doc_id, pages=[(c.page_start, c.page_end) for c in chunks])
        return plaintext_chunks

    def ingest_graph(self, plaintext_chunks, acl: DocACL, doc_id=None):
        """Extract entities/relations per chunk and write them to Neo4j.
        Returns ExtractionStats (None when the graph is disabled)."""
        if self.kg_builder is None:
            return None
        return self.kg_builder.process_chunks(plaintext_chunks, acl, doc_id=doc_id)

    def ingest_document(self, file_path, acl: DocACL, doc_id=None):
        """Idempotent: the same file with the same ACL always yields the same doc_id and point ids, so running
        this twice (or retrying after a partial failure) leaves exactly one copy.
        Returns {"doc_id", "chunks"}."""
        doc_id = doc_id or compute_doc_id(file_path, acl)
        plaintext_chunks = self.ingest_vectors(file_path, acl, doc_id=doc_id)
        self.ingest_graph(plaintext_chunks, acl, doc_id=doc_id)
        return {"doc_id": doc_id, "chunks": len(plaintext_chunks)}

    def delete_document(self, doc_id) -> dict:
        """Remove a document everywhere: chunks, graph facts, then its registry entry (last, so a half-finished
        delete stays visible and can simply be retried). Idempotent."""
        from embedder import delete_document_chunks
        cfg = self.config
        report = {"chunks_deleted": delete_document_chunks(doc_id, cfg.qdrant_path, cfg.collection)}
        if self.kg_builder is not None:
            report["graph"] = self.kg_builder.delete_document(doc_id)
        self.registry.delete(doc_id)
        return report

    # ---------- retrieval ----------

    def retrieve(self, query, user: User) -> RetrievalResult:
        cfg = self.config
        result = RetrievalResult()

        t0 = time.perf_counter()
        detail = retrieve_detailed(
            query, cfg.qdrant_path, user, self.policy, candidate_k=cfg.candidate_k, final_k=cfg.final_k,
            reranker=self.reranker, encryptor=self.encryptor,
            collection_name=cfg.collection, rerank_margin=cfg.rerank_margin,
            hybrid=cfg.hybrid, hybrid_pool=cfg.hybrid_pool,
        )
        result.chunks, result.dense_pool = detail["results"], detail["pool"]
        result.timings_ms["dense_and_rerank"] = (time.perf_counter() - t0) * 1000

        if self.kg_builder is not None:
            t0 = time.perf_counter()
            seed_keys = []
            if cfg.graph_expand and result.chunks:
                seed_keys = self._graph_expand(query, user, result, detail)
            if cfg.graph_k > 0:
                result.graph_pool = self.kg_builder.get_graph_context(query, user, seed_keys=seed_keys)
                if self.reranker is not None:
                    result.graph_facts = self.reranker.rerank(query, result.graph_pool, top_k=cfg.graph_k)
                else:
                    result.graph_facts = [{"chunk": f, "score": 0.0} for f in result.graph_pool[:cfg.graph_k]]
            result.timings_ms["graph"] = (time.perf_counter() - t0) * 1000

        return result

    def graph_candidates(self, query, user, detail):
        """Score the search pool plus graph-linked passages for hop-aware ranking. Returns a list of
        {"chunk", "doc_id", "chunk_index", "ce", "cover", "page_q", "page_s", "via_graph"} and the seed entity keys.

        A second-hop passage does not look like the question, so the cross-encoder alone ranks it too low. The graph supplies
        what the text cannot: which entities the question names (Q), which entities the best passages mention (S, the
        bridge), which passages mention them, and which passage is the "page" (title) of an entity. Those become bonuses."""
        cfg, kg = self.config, self.kg_builder
        pool, metas = detail["pool"], detail["pool_meta"]
        if not pool or self.reranker is None:
            return [], []
        scored = self.reranker.rerank(query, pool, top_k=len(pool))          # best first, every pool item scored
        items = [{"chunk": r["chunk"], **metas[r["original_index"]], "ce": r["score"], "via_graph": 0,
                  "pool_rank": 1.0 / (1 + r["original_index"])} for r in scored]
        seed_refs = [(c["doc_id"], c["chunk_index"]) for c in items[:cfg.graph_seed_k]
                     if c.get("doc_id") is not None and c.get("chunk_index") is not None]
        s_keys = set(kg.entities_in_chunks(seed_refs, user))
        q_keys = set(kg.query_entity_keys(query))
        cand = kg.candidate_chunks(sorted(q_keys | s_keys), user, cfg.graph_hub_limit)
        in_pool = {(c.get("doc_id"), c.get("chunk_index")) for c in items}
        new_refs = sorted((r for r in cand if r not in in_pool), key=lambda r: (-len(cand[r]["keys"]) - 2 * len(cand[r]["pages"]), r))
        extra = load_chunks(new_refs[:cfg.graph_chunk_pool], user, self.policy, self.encryptor, cfg.qdrant_path, cfg.collection)
        if extra:
            for r in self.reranker.rerank(query, [e["chunk"] for e in extra], top_k=len(extra)):
                items.append({**extra[r["original_index"]], "ce": r["score"], "via_graph": 1, "pool_rank": 0.0})
        mentions = kg.entities_by_chunk([(c["doc_id"], c["chunk_index"]) for c in items if c.get("doc_id") is not None], user)
        best_ce = max(c["ce"] for c in items)
        bridge = s_keys - q_keys
        for c in items:
            ref = (c.get("doc_id"), c.get("chunk_index"))
            pages = cand.get(ref, {}).get("pages", set())
            seen = mentions.get(ref, set())
            c["cover"] = len(seen & q_keys)
            c["cover_frac"] = c["cover"] / max(1, len(q_keys))
            c["cover_s"] = min(5, len(seen & bridge))
            c["page_q"] = 1 if pages & q_keys else 0
            c["page_s"] = 1 if pages & bridge else 0
            c["ce_gap"] = c["ce"] - best_ce
        return items, sorted(q_keys | s_keys)

    def graph_select(self, items):
        """Final passages by the hop-aware score (graph_ranker), best first."""
        cfg = self.config
        for c in items:
            c["score"] = graph_ranker.score(c)
        chosen = sorted(items, key=lambda c: -c["score"])[:cfg.final_k]
        if cfg.rerank_margin is not None and chosen:
            chosen = [c for c in chosen if c["score"] >= chosen[0]["score"] - cfg.rerank_margin]
        return chosen

    def _graph_expand(self, query, user, result, detail):
        """Graph-guided passage hops (the multi-hop step): choose the final passages with the hop-aware score.
        Returns the seed entity keys (also used to fetch graph facts)."""
        items, keys = self.graph_candidates(query, user, detail)
        if items:
            result.chunks = self.graph_select(items)
            result.graph_chunks_added = sum(1 for r in result.chunks if r.get("via_graph"))
        return keys

    def retrieve_context(self, query, user: User):
        return self.retrieve(query, user).to_context()

    # ---------- generation ----------

    async def stream_answer(self, query, context, history=None):
        ctx = "\n\n".join(context)
        async for token in stream_generate(self.config.generator_model, query, ctx, history):
            yield token
