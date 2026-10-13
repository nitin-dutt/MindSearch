"""Pipeline configuration.

One object describes how a RAGPipeline retrieves, so the same code can run as a
dense-only baseline, dense + rerank, or the full GraphRAG pipeline. The evaluation
harness uses these presets for the ablation study (alignment_plan EVAL-09).
"""
from dataclasses import dataclass, asdict, replace

from settings import get_settings

_cfg = get_settings()
DEFAULT_GENERATOR_MODEL = _cfg.generator_model


@dataclass(frozen=True)
class PipelineConfig:
    name: str = "full"
    # storage
    # A Qdrant server URL when QDRANT_URL is set (the Docker stack), else an embedded folder
    qdrant_path: str = _cfg.qdrant_location
    collection: str = _cfg.qdrant_collection
    # ingestion
    chunk_size: int = 150          # words
    chunk_overlap: int = 20        # words
    split_long_paragraphs: bool = True   # cut paragraphs longer than chunk_size at sentence boundaries (False for the benchmark corpora,
                                         # whose lines are already whole passages)
    # retrieval
    use_reranker: bool = True
    use_graph: bool = True
    candidate_k: int = 20          # candidate pool pulled from Qdrant
    hybrid: bool = False           # fuse dense and keyword (BM25-style) rankings with RRF instead of dense only
    hybrid_pool: int = 50          # candidates each branch contributes to the fusion
    final_k: int = 3               # max chunks kept after reranking
    rerank_margin: float = None    # None = always keep final_k. Else keep the best chunk plus others whose
                                   # cross-encoder score is within this many points of the best (dynamic context)
    graph_k: int = 3               # graph facts kept after reranking (0 = no facts in the prompt)
    graph_expand: bool = False     # graph-guided passage hops: also consider passages that share entities with the top ones
    graph_seed_k: int = 3          # how many of the top passages seed the expansion
    graph_chunk_pool: int = 15     # extra passages the expansion may add before the cross-encoder picks
    graph_hub_limit: int = 12      # an entity found in more passages than this is too generic to follow
    # generation
    generator_model: str = _cfg.generator_model

    def to_dict(self):
        return asdict(self)

    def with_overrides(self, **kwargs):
        return replace(self, **kwargs)


# Defaults match the behaviour the project had before the config existed
# (20 candidates -> cross-encoder -> top 3 chunks + top 3 graph facts).
PRESETS = {
    "dense_only": PipelineConfig(name="dense_only", use_reranker=False, use_graph=False),
    "dense_rerank": PipelineConfig(name="dense_rerank", use_reranker=True, use_graph=False),
    "full": PipelineConfig(name="full", use_reranker=True, use_graph=True),
    # Dynamic context, margin tuned on squad_dev (see evaluation/tune_context.py and alignment_plan RET-04):
    # about 60% fewer context words for -0.005 recall. "fast" also reranks only 10 candidates (~2x faster).
    "dense_rerank_dynamic": PipelineConfig(name="dense_rerank_dynamic", use_reranker=True, use_graph=False,
                                           rerank_margin=4.0),
    # Hybrid retrieval (alignment_plan RET-02): dense + keyword rankings fused by RRF, then the cross-encoder.
    "hybrid_rerank": PipelineConfig(name="hybrid_rerank", use_reranker=True, use_graph=False, hybrid=True),
    "hybrid_rerank_dynamic": PipelineConfig(name="hybrid_rerank_dynamic", use_reranker=True, use_graph=False,
                                            hybrid=True, rerank_margin=4.0),
    # Production default: hybrid retrieval + the knowledge graph, WITHOUT the dynamic cutoff. The cutoff (margin 4.0) was tuned on
    # single-hop SQuAD and cuts the second supporting passage of multi-hop questions (HotpotQA both-passages-in-top-3: 0.700 -> 0.505).
    # Multi-hop questions need more than one passage: the graph-aware preset sends 4 (cross-validated on HotpotQA: both supporting
    # passages in the context 0.795 -> 0.880 compared with plain hybrid at the same 4 passages).
    "hybrid_rerank_k4": PipelineConfig(name="hybrid_rerank_k4", use_reranker=True, use_graph=False, hybrid=True, final_k=4),
    "hybrid_graph_chunks": PipelineConfig(name="hybrid_graph_chunks", use_reranker=True, use_graph=True, hybrid=True,
                                          graph_expand=True, graph_k=0, final_k=4),
    # With graph facts added to the prompt (measured on HotpotQA: no further gain over hybrid_graph_chunks, which is the default).
    # No dynamic cutoff here: it was tuned on single-hop SQuAD and cuts the second hop (both passages in the top 3: 0.700 -> 0.505).
    "full_hybrid": PipelineConfig(name="full_hybrid", use_reranker=True, use_graph=True, hybrid=True, graph_expand=True, final_k=4),
    "hybrid_graph_facts": PipelineConfig(name="hybrid_graph_facts", use_reranker=True, use_graph=True, hybrid=True),
    "dense_rerank_fast": PipelineConfig(name="dense_rerank_fast", use_reranker=True, use_graph=False,
                                        candidate_k=10, rerank_margin=4.0),
}


def get_preset(name: str) -> PipelineConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown config '{name}'. Available: {', '.join(PRESETS)}")
    return PRESETS[name]
