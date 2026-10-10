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
    # retrieval
    use_reranker: bool = True
    use_graph: bool = True
    candidate_k: int = 20          # candidate pool pulled from Qdrant
    hybrid: bool = False           # fuse dense and keyword (BM25-style) rankings with RRF instead of dense only
    hybrid_pool: int = 50          # candidates each branch contributes to the fusion
    final_k: int = 3               # max chunks kept after reranking
    rerank_margin: float = None    # None = always keep final_k. Else keep the best chunk plus others whose
                                   # cross-encoder score is within this many points of the best (dynamic context)
    graph_k: int = 3               # graph facts kept after reranking
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
    # Production default: hybrid retrieval + dynamic context + the knowledge graph. Retrieval measured on squad_dev_xl (RET-02);
    # end-to-end answer quality of this exact combination is re-judged at the end of Phase 5.
    "full_hybrid": PipelineConfig(name="full_hybrid", use_reranker=True, use_graph=True, hybrid=True, rerank_margin=4.0),
    "dense_rerank_fast": PipelineConfig(name="dense_rerank_fast", use_reranker=True, use_graph=False,
                                        candidate_k=10, rerank_margin=4.0),
}


def get_preset(name: str) -> PipelineConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown config '{name}'. Available: {', '.join(PRESETS)}")
    return PRESETS[name]
