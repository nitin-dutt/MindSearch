from metrics import timed


def retrieve_detailed(query, store, user, candidate_k=20, final_k=3, reranker=None, rerank_margin=None, hybrid=False,
                      hybrid_pool=50, doc_ids=None):
    """
    Search `store` (chunk_store.ChunkStore) as `user`, optionally followed by cross-encoder re-ranking and a dynamic cutoff
    (rerank_margin) that sends only the chunks that score close to the best one. The store applies the access filter and decrypts;
    with `doc_ids` the search is limited to those documents.

    Returns a dict:
      results   - up to final_k passages {"chunk", "score", "doc_id", "chunk_index", "page"}, best first
      pool      - the texts of the search candidates (up to candidate_k) in search order, before re-ranking. The evaluation harness
                  uses it to measure the search stage.
      pool_meta - where each pool item came from ({"doc_id", "chunk_index", "page"[, "ocr"]}), same order as `pool`
      scored    - the cross-encoder's score for EVERY pool item, best first ([{"chunk", "score", "original_index"}]; empty without a
                  reranker). The graph step reuses it instead of scoring the same passages a second time (alignment_plan PERF-01).

    An empty index (or nothing the user may see) returns empty lists. No placeholder text is injected into the context.
    """
    found = store.search(query, user, k=candidate_k, hybrid=hybrid, hybrid_pool=hybrid_pool, doc_ids=doc_ids)
    if not found:
        return {"results": [], "pool": [], "pool_meta": [], "scored": []}
    pool = [p["chunk"] for p in found]
    pool_meta = [{k: v for k, v in p.items() if k not in ("chunk", "score")} for p in found]

    scored = []
    if reranker:
        with timed("rerank"):
            scored = reranker.rerank(query, pool, top_k=len(pool))     # every candidate scored ONCE, best first (PERF-01)
        results = [{"chunk": r["chunk"], "score": r["score"], **pool_meta[r["original_index"]]} for r in scored[:final_k]]
        if rerank_margin is not None and results:
            # dynamic context: keep the best chunk, drop others that score far below it
            best = results[0]["score"]
            results = [r for r in results if r["score"] >= best - rerank_margin]
    else:
        results = found[:final_k]

    return {"results": results, "pool": pool, "pool_meta": pool_meta, "scored": scored}
