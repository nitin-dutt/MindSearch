import logging

from embedder import search_embeddings

log = logging.getLogger("mindsearch.security")


def retrieve(query, db_path, user, policy, candidate_k=20, final_k=3, reranker=None, encryptor=None,
             collection_name="mindsearch_docs", rerank_margin=None):
    """
    Dense retrieval from Qdrant, optionally followed by cross-encoder re-ranking and a dynamic cutoff
    (rerank_margin) that sends only the chunks that score close to the best one.

    Returns (results, pool):
      results - up to final_k dicts {"chunk": str, "score": float}, best first
      pool    - the decrypted dense candidates (up to candidate_k) in dense-score order,
                before re-ranking. The evaluation harness uses it to measure the dense stage.

    An empty index (or nothing the user may see) returns ([], []). No placeholder text is
    injected into the context.
    """
    dense_scores, encrypted_payloads, aads = search_embeddings(
        query, user, policy, db_path=db_path, collection_name=collection_name, k=candidate_k
    )

    if not encrypted_payloads:
        return [], []

    # Decrypt. A chunk that fails its integrity check (wrong key, copied ciphertext, edited access rule) is
    # dropped and reported; it is never served and never replaced by empty text.
    pool, pool_scores = [], []
    for score, payload, aad in zip(dense_scores, encrypted_payloads, aads):
        try:
            pool.append(encryptor.decrypt(payload, aad))
            pool_scores.append(score)
        except Exception:  # noqa: BLE001 - InvalidTag, short payload, bad utf-8: all mean "do not serve"
            log.error("SECURITY: a stored chunk failed its integrity check and was skipped (possible tampering)")
    if not pool:
        return [], []

    if reranker and pool:
        reranked = reranker.rerank(query, pool, top_k=final_k)
        results = [{"chunk": r["chunk"], "score": r["score"]} for r in reranked]
        if rerank_margin is not None and results:
            # dynamic context: keep the best chunk, drop others that score far below it
            best = results[0]["score"]
            results = [r for r in results if r["score"] >= best - rerank_margin]
    else:
        results = [{"chunk": pool[i], "score": pool_scores[i]} for i in range(min(final_k, len(pool)))]

    return results, pool
