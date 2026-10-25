import logging

from embedder import fetch_chunks, search_embeddings
from metrics import DECRYPT_FAILURES, timed

log = logging.getLogger("mindsearch.security")


def _decrypt_hits(payloads, aads, scores, metas, encryptor):
    """Decrypt retrieved chunks. A chunk that fails its integrity check (wrong key, copied ciphertext, edited access rule)
    is dropped and reported; it is never served and never replaced by empty text. Returns (texts, scores, metas)."""
    texts, kept_scores, kept_metas = [], [], []
    for score, payload, aad, meta in zip(scores, payloads, aads, metas):
        try:
            texts.append(encryptor.decrypt(payload, aad))
            kept_scores.append(score)
            kept_metas.append(meta)
        except Exception:  # noqa: BLE001 - InvalidTag, short payload, bad utf-8: all mean "do not serve"
            log.error("SECURITY: a stored chunk failed its integrity check and was skipped (possible tampering)")
            DECRYPT_FAILURES.inc()
    return texts, kept_scores, kept_metas


def decrypt_hits(*args, **kwargs):
    with timed("decrypt"):
        return _decrypt_hits(*args, **kwargs)


def load_chunks(refs, user, policy, encryptor, db_path=None, collection_name=None):
    """Decrypted passages for (doc_id, chunk_index) references the user may read: [{"chunk", "doc_id", "chunk_index"}]."""
    payloads, aads, metas = fetch_chunks(refs, user, policy, db_path=db_path, collection_name=collection_name)
    texts, _, kept = decrypt_hits(payloads, aads, [0.0] * len(payloads), metas, encryptor)
    return [{"chunk": t, **m} for t, m in zip(texts, kept)]


def retrieve(*args, **kwargs):
    """(results, pool) - see retrieve_detailed."""
    detail = retrieve_detailed(*args, **kwargs)
    return detail["results"], detail["pool"]


def retrieve_detailed(query, db_path, user, policy, candidate_k=20, final_k=3, reranker=None, encryptor=None,
                      collection_name=None, rerank_margin=None, hybrid=False, hybrid_pool=50, doc_ids=None):
    """
    Dense (or hybrid dense + keyword) retrieval from Qdrant, optionally followed by cross-encoder re-ranking and a dynamic cutoff
    (rerank_margin) that sends only the chunks that score close to the best one.

    Returns (results, pool):
      results - up to final_k dicts {"chunk", "score", "doc_id", "chunk_index"}, best first (doc_id and
                chunk_index say which document and which part of it the chunk came from)
      pool    - the decrypted dense candidates (up to candidate_k) in dense-score order,
                before re-ranking. The evaluation harness uses it to measure the dense stage.

    An empty index (or nothing the user may see) returns ([], []). No placeholder text is
    injected into the context.
    """
    dense_scores, encrypted_payloads, aads, metas = search_embeddings(
        query, user, policy, db_path=db_path, collection_name=collection_name, k=candidate_k,
        hybrid=hybrid, encryptor=encryptor, hybrid_pool=hybrid_pool, doc_ids=doc_ids,
    )

    if not encrypted_payloads:
        return {"results": [], "pool": [], "pool_meta": []}

    pool, pool_scores, pool_meta = decrypt_hits(encrypted_payloads, aads, dense_scores, metas, encryptor)
    if not pool:
        return {"results": [], "pool": [], "pool_meta": []}

    if reranker and pool:
        with timed("rerank"):
            reranked = reranker.rerank(query, pool, top_k=final_k)
        results = [{"chunk": r["chunk"], "score": r["score"], **pool_meta[r["original_index"]]} for r in reranked]
        if rerank_margin is not None and results:
            # dynamic context: keep the best chunk, drop others that score far below it
            best = results[0]["score"]
            results = [r for r in results if r["score"] >= best - rerank_margin]
    else:
        results = [{"chunk": pool[i], "score": pool_scores[i], **pool_meta[i]} for i in range(min(final_k, len(pool)))]

    return {"results": results, "pool": pool, "pool_meta": pool_meta}
