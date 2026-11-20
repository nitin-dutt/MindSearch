"""Why does graph extraction fail for some passages? (alignment_plan MODEL-01 follow-up, 2026-11-19)

    python -m evaluation.extraction_failures --dataset hotpot [--sample 30] [--max-tokens 800]

Run from backend/ inside the worker container with KG_EXTRACTOR=vllm. Rebuilds the evaluation's own chunks (same preset settings),
finds the passages missing from the extraction cache (failures are never cached, so those are exactly the failed ones), re-runs a
sample against the live vLLM model with the production request, and classifies each reply:
  ok          valid extraction JSON now (the failure did not repeat)
  truncated   the reply stopped at the token limit (finish_reason 'length'), so the JSON is cut off
  invalid     the reply finished but is not valid extraction JSON
  error       the request itself failed (timeout, server error)
Read-only: it does not write to the cache or the graph.
"""
import argparse
import hashlib
import json
import os
from collections import Counter

from evaluation.datasets import DATASETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpot", choices=list(DATASETS))
    ap.add_argument("--config", default="hybrid_graph_chunks")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=800)
    args = ap.parse_args()

    import kg_extractors as K
    from chunker import run_chunking
    from pipeline_config import get_preset

    cfg = get_preset(args.config)
    corpus = os.path.join("eval_data", DATASETS[args.dataset]["corpus"])
    cache_path = os.path.join("qdrant_eval", "kg_cache", f"eval_{args.dataset}.jsonl")
    cached = {json.loads(line)["key"] for line in open(cache_path, encoding="utf-8")}
    ex = K.VLLMExtractor()
    # the evaluation chunks benchmark corpora WITHOUT splitting long paragraphs (generate.build_pipeline), so gold passages stay whole
    chunks = run_chunking(corpus, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap, split_long=False)
    texts = [c if isinstance(c, str) else c.text for c in chunks]
    long_enough = [t for t in texts if len(t.split()) >= 5]
    key = lambda t: hashlib.sha256(f"{K.PROMPT_VERSION}|{ex.name}|{t}".encode()).hexdigest()
    failed = [t for t in long_enough if key(t) not in cached]
    print(f"extractor {ex.name}: {len(long_enough)} passages, {len(failed)} not in the cache (= failed)")

    verdicts, examples = Counter(), {}
    tokens = []
    for t in failed[:args.sample]:
        try:
            r = ex.client.chat.completions.create(
                model=ex.model, temperature=0, max_tokens=args.max_tokens,
                messages=[{"role": "system", "content": K.SYSTEM_PROMPT}, {"role": "user", "content": f"Text:\n{t}"}],
                **K._extraction_request())
        except Exception as e:  # noqa: BLE001 - reported, not hidden
            verdicts["error"] += 1
            examples.setdefault("error", f"{type(e).__name__}: {str(e)[:150]}")
            continue
        out = r.choices[0].message.content or ""
        tokens.append(r.usage.completion_tokens)
        if r.choices[0].finish_reason == "length":
            v = "truncated"
        else:
            try:
                K.parse_extraction(out)
                v = "ok"
            except K.ExtractionError as e:
                v = "invalid"
                examples.setdefault("invalid", f"{str(e)[:160]} | reply ends: {out[-80:]!r}")
        verdicts[v] += 1
        examples.setdefault(v + "_words", f"passage {len(t.split())} words, reply {r.usage.completion_tokens} tokens")
    print(f"re-ran {sum(verdicts.values())} failed passages with max_tokens={args.max_tokens}: {dict(verdicts)}")
    if tokens:
        print(f"reply tokens: min {min(tokens)}, median {sorted(tokens)[len(tokens)//2]}, max {max(tokens)}")
    for k, v in examples.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
