"""Multi-turn evaluation (alignment_plan CONV-03, used by CONV-04 and CONV-06). Judge-free.

    python -m evaluation.prepare_conversations                      # once: writes eval_data/conv_dev_conversations.json
    python -m evaluation.conversation_eval --retrieval-only          # every strategy, searches only (fast)
    python -m evaluation.conversation_eval --strategies needed agent # full turns, with the model's answer

Each chat goes through the same code as a real chat request (`chat.ChatService.begin`, then the turn's events), so the
rewrite, the retry with a forced rewrite, the agent's search plan, the refusal gate and the model's own "I don't know"
are all measured. Strategies:
  standalone       the full question is typed (no history): the upper bound a perfect rewriter would reach
  raw              the last message is searched as typed, history ignored for searching (no rewriter)
  always           every last message is rewritten from the history
  needed_noretry   rewrite only when the message points back at the chat (the current rule) and no retry
  needed           the current default: rewrite when needed, plus one forced-rewrite retry when nothing usable is found
  agent            CONV-06: the model plans 1-3 searches with the search_documents tool (a failed plan is an error)

Measures, per strategy and per kind of chat (the kinds are described in prepare_conversations.py):
  gold@k          the gold passage is among the first k passages the answer would be written from (answerable chats)
  right_outcome   the reply refuses exactly when it should: answer for answerable chats, "I don't know" for unanswerable ones
                  (--retrieval-only: decided by the refusal gate alone, without the model)
  latency         seconds until the turn was decided (searches, rewrites, plan); with generation also the whole reply
  agent           how often the plan was used vs fell back (timeout, bad arguments, model error)
Paired bootstrap CIs compare every strategy with `needed`. Run inside the worker container (it needs the vLLM server).
"""
import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone

from evaluation import metrics as M
from evaluation.datasets import load_dataset
from evaluation.generate import RESULTS_DIR, build_pipeline
from evaluation.prepare_conversations import OUT as CONVERSATIONS
from pipeline_config import get_preset

STRATEGIES = ("standalone", "raw", "always", "needed_noretry", "needed", "agent")
CATEGORIES = ("pronoun", "topic_switch", "vague", "refusal_mid", "unanswerable")


class RecordingAudit:
    """Stands in for the audit log: keeps the event names so the agent's used / failed plans can be counted."""

    def __init__(self):
        self.events = []

    def log(self, event, **fields):
        self.events.append((event, fields))


class AlwaysRewrite:
    """Every message is rewritten, as if each one pointed back at the chat."""

    def __init__(self, rewriter):
        self.inner = rewriter

    async def rewrite(self, history, question, force=False):
        return await self.inner.rewrite(history, question, force=True)


class NoRetry:
    """The current rule without the second chance: a forced rewrite (only the retry asks for one) changes nothing."""

    def __init__(self, rewriter):
        self.inner = rewriter

    async def rewrite(self, history, question, force=False):
        if force:
            return question, False
        return await self.inner.rewrite(history, question)


def make_service(strategy, rag, model, audit):
    import agent
    from chat import ChatService
    from conversation import QueryRewriter
    from llm import complete_chat, complete_tool_call
    from settings import get_settings
    cfg = get_settings()
    rewriter = QueryRewriter(model, complete_chat, enabled=True, timeout=cfg.rewrite_timeout_s)
    planner = None
    if strategy in ("raw", "standalone"):
        rewriter = None
    elif strategy == "always":
        rewriter = AlwaysRewrite(rewriter)
    elif strategy == "needed_noretry":
        rewriter = NoRetry(rewriter)
    elif strategy == "agent":
        planner = agent.SearchPlanner(model, complete_tool_call, enabled=True, timeout=cfg.agent_timeout_s,
                                      max_queries=cfg.agent_max_queries)
    return ChatService(rag, audit, planner=planner, rewriter=rewriter, workers=1)


async def one_chat(service, rag, chat, strategy, generate, user):
    from answers import is_abstention
    history, question = chat["history"], chat["question"]
    if strategy == "standalone":
        history, question = [], chat["standalone"]
    n_audit = len(service.audit.events)
    t0 = time.perf_counter()
    row = {"id": chat["id"], "category": chat["category"], "strategy": strategy, "expect": chat["expect"]}
    if generate:
        try:
            turn = await service.begin(history, question, user)
        except Exception as e:  # noqa: BLE001 - a model failure is counted, not fatal
            return {**row, "error": type(e).__name__}
        decided = time.perf_counter() - t0
        outcome, text = turn.outcome, []
        async for ev in turn.events():
            if ev["type"] == "outcome":
                outcome = ev["outcome"]
            elif ev["type"] == "token":
                text.append(ev["text"])
            elif ev["type"] == "error":
                outcome = "error"
        answer = "".join(text)
        refused = outcome in ("refused", "none_found") or is_abstention(answer)
        row.update(answer=answer, seconds_total=time.perf_counter() - t0)
    else:
        from conversation import trim_history
        query, rewritten, searches, result, _, used = await service._retrieve(trim_history(history), question, user, None)
        decided = time.perf_counter() - t0
        turn = type("T", (), {"query": query, "rewritten": rewritten, "searches": searches, "result": result})
        refused = not result.to_context() or bool(rag.should_abstain(result))
    chunks = [c["chunk"] for c in turn.result.chunks]
    row.update(query=turn.query, rewritten=turn.rewritten, searches=turn.searches, seconds_decided=decided, refused=refused,
               right_outcome=float(refused == (chat["expect"] == "refuse")))
    if chat["expect"] == "answer":
        cover = M.coverage(chunks, chat["gold_passages"])
        row.update({f"gold@{k}": float(M.hit_at_k(cover, k)) for k in (1, 3, 5)})
    events = [e for e, _ in service.audit.events[n_audit:]]
    if strategy == "agent":
        row["plan"] = "used" if "tool_call" in events else ("failed" if "tool_call_failed" in events else "off")
    return row


async def run(args):
    with open(CONVERSATIONS, encoding="utf-8") as f:
        data = json.load(f)
    chats = data["conversations"]
    if args.limit:
        chats = chats[:args.limit]
    corpus, _ = load_dataset(data["meta"]["corpus"], 1)
    cfg = get_preset(args.config)
    if cfg.use_graph:
        # The graph lives in the shared Neo4j database: an evaluation graph would have to replace the user's own one.
        raise SystemExit(f"{args.config} uses the knowledge graph; use a preset without it (e.g. hybrid_rerank).")
    rag, _, _ = build_pipeline(data["meta"]["corpus"], corpus, cfg)
    user = rag.policy.superuser()

    rows = []
    for strategy in args.strategies:
        service = make_service(strategy, rag, cfg.generator_model, RecordingAudit())
        t0 = time.time()
        for n, chat in enumerate(chats, start=1):
            rows.append(await one_chat(service, rag, chat, strategy, not args.retrieval_only, user))
            if n % 25 == 0:
                print(f"  {strategy}: {n}/{len(chats)} ({time.time() - t0:.0f}s)")

    keys = ["gold@1", "gold@3", "gold@5", "right_outcome", "seconds_decided"] + ([] if args.retrieval_only else ["seconds_total"])
    summary = {"n_chats": len(chats), "config": args.config, "retrieval_only": args.retrieval_only, "strategies": {}, "paired": {}}
    for s in args.strategies:
        mine = [r for r in rows if r["strategy"] == s and "error" not in r]
        entry = {"all": M.summarize(mine, keys), "errors": sum(1 for r in rows if r["strategy"] == s and "error" in r),
                 "rewritten": sum(bool(r.get("rewritten")) for r in mine) / max(1, len(mine)),
                 "p95_seconds_decided": M.percentile([r["seconds_decided"] for r in mine], 0.95)}
        entry["by_category"] = {c: M.summarize([r for r in mine if r["category"] == c], ["gold@3", "right_outcome"])
                                for c in CATEGORIES}
        if s == "agent":
            entry["plans"] = {k: sum(r.get("plan") == k for r in mine) for k in ("used", "failed", "off")}
        summary["strategies"][s] = entry
    if "needed" in args.strategies:
        base = {r["id"]: r for r in rows if r["strategy"] == "needed"}
        for s in args.strategies:
            if s == "needed":
                continue
            other = {r["id"]: r for r in rows if r["strategy"] == s}
            for k in ("gold@3", "right_outcome"):
                ids = [i for i in base if i in other]
                d, lo, hi, n = M.paired_bootstrap_diff([other[i].get(k) for i in ids], [base[i].get(k) for i in ids])
                summary["paired"][f"{s} minus needed: {k}"] = {"diff": d, "ci_low": lo, "ci_high": hi, "n": n}

    tag = "retrieval" if args.retrieval_only else "full"
    run_dir = os.path.join(RESULTS_DIR, f"conv__{data['meta']['name']}__{args.config}__{tag}__"
                                        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "chats.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== multi-turn, {len(chats)} chats, config {args.config}, {tag} (mean [95% CI]) ===")
    for s, e in summary["strategies"].items():
        a = e["all"]
        line = "  ".join(f"{k} {a[k]['mean']:.3f}" for k in ("gold@3", "right_outcome") if a[k]["mean"] is not None)
        print(f"  {s:<15} {line}  decided in {a['seconds_decided']['mean']:.2f}s (p95 {e['p95_seconds_decided']:.2f}s)"
              f"  rewritten {e['rewritten']:.0%}  errors {e['errors']}" + (f"  plans {e['plans']}" if "plans" in e else ""))
        print("      by kind (gold@3 / right outcome): " + "  ".join(
            f"{c} {e['by_category'][c]['gold@3']['mean'] if e['by_category'][c]['gold@3']['mean'] is not None else float('nan'):.2f}"
            f"/{e['by_category'][c]['right_outcome']['mean']:.2f}" for c in CATEGORIES if e['by_category'][c]['right_outcome']['n']))
    for key, v in summary["paired"].items():
        sig = " *" if v["ci_low"] is not None and (v["ci_low"] > 0 or v["ci_high"] < 0) else ""
        print(f"  {key:<40} {v['diff']:+.3f} ({v['ci_low']:+.3f} to {v['ci_high']:+.3f}){sig}")
    print(f"\nSaved {run_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    ap.add_argument("--config", default="hybrid_rerank", help="pipeline preset without the graph (default hybrid_rerank, as in the squad_dev_unans gate runs)")
    ap.add_argument("--retrieval-only", action="store_true", help="stop after the search: no answer is generated")
    ap.add_argument("--limit", type=int, help="first N chats only (smoke test)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
