"""Search as a tool (alignment_plan CONV-06): validating the model's search plan, merging several searches, and the chat endpoint
with CHAT_STRATEGY=agent. The model is a fake that returns prepared tool arguments, so no language model is needed."""
import asyncio
import json

import pytest

import agent
from agent import PlanOutcome, SearchPlan, SearchPlanner, parse_plan

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from audit import AuditLog  # noqa: E402
from ratelimit import RateLimiter  # noqa: E402
from test_api import SECRET, StubRewriter, convo, env  # noqa: E402,F401  (env is a fixture)

QUESTION = "compare the refund policy with the warranty terms"


def args(**kwargs):
    return json.dumps(kwargs)


# ------------------------------------------------------------------ the tool arguments are untrusted text: validation

def test_a_valid_plan_is_accepted_trimmed_deduplicated_and_capped():
    plan = parse_plan(args(question="  compare refunds   and warranty ", queries=["refund policy", " Refund  Policy ", "warranty terms",
                                                                                  "returns", "shipping"]), QUESTION, max_queries=3)
    assert plan == SearchPlan(question="compare refunds and warranty", queries=["refund policy", "warranty terms", "returns"])


@pytest.mark.parametrize("raw", ["", "not json", "[1, 2]", '{"question": "q"}', '{"queries": "refund policy"}', '{"queries": []}',
                                 '{"queries": ["", "   "]}', '{"queries": [1, null, {"a": 1}]}',
                                 json.dumps({"queries": ["x" * (agent.MAX_QUERY_CHARS + 1)]})])
def test_unusable_arguments_give_no_plan(raw):
    assert parse_plan(raw, QUESTION, max_queries=3) is None


def test_a_missing_or_absurd_question_makes_the_plan_unusable():
    """No quiet substitute: a plan without a usable standalone question is not a plan (the turn then fails visibly)."""
    assert parse_plan(args(queries=["refund policy"]), QUESTION, 3) is None
    assert parse_plan(args(question="x" * 5000, queries=["refund policy"]), QUESTION, 3) is None
    assert parse_plan(args(question=["not", "text"], queries=["refund policy"]), QUESTION, 3) is None


def test_extra_arguments_cannot_smuggle_in_a_user_or_a_scope():
    """The plan object has only search words. Whatever else the model writes (a user, a level, document ids) is dropped here,
    and the API never reads identity from the plan."""
    plan = parse_plan(args(question="q?", queries=["salaries"], user="root", level="Executive", doc_ids=["a" * 32], is_super=True),
                      QUESTION, 3)
    assert plan == SearchPlan(question="q?", queries=["salaries"])
    assert set(vars(plan)) == {"question", "queries"}


# ------------------------------------------------------------------ the planner never raises

def planner_with(call, **kwargs):
    return SearchPlanner("model", call, **kwargs)


def test_the_planner_sends_the_tool_the_chat_and_the_question_and_returns_the_plan():
    seen = {}

    async def call(model, messages, tool, max_tokens=250):
        seen.update(model=model, messages=messages, tool=tool)
        return args(question="What is the price of the Contoso acquisition?", queries=["Contoso acquisition price"])

    history = convo(("user", "When will the Contoso acquisition be announced?"), ("assistant", "On the first of March."))
    outcome = asyncio.run(planner_with(call).plan(history, "what about its price?"))
    assert outcome == PlanOutcome(plan=SearchPlan("What is the price of the Contoso acquisition?", ["Contoso acquisition price"]))
    assert seen["tool"]["function"]["name"] == "search_documents"
    assert seen["tool"]["function"]["parameters"]["properties"]["queries"]["maxItems"] == 3
    user_message = seen["messages"][-1]["content"]
    assert "User: When will the Contoso acquisition be announced?" in user_message and "Latest message: what about its price?" in user_message


def test_a_timeout_a_model_error_and_bad_arguments_are_reported_not_raised():
    async def slow(*a, **k):
        await asyncio.sleep(1)

    async def down(*a, **k):
        raise RuntimeError("connection refused")

    async def prose(*a, **k):
        return "I think you should search for refunds."

    assert asyncio.run(planner_with(slow, timeout=0.01).plan([], QUESTION)) == PlanOutcome(error="timeout")
    assert asyncio.run(planner_with(down).plan([], QUESTION)) == PlanOutcome(error="model_error")
    assert asyncio.run(planner_with(prose).plan([], QUESTION)) == PlanOutcome(error="invalid_arguments")


# ------------------------------------------------------------------ merging several searches

def test_merging_interleaves_the_searches_and_removes_duplicates():
    pytest.importorskip("sentence_transformers")
    from rag_pipeline import RetrievalResult

    def c(doc, i, ce):
        return {"chunk": f"{doc}{i}", "doc_id": doc, "chunk_index": i, "ce": ce, "score": ce}

    refunds = RetrievalResult(chunks=[c("r", 0, 2.0), c("r", 1, 1.0), c("s", 9, 0.5)], timings_ms={"dense_and_rerank": 10.0})
    warranty = RetrievalResult(chunks=[c("w", 0, 5.0), c("s", 9, 4.0), c("w", 2, 3.0)], timings_ms={"dense_and_rerank": 20.0})
    merged = RetrievalResult.merge([refunds, warranty], max_passages=4)
    # the search with the best top passage goes first; then one passage from each in turn; the shared passage appears once
    assert [x["chunk"] for x in merged.chunks] == ["w0", "r0", "s9", "r1"]
    assert merged.top_relevance() == 5.0 and merged.timings_ms == {"dense_and_rerank": 30.0}
    assert [s["n"] for s in merged.sources()] == [1, 2, 3, 4]                       # numbered for citations like any result
    assert RetrievalResult.merge([refunds], max_passages=1) is refunds              # a single search is returned untouched
    assert RetrievalResult.merge([RetrievalResult(), RetrievalResult()], 4).chunks == []


def test_the_merged_result_is_refused_only_if_every_search_would_be():
    pytest.importorskip("sentence_transformers")
    from rag_pipeline import RAGPipeline, RetrievalResult
    from pipeline_config import PipelineConfig

    rag = RAGPipeline.__new__(RAGPipeline)                                          # only the gate is used: no models, no stores
    rag.config = PipelineConfig(abstain_below=-9.0)
    weak = RetrievalResult(chunks=[{"chunk": "a", "doc_id": "a", "chunk_index": 0, "score": -11.0}])
    strong = RetrievalResult(chunks=[{"chunk": "b", "doc_id": "b", "chunk_index": 0, "score": 3.0}])
    assert rag.should_abstain(rag.merge_results([weak, strong], 6)) is False
    assert rag.should_abstain(rag.merge_results([weak, weak], 6)) is True


# ------------------------------------------------------------------ the chat endpoint with the agent strategy

class FakePlanner:
    enabled = True

    def __init__(self, outcome):
        self.outcome, self.calls = outcome, []

    async def plan(self, history, question):
        self.calls.append((history, question))
        return self.outcome


def agent_app(env, planner, rewriter=None):  # noqa: F811
    app = api.create_app(env.rag, env.store, SECRET, upload_dir=env.upload_dir, audit=AuditLog(env.audit_path),
                         login_limiter=RateLimiter(max_attempts=100), rewriter=rewriter or StubRewriter(), planner=planner)
    client = TestClient(app)
    tok = client.post("/v1/auth/login", json={"username": "staff", "password": "pw-staff"}).json()["access_token"]
    return client, {"Authorization": f"Bearer {tok}"}


def events_of(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


def audit_rows(env):  # noqa: F811
    return [json.loads(line) for line in open(env.audit_path, encoding="utf-8")]


def test_every_planned_query_is_searched_as_the_caller_and_the_user_sees_what_was_searched(env):  # noqa: F811
    planner = FakePlanner(PlanOutcome(plan=SearchPlan(QUESTION, ["refund policy", "warranty terms"])))
    rewriter = StubRewriter()
    client, h = agent_app(env, planner, rewriter)
    r = client.post("/v1/chat/completions", headers=h, json={"messages": convo(("user", QUESTION))})
    assert r.status_code == 200 and "Hello" in r.text
    assert sorted(env.rag.queries) == ["refund policy", "warranty terms"]
    assert {u.id for u in env.rag.asked_as} == {"staff"}                    # identity comes from the token, for every search
    assert rewriter.calls == []                                             # the plan replaced the rewrite step
    events = events_of(r)
    assert {"type": "searches", "queries": ["refund policy", "warranty terms"]} in events
    assert not any(e["type"] == "standalone" for e in events)              # the question itself was not changed


def test_a_follow_up_is_answered_as_the_standalone_question_from_the_plan(env):  # noqa: F811
    planner = FakePlanner(PlanOutcome(plan=SearchPlan("What is the price of the Contoso acquisition?", ["Contoso acquisition price"])))
    client, h = agent_app(env, planner)
    history = convo(("user", "When will the Contoso acquisition be announced?"), ("assistant", "On the first of March."),
                    ("user", "what about its price?"))
    r = client.post("/v1/chat/completions", headers=h, json={"messages": history})
    events = events_of(r)
    assert events[0] == {"type": "standalone", "question": "What is the price of the Contoso acquisition?"}
    assert not any(e["type"] == "searches" for e in events)                         # one search: nothing extra to show
    assert env.rag.queries == ["Contoso acquisition price"]
    assert planner.calls[0][1] == "what about its price?" and len(planner.calls[0][0]) == 2


def test_the_file_scope_chosen_by_the_user_applies_to_every_planned_search(env):  # noqa: F811
    planner = FakePlanner(PlanOutcome(plan=SearchPlan(QUESTION, ["refund policy", "warranty terms"])))
    client, h = agent_app(env, planner)
    client.post("/v1/chat/completions", headers=h, json={"messages": convo(("user", QUESTION)), "doc_ids": ["a" * 32]})
    assert env.rag.scopes == [["a" * 32], ["a" * 32]]


def test_tool_calls_are_audited_without_the_query_text(env):  # noqa: F811
    planner = FakePlanner(PlanOutcome(plan=SearchPlan(QUESTION, ["refund policy", "warranty terms"])))
    client, h = agent_app(env, planner)
    client.post("/v1/chat/completions", headers=h, json={"messages": convo(("user", QUESTION))})
    rows = audit_rows(env)
    call = next(r for r in rows if r["event"] == "tool_call")
    assert call["user"] == "staff" and call["tool"] == "search_documents" and len(call["queries"]) == 2
    chat = next(r for r in rows if r["event"] == "chat")
    assert chat["strategy"] == "agent" and chat["searches"] == 2
    raw = open(env.audit_path, encoding="utf-8").read()
    assert "refund policy" not in raw and "warranty" not in raw             # fingerprints only, like the question


@pytest.mark.parametrize("error", ["timeout", "model_error", "invalid_arguments"])
def test_a_failed_plan_fails_the_turn_visibly_and_never_falls_back_to_rewriting(env, error):  # noqa: F811
    """A quiet fallback to the rewrite strategy once hid that the model server had tool calling switched off (2026-11-17)."""
    rewriter = StubRewriter()
    client, h = agent_app(env, FakePlanner(PlanOutcome(error=error)), rewriter)
    r = client.post("/v1/chat/completions", headers=h, json={"messages": convo(("user", QUESTION))})
    assert r.status_code == 503 and "Hello" not in r.text
    assert env.rag.queries == [] and rewriter.calls == []                   # nothing searched, no rewrite tried instead
    rows = audit_rows(env)
    assert any(r["event"] == "tool_call_failed" and r["reason"] == error for r in rows)
    assert any(r["event"] == "chat_failed" and r.get("stage") == "search" for r in rows)


def test_startup_refuses_the_agent_strategy_when_tool_calling_does_not_work():
    from agent import SearchPlanner, check_tool_calling

    async def no_tools(*a, **k):
        raise RuntimeError("400: tool_choice requires --tool-call-parser")

    with pytest.raises(RuntimeError, match="tool-call-parser"):
        check_tool_calling(SearchPlanner("m", no_tools))

    async def works(*a, **k):
        return '{"question": "When was the refund policy last updated?", "queries": ["refund policy update date"]}'

    check_tool_calling(SearchPlanner("m", works))                           # a working model passes silently


def test_when_no_planned_search_finds_anything_the_no_context_answer_is_given(env):  # noqa: F811
    env.rag.empty_queries = {"refund policy", "warranty terms"}
    client, h = agent_app(env, FakePlanner(PlanOutcome(plan=SearchPlan(QUESTION, ["refund policy", "warranty terms"]))))
    r = client.post("/v1/chat/completions", headers=h, json={"messages": convo(("user", QUESTION))})
    assert api.NO_CONTEXT_MESSAGE in r.text and "Hello" not in r.text


def test_the_agent_strategy_is_off_by_default(env):  # noqa: F811
    """Without CHAT_STRATEGY=agent the app builds a disabled planner: behaviour is exactly the rewrite strategy."""
    r = env.client.post("/v1/chat/completions", headers=env.login("staff"), json={"messages": convo(("user", QUESTION))})
    assert r.status_code == 200 and env.rag.queries == [QUESTION]
    assert not any(row["event"].startswith("tool_call") for row in audit_rows(env))


def test_the_model_server_is_started_with_tool_calling_switched_on():
    """Found 2026-11-17 on the real stack: without these flags vLLM answers every named tool call with HTTP 400, so the agent step
    always fell back to rewriting, while every test with a fake model passed."""
    import os
    compose = os.path.join(os.path.dirname(__file__), "..", "..", "docker-compose.yml")
    if not os.path.exists(compose):
        pytest.skip("docker-compose.yml is not in this checkout")
    text = open(compose, encoding="utf-8").read()
    assert "--enable-auto-tool-choice" in text and "--tool-call-parser" in text
