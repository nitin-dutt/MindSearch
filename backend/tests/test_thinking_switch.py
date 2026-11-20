"""MODEL-01: GENERATOR_THINKING controls a "thinking" model's step-by-step reasoning (Qwen3.5 thinks by default). The default is
'off' (the default generator is Qwen3.5 since 2026-11-19); 'none' sends nothing, for a model without the option (Qwen2.5, the judge). 'off' turns thinking off on every call; 'on' lets ANSWERS think but keeps the short helper
calls (follow-up rewriting, the agent's search plan) off, because their small token budgets would be spent on thinking."""
import asyncio
from types import SimpleNamespace

import pytest

import llm
import settings as S


class _Recorder:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            async def gen():
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))])
            return gen()
        msg = SimpleNamespace(content="ok", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


@pytest.fixture
def rec(monkeypatch):
    r = _Recorder()
    monkeypatch.setattr(llm.client.chat.completions, "create", r.create)
    return r


def _run_all():
    async def go():
        async for _ in llm.stream_generate("m", "q", "ctx"):
            pass
        await llm.complete_chat("m", [{"role": "user", "content": "x"}])
        await llm.complete_tool_call("m", [{"role": "user", "content": "x"}], {"type": "function", "function": {"name": "search"}})
    asyncio.run(go())


def _thinking(call):
    return (call.get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking")


def test_none_sends_nothing_extra(rec, monkeypatch):
    monkeypatch.setenv("GENERATOR_THINKING", "none")
    _run_all()
    assert all("extra_body" not in c for c in rec.calls)


def test_the_default_is_off(rec, monkeypatch):
    monkeypatch.delenv("GENERATOR_THINKING", raising=False)
    _run_all()
    assert [_thinking(c) for c in rec.calls] == [False, False, False]


def test_off_turns_thinking_off_on_every_call(rec, monkeypatch):
    monkeypatch.setenv("GENERATOR_THINKING", "off")
    _run_all()
    assert [_thinking(c) for c in rec.calls] == [False, False, False]


def test_on_lets_only_the_answer_think(rec, monkeypatch):
    monkeypatch.setenv("GENERATOR_THINKING", "on")
    _run_all()
    assert [_thinking(c) for c in rec.calls] == [True, False, False]


def test_a_bad_value_fails_at_start_up():
    with pytest.raises(S.ConfigError):
        S.Settings.from_env({"GENERATOR_THINKING": "maybe"})



def test_the_graph_extractor_never_thinks(monkeypatch):
    """The vllm graph extractor calls the model directly (kg_extractors.py); a thinking model would spend its token budget thinking
    before the JSON. It sends enable_thinking=false unless GENERATOR_THINKING=none, alongside its JSON schema."""
    import kg_extractors as K
    sent = []

    class Fake:
        def create(self, **kw):
            sent.append(kw)
            msg = SimpleNamespace(content='{"triples": []}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    ex = K.VLLMExtractor.__new__(K.VLLMExtractor)
    ex.model, ex.truncated = "m", 0
    ex.client = SimpleNamespace(chat=SimpleNamespace(completions=Fake()))
    monkeypatch.delenv("GENERATOR_THINKING", raising=False)
    ex.extract("Alice met Bob in Paris.")
    monkeypatch.setenv("GENERATOR_THINKING", "none")
    ex.extract("Alice met Bob in Paris.")
    assert sent[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in sent[1].get("extra_body", {})


def test_the_graph_extractor_has_the_server_enforce_its_json_schema(monkeypatch):
    """Found 2026-11-19: vLLM 0.29 silently ignores the old `guided_json` field, so the schema was never enforced; Qwen2.5 happened to
    write bare JSON, Qwen3.5 wrapped it in a ```json fence and every extraction failed. The OpenAI-standard response_format json_schema
    is enforced (checked on the live server: bare, valid JSON)."""
    import kg_extractors as K
    sent = []

    class Fake:
        def create(self, **kw):
            sent.append(kw)
            msg = SimpleNamespace(content='{"triples": []}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    ex = K.VLLMExtractor.__new__(K.VLLMExtractor)
    ex.model, ex.truncated = "m", 0
    ex.client = SimpleNamespace(chat=SimpleNamespace(completions=Fake()))
    ex.extract("Alice met Bob in Paris.")
    rf = sent[0]["response_format"]
    expected = K.Extraction.model_json_schema()
    expected["properties"]["triples"]["maxItems"] = K.MAX_TRIPLES_PER_PASSAGE      # the cap against repetition loops
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"] == expected
    assert "guided_json" not in sent[0].get("extra_body", {})


def _fake_vllm_extractor(K, content, finish_reason="stop"):
    sent = []

    class Fake:
        def create(self, **kw):
            sent.append(kw)
            msg = SimpleNamespace(content=content)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)])

    ex = K.VLLMExtractor.__new__(K.VLLMExtractor)
    ex.model, ex.truncated = "m", 0
    ex.client = SimpleNamespace(chat=SimpleNamespace(completions=Fake()))
    return ex, sent


def test_the_extraction_reply_budget_is_a_setting_with_room_for_qwen35(monkeypatch):
    """Measured 2026-11-19 on the HotpotQA dev graph: 119 of Qwen3.5's 133 failed passages were cut off at the old 800-token reply
    limit (it extracts about 50% more facts than Qwen2.5). KG_EXTRACT_MAX_TOKENS sets the budget; the default leaves room."""
    import kg_extractors as K
    import settings as S
    assert S.Settings.from_env({}).kg_extract_max_tokens >= 2048
    monkeypatch.setenv("KG_EXTRACT_MAX_TOKENS", "3000")
    ex, sent = _fake_vllm_extractor(K, '{"triples": []}')
    ex.extract("Alice met Bob in Paris.")
    assert sent[0]["max_tokens"] == 3000
    with pytest.raises(S.ConfigError):
        S.Settings.from_env({"KG_EXTRACT_MAX_TOKENS": "0"})


def test_a_reply_cut_off_at_the_limit_fails_with_its_own_reason():
    """A truncated reply used to show up only as 'not valid JSON'. It now says what happened, so the cause is visible in the log."""
    import kg_extractors as K
    ex, _ = _fake_vllm_extractor(K, '{"triples": [{"subject": "Alice", "rel', finish_reason="length")
    with pytest.raises(K.ExtractionError, match="token limit"):
        ex.extract("Alice met Bob in Paris.")


def test_the_server_enforced_schema_caps_the_number_of_facts():
    """Found 2026-11-19: most extractions still failing at a 2,048-token budget were repetition loops (one passage: the same fact
    63 times). The schema sent to vLLM caps the list at MAX_TRIPLES_PER_PASSAGE, so the server closes the JSON instead of looping
    until the token limit. Measured Qwen3.5 output on 1,487 dev passages: p99 43 distinct facts, max 70, so 60 cuts off almost nothing.
    The cap is only in the request: replies parsed without server enforcement (Ollama) may still be longer."""
    import kg_extractors as K
    schema = K._extraction_request()["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["triples"]["maxItems"] == K.MAX_TRIPLES_PER_PASSAGE == 60
    long_reply = '{"triples": [' + ",".join(['{"subject": "A", "relation": "R", "object": "B"}'] * 70) + "]}"
    assert len(K.parse_extraction(long_reply)) == 70
