"""Citations, 'I don't know' detection, prompt profiles and the refusal gate (GEN-01, GEN-02, GEN-03)."""
import pytest

import answers as A
from rag_pipeline import RAGPipeline, RetrievalResult
from pipeline_config import PipelineConfig


# ------------------------------------------------------------------ citations

def test_citations_are_read_in_every_common_shape():
    info = A.extract_citations("The tower opened in 1889 [1]. It is in Paris [2][3]; see also [1, 3].", n_sources=3)
    assert info["cited"] == [1, 2, 3] and info["invalid"] == [] and info["mentions"] == 5


def test_citations_to_passages_that_do_not_exist_are_reported_as_invalid():
    info = A.extract_citations("It happened in 1889 [1] and later [7] or [0].", n_sources=3)
    assert info["cited"] == [1] and info["invalid"] == [0, 7]


def test_no_citation_markers_means_nothing_cited():
    assert A.extract_citations("Plain answer, no markers. Array a[i] is not a citation.", 3)["cited"] == []


def test_mark_cited_flags_exactly_the_cited_sources():
    sources = [{"n": 1, "filename": "a"}, {"n": 2, "filename": "b"}, {"n": 3, "filename": "c"}]
    marked, info = A.mark_cited(sources, "Yes [2].")
    assert [s["cited"] for s in marked] == [False, True, False] and info["cited"] == [2]


# ------------------------------------------------------------------ abstention

@pytest.mark.parametrize("text", [A.ABSTAIN_MESSAGE, "I don’t know based on the documents available to you.",
                                  "I do not know.", "i don't know", "Sorry. " + A.ABSTAIN_MESSAGE])
def test_the_idk_sentence_is_recognised(text):
    assert A.is_abstention(text)


@pytest.mark.parametrize("text", ["The documents do not mention the price, but the deal closed in May [1].",
                                  "Paris [1].", "", "Nobody knows exactly, but the report says 1889 [2]."])
def test_a_real_answer_is_never_mistaken_for_idk(text):
    assert not A.is_abstention(text)


# ------------------------------------------------------------------ prompts

def test_the_two_prompt_profiles(monkeypatch):
    import llm
    monkeypatch.setenv("PROMPT_PROFILE", "enterprise")
    enterprise = llm.get_system_prompt()
    assert "SEVERAL documents" in enterprise and "Combine them" in enterprise        # multi-hop: do not refuse when the answer is spread out
    assert A.ABSTAIN_MESSAGE in enterprise and "[1]" in enterprise and 'id=\\"N\\"' not in enterprise and "1-2 sentences" not in enterprise
    monkeypatch.setenv("PROMPT_PROFILE", "benchmark")
    assert llm.get_system_prompt() == llm.BENCHMARK_PROMPT and "1-2 sentences" in llm.BENCHMARK_PROMPT   # the original, unchanged


def test_an_unknown_prompt_profile_fails_loudly(monkeypatch):
    import settings
    with pytest.raises(settings.ConfigError):
        settings.Settings.from_env({"PROMPT_PROFILE": "chatty"})


# ------------------------------------------------------------------ the refusal gate and numbered context

def result_with(*scores, key="score"):
    return RetrievalResult(chunks=[{"chunk": f"text {i}", "doc_id": "d", "chunk_index": i, key: s} for i, s in enumerate(scores)])


def gate(threshold):
    rag = RAGPipeline.__new__(RAGPipeline)           # the gate only needs the config
    rag.config = PipelineConfig(abstain_below=threshold)
    return rag.should_abstain


def test_the_gate_is_off_without_a_threshold():
    assert gate(None)(result_with(-9.0)) is False


def test_the_gate_refuses_only_when_the_best_passage_is_below_the_threshold():
    assert gate(0.0)(result_with(-3.2, -5.0)) is True
    assert gate(0.0)(result_with(2.1, -5.0)) is False
    assert gate(0.0)(RetrievalResult()) is False                 # nothing retrieved is handled by the empty-context path


def test_graph_ranked_results_are_judged_by_their_cross_encoder_score_not_the_ranker_score():
    res = RetrievalResult(chunks=[{"chunk": "t", "doc_id": "d", "chunk_index": 0, "score": 50.0, "ce": -4.0}])
    assert res.top_relevance() == -4.0 and gate(0.0)(res) is True


def test_context_numbers_match_the_sources_list():
    res = result_with(1.0, 0.5)
    assert res.to_context() == ['<doc id="1">text 0</doc>', '<doc id="2">text 1</doc>']
    assert [s["n"] for s in res.sources()] == [1, 2]


def test_only_the_production_presets_have_the_refusal_gate_on():
    from pipeline_config import PRESETS
    on = {name for name, cfg in PRESETS.items() if cfg.abstain_below is not None}
    assert on == {"hybrid_graph_chunks", "full_hybrid"} and PRESETS["hybrid_graph_chunks"].abstain_below == -9.0
    assert PRESETS["dense_only"].abstain_below is None and PRESETS["hybrid_rerank"].abstain_below is None   # ablations stay comparable


def test_the_date_line_is_added_only_when_the_caller_gives_a_date(monkeypatch):
    import datetime

    import llm
    monkeypatch.setenv("PROMPT_PROFILE", "enterprise")
    plain = llm.build_system_prompt()
    assert plain == llm.get_system_prompt()                                  # the benchmark runs never pass a date: prompts stay as measured
    dated = llm.build_system_prompt(datetime.date(2026, 10, 24))
    assert dated.startswith(plain) and "Saturday, 24 October 2026" in dated and "never assume a different year" in dated
    assert "Do not use LaTeX" in dated and "LaTeX" not in plain                  # the plain-text rule is live-chat only, like the date
