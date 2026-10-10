"""Extraction logic with fake backends: no Ollama, vLLM or Neo4j needed."""
import json

import pytest

from kg_extractors import (CachedExtractor, Extraction, ExtractionError, Triple, clean_triples,
                           extract_chunks, extract_with_retry, normalize_relation, parse_extraction)


class FakeExtractor:
    name = "fake:test"

    def __init__(self, table=None, fail_on=()):
        self.table, self.fail_on, self.calls, self.truncated = table or {}, set(fail_on), 0, 0

    def extract(self, chunk):
        self.calls += 1
        if chunk in self.fail_on:
            raise ExtractionError("boom")
        return [Triple(**t) for t in self.table.get(chunk, [])]


def T(s, r, o):
    return {"subject": s, "relation": r, "object": o}


def test_relation_normalization():
    assert normalize_relation("founded by") == "FOUNDED_BY"
    assert normalize_relation("  uses!! ") == "USES"
    assert normalize_relation("???") == "RELATED_TO"


def test_clean_triples_drops_junk_and_duplicates():
    raw = [Triple(**T("Apple ", "founded by", "Steve Jobs")),
           Triple(**T("apple", "FOUNDED_BY", "steve  jobs")),     # duplicate after normalization
           Triple(**T("Apple", "IS", "Apple")),                    # self reference
           Triple(**T("", "USES", "X")),                           # empty subject
           Triple(**T("A" * 200, "USES", "B"))]                    # oversized name
    out = clean_triples(raw)
    assert [(t.subject, t.relation, t.object) for t in out] == [("Apple", "FOUNDED_BY", "Steve Jobs")]


def test_parse_extraction_accepts_valid_and_rejects_prose():
    good = json.dumps({"triples": [T("A", "USES", "B")]})
    assert parse_extraction(good)[0].subject == "A"
    assert parse_extraction('{"triples": []}') == []
    # the real failure we saw: the model answered in prose instead of JSON
    with pytest.raises(ExtractionError):
        parse_extraction("It seems like you've provided a variety of disconnected paragraphs")


def test_schema_demands_the_three_fields():
    schema = Extraction.model_json_schema()
    assert set(schema["$defs"]["Triple"]["required"]) == {"subject", "relation", "object"}


def test_retry_recovers_from_one_failure_and_gives_up_after_two():
    class Flaky(FakeExtractor):
        def extract(self, chunk):
            self.calls += 1
            if self.calls == 1:
                raise ExtractionError("first attempt bad")
            return [Triple(**T("A", "USES", "B"))]

    assert extract_with_retry(Flaky(), "x", attempts=2)[0].object == "B"
    with pytest.raises(ExtractionError):
        extract_with_retry(FakeExtractor(fail_on=["x"]), "x", attempts=2)


def test_extract_chunks_counts_ok_failed_and_skipped_separately():
    ok, bad, short = "Marie Curie discovered radium in Paris.", "this one will fail every time here", "tiny"
    ex = FakeExtractor({ok: [T("Marie Curie", "DISCOVERED", "Radium")]}, fail_on=[bad])
    triples, stats = extract_chunks(ex, [ok, bad, short], workers=2)
    assert (stats.chunks_ok, stats.chunks_failed, stats.chunks_skipped_short) == (1, 1, 1)
    assert stats.failure_rate == 0.5
    assert [(t.subject, t.object) for t in triples] == [("Marie Curie", "Radium")]
    assert stats.triples_kept == 1 and stats.extractor == "fake:test"


def test_an_all_failing_run_reports_total_failure_instead_of_hiding_it():
    chunks = [f"chunk number {i} has enough words" for i in range(4)]
    _, stats = extract_chunks(FakeExtractor(fail_on=chunks), chunks)
    assert stats.chunks_failed == 4 and stats.failure_rate == 1.0 and stats.triples_kept == 0


def test_cache_makes_reruns_free_and_survives_restart(tmp_path):
    path = str(tmp_path / "kg.jsonl")
    chunk = "Marie Curie discovered radium in Paris."
    inner = FakeExtractor({chunk: [T("Marie Curie", "DISCOVERED", "Radium")]})
    first = CachedExtractor(inner, path)
    assert first.extract(chunk)[0].object == "Radium" and inner.calls == 1

    again = CachedExtractor(inner, path)          # new object, same file = "process restarted"
    assert again.extract(chunk)[0].object == "Radium"
    assert inner.calls == 1 and again.hits == 1   # no second model call


def test_cache_never_stores_failures(tmp_path):
    path = str(tmp_path / "kg.jsonl")
    cached = CachedExtractor(FakeExtractor(fail_on=["bad chunk"]), path)
    with pytest.raises(ExtractionError):
        cached.extract("bad chunk")
    assert not (tmp_path / "kg.jsonl").exists()
