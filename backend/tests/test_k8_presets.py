"""GEN-72 (2026-11-19, under measurement): send the model 8 passages instead of 4 (production) or 3 (plain hybrid). Measured on the
HotpotQA dev set: answers containing the gold answer 0.685 -> 0.730 (paired +0.045, CI +0.000..+0.090). Behind PIPELINE_PRESET; the
default preset is unchanged."""
from dataclasses import asdict

import settings as S
from pipeline_config import PRESETS, get_preset


def _without(cfg, *keys):
    d = asdict(cfg)
    for k in ("name", *keys):
        d.pop(k)
    return d


def test_the_production_default_is_unchanged():
    assert S.Settings.from_env({}).pipeline_preset == "hybrid_graph_chunks"
    assert get_preset("hybrid_graph_chunks").final_k == 4


def test_the_k8_presets_differ_from_their_base_only_in_the_number_of_passages():
    for k8, base in (("hybrid_graph_chunks_k8", "hybrid_graph_chunks"), ("hybrid_rerank_k8", "hybrid_rerank")):
        assert PRESETS[k8].name == k8 and PRESETS[k8].final_k == 8
        assert _without(PRESETS[k8], "final_k") == _without(PRESETS[base], "final_k")
