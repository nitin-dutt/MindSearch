"""PERF-03: where and how the cross-encoder runs is a switch (RERANKER_BACKEND / RERANKER_DEVICE); the default is today's PyTorch on
the CPU, and a backend or device that cannot be used stops start-up with a clear error (no silent fall back to the CPU)."""
import builtins
import importlib.util

import pytest

import reranker as R
import settings as S


def test_the_default_is_pytorch_on_the_cpu():
    cfg = S.Settings.from_env({})
    assert (cfg.reranker_backend, cfg.reranker_device) == ("torch", "cpu")
    assert cfg.reranker_onnx_file == "onnx/model_qint8_avx512_vnni.onnx"


@pytest.mark.parametrize("env", [{"RERANKER_BACKEND": "tensorrt"}, {"RERANKER_DEVICE": "gpu"},
                                 {"RERANKER_BACKEND": "onnx", "RERANKER_DEVICE": "cuda"}])
def test_unknown_or_impossible_combinations_fail_at_start_up(env):
    with pytest.raises(S.ConfigError):
        S.Settings.from_env(env)


def test_asking_for_the_gpu_without_one_is_an_error_not_a_cpu_fallback(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    built = []
    monkeypatch.setattr(R, "CrossEncoder", lambda *a, **k: built.append(k) or object())
    with pytest.raises(R.RerankerUnavailable, match="RERANKER_DEVICE=cuda"):
        R.ReRanker(backend="torch", device="cuda")
    assert built == []                                       # nothing was loaded on the CPU instead


def test_the_onnx_backend_without_onnxruntime_is_an_error(monkeypatch):
    real_import = builtins.__import__

    def no_onnxruntime(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("No module named 'onnxruntime'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_onnxruntime)
    with pytest.raises(R.RerankerUnavailable, match="onnxruntime"):
        R.ReRanker(backend="onnx", device="cpu")


def test_rerank_sorts_by_the_backend_scores_whatever_the_backend():
    class Fixed:
        def predict(self, pairs):
            return [len(c) for _, c in pairs]                 # longer chunk = higher score

    rr = R.ReRanker.__new__(R.ReRanker)
    rr.model = Fixed()
    out = rr.rerank("q", ["aa", "a", "aaa"], top_k=2)
    assert [o["chunk"] for o in out] == ["aaa", "aa"] and [o["original_index"] for o in out] == [2, 0]


@pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None, reason="onnxruntime not installed (PERF-03 experiment only)")
def test_the_int8_onnx_model_ranks_like_the_pytorch_model():
    """Real models (downloads the 23 MB int8 file once). Scores may shift a little; the ORDER must match on clear cases."""
    q = "What is the capital of France?"
    chunks = ["Paris is the capital and largest city of France.", "The Seine flows through Paris.",
              "Bananas are rich in potassium.", "France is a country in Western Europe."]
    torch_order = [r["original_index"] for r in R.ReRanker(backend="torch", device="cpu").rerank(q, chunks, top_k=4)]
    onnx_scored = R.ReRanker(backend="onnx", device="cpu").rerank(q, chunks, top_k=4)
    assert [r["original_index"] for r in onnx_scored] == torch_order
    assert onnx_scored[0]["score"] > 0 > onnx_scored[-1]["score"]       # raw logits, like the PyTorch model (abstain gate uses them)


class _RecordingModel:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def predict(self, pairs):
        if self.fail:
            raise RuntimeError("CUDA error: out of memory")
        self.calls.append(pairs)
        return [0.0] * len(pairs)


def _fake_gpu(monkeypatch, model):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(R, "CrossEncoder", lambda *a, **k: model)


def test_the_gpu_reranker_is_warmed_up_while_the_api_starts(monkeypatch):
    """The first GPU call after a restart loads the CUDA kernels (about 15 s measured on 2026-11-18). Doing it at start-up means
    the first user does not pay for it. The warm-up scores a batch shaped like a real search (pool of 20, passage-length text)."""
    model = _RecordingModel()
    _fake_gpu(monkeypatch, model)
    R.ReRanker(backend="torch", device="cuda")
    assert len(model.calls) == 1 and len(model.calls[0]) == R.WARMUP_PAIRS == 20
    assert all(len(passage.split()) >= 100 for _, passage in model.calls[0])


def test_the_cpu_reranker_starts_exactly_as_before_without_a_warm_up(monkeypatch):
    model = _RecordingModel()
    monkeypatch.setattr(R, "CrossEncoder", lambda *a, **k: model)
    R.ReRanker(backend="torch", device="cpu")
    assert model.calls == []


def test_a_failed_gpu_warm_up_stops_start_up(monkeypatch):
    """No fallback: if the GPU cannot score at start-up (for example out of memory next to vLLM), the API must not start."""
    _fake_gpu(monkeypatch, _RecordingModel(fail=True))
    with pytest.raises(R.RerankerUnavailable, match="warm-up"):
        R.ReRanker(backend="torch", device="cuda")
