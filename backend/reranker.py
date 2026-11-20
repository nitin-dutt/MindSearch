"""The cross-encoder re-ranker: reads the question and each passage together and scores how well the passage answers it.

Where and how it runs is a switch (alignment_plan PERF-03), default = PyTorch on the CPU, exactly as before:
  RERANKER_BACKEND=torch  RERANKER_DEVICE=cpu    sentence-transformers CrossEncoder on the CPU (default)
  RERANKER_BACKEND=torch  RERANKER_DEVICE=cuda   the same model on the GPU (needs a CUDA PyTorch build and a visible GPU)
  RERANKER_BACKEND=onnx   RERANKER_DEVICE=cpu    onnxruntime with the model repo's ready-made int8 file (RERANKER_ONNX_FILE)
A backend or device that cannot be used raises RerankerUnavailable when the pipeline is built, so the API does not start; it never
falls back to another backend silently. Every backend returns the model's raw scores (logits), which the 'I don't know' gate uses.
"""
import os

from sentence_transformers import CrossEncoder

from settings import get_settings

RERANKER_MODEL_NAME = get_settings().reranker_model     # RERANKER_MODEL (settings.py)


class RerankerUnavailable(RuntimeError):
    """The configured re-ranker backend or device cannot be used on this machine."""


# GPU warm-up (PERF-03): the first GPU call after a restart loads the CUDA kernels, which took about 15 s on 2026-11-18. Scoring one
# batch shaped like a real search (a pool of 20 passages of about chunk length) at start-up moves that cost off the first user.
WARMUP_PAIRS = 20
_WARMUP_QUESTION = "Which country did the travel policy say employees must book flights through?"
_WARMUP_PASSAGE = " ".join(["The policy describes how employees request, approve and book business travel, and which receipts"
                            " finance needs before a claim is paid."] * 6)


def _cpu_threads():
    """Same share of the CPU the PyTorch path gets (api._set_torch_threads): cores / concurrent retrievals, or TORCH_THREADS."""
    cfg = get_settings()
    return cfg.torch_threads or max(1, (os.cpu_count() or 4) // cfg.retrieval_workers)


class OnnxCrossEncoder:
    """A cross-encoder exported to ONNX, run with onnxruntime. Same `predict(pairs)` contract as CrossEncoder: one raw score per
    (question, passage) pair, the pair truncated to the model's 512 tokens. Thread-safe: onnxruntime sessions allow concurrent runs."""

    def __init__(self, model_name, onnx_file, threads, batch_size=32):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RerankerUnavailable("RERANKER_BACKEND=onnx needs the onnxruntime package, which is not installed") from e
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = min(512, self.tokenizer.model_max_length)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        path = hf_hub_download(model_name, onnx_file)
        self.session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.batch_size = batch_size

    def predict(self, pairs):
        import numpy as np
        scores = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            enc = self.tokenizer([q for q, _ in batch], [c for _, c in batch], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="np")
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            logits = self.session.run(None, feed)[0]
            scores.extend(logits.reshape(len(batch), -1)[:, 0].tolist())
        return np.asarray(scores, dtype=np.float32)


class ReRanker:
    def __init__(self, model_name=RERANKER_MODEL_NAME, backend=None, device=None):
        cfg = get_settings()
        backend = backend or cfg.reranker_backend
        device = device or cfg.reranker_device
        self.backend, self.device = backend, device
        if backend == "onnx":
            if device != "cpu":
                raise RerankerUnavailable("RERANKER_BACKEND=onnx runs on the CPU only; use RERANKER_DEVICE=cpu")
            self.model = OnnxCrossEncoder(model_name, cfg.reranker_onnx_file, threads=_cpu_threads())
        elif backend == "torch":
            if device == "cuda":
                import torch
                if not torch.cuda.is_available():
                    raise RerankerUnavailable(
                        f"RERANKER_DEVICE=cuda but PyTorch sees no GPU (torch {torch.__version__}, CUDA build: {torch.version.cuda}). "
                        "Build the API image with a CUDA PyTorch and give the container the GPU, or set RERANKER_DEVICE=cpu.")
            self.model = CrossEncoder(model_name, device=device)
            if device == "cuda":
                self._warm_up()
        else:
            raise RerankerUnavailable(f"unknown RERANKER_BACKEND '{backend}'")

    def _warm_up(self):
        """Score one realistic batch on the GPU now. A failure here (for example out of memory next to vLLM) stops start-up."""
        try:
            self.model.predict([[_WARMUP_QUESTION, _WARMUP_PASSAGE]] * WARMUP_PAIRS)
        except Exception as e:
            raise RerankerUnavailable(f"RERANKER_DEVICE=cuda: the GPU warm-up failed ({e}). Is vLLM leaving enough GPU memory?") from e

    def rerank(self, query, chunks, top_k=5):
        """Score every chunk against `query`; return the best `top_k` as {'chunk', 'score', 'original_index'}, best first."""
        if not chunks:
            return []
        scores = self.model.predict([[query, chunk] for chunk in chunks])
        results = [{"chunk": chunks[i], "score": float(s), "original_index": i} for i, s in enumerate(scores)]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
