"""docker-compose.yml switches (alignment_plan Phase 9). Since 2026-11-19 the default stack is the measured winner of MODEL-01 / PERF-03:
Qwen3.5-4B (4-bit), CUDA graphs, at most 12 answers at once, the Qwen3.5 prompt and the cross-encoder re-ranker on the GPU. Every
piece stays a switch, and models/qwen25-7b.env brings back the previous stack (models/judge-llama31-8b.env serves the judge)."""
import os
import re

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
COMPOSE = os.path.join(ROOT, "docker-compose.yml")
QWEN35 = "RedHatAI/Qwen3.5-4B-quantized.w4a16"


def _compose_text():
    if not os.path.exists(COMPOSE):
        pytest.skip("docker-compose.yml is not in this checkout (tests run inside the container)")
    return open(COMPOSE, encoding="utf-8").read()


def _env_file(name):
    _compose_text()
    path = os.path.join(ROOT, "models", name)
    if not os.path.exists(path):
        # models/*.env is git-ignored (local presets), so a fresh clone does not have it
        pytest.skip(f"models/{name} is not in this checkout (local preset, git-ignored)")
    pairs = {}
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            k, _, v = line.partition("=")
            pairs[k.strip()] = v.strip()
    return pairs


def test_kv_cache_dtype_is_a_switch_that_defaults_to_auto():
    """PERF-02: VLLM_KV_CACHE_DTYPE picks the KV cache precision; `auto` = the model's 16-bit type (fp8 was measured and rejected).
    A bad value is not caught here on purpose: vLLM rejects it at start-up and the API never starts (loud, no fallback)."""
    assert re.search(r"--kv-cache-dtype \$\{VLLM_KV_CACHE_DTYPE:-auto\}", _compose_text())


def test_the_default_generator_is_qwen35_4b_with_its_own_server_settings():
    """MODEL-01: the 4-bit format, tool-call parser and reasoning parser belong to the model; thinking is off for every call."""
    text = _compose_text()
    assert f"--model ${{VLLM_MODEL:-${{GENERATOR_MODEL:-{QWEN35}}}}}" in text
    assert "--quantization ${VLLM_QUANTIZATION:-compressed-tensors}" in text
    assert "--tool-call-parser ${VLLM_TOOL_PARSER:-qwen3_coder}" in text
    assert text.count(f"GENERATOR_MODEL=${{GENERATOR_MODEL:-{QWEN35}}}") == 2          # API and worker ask for the same model
    assert text.count("GENERATOR_THINKING=${GENERATOR_THINKING:-off}") == 2


def test_qwen35_flags_are_on_by_default_and_can_be_dropped_by_an_empty_value():
    """`${VAR-default}` (one dash) uses the default only when VAR is unset; VAR= (empty) drops the flag, for a model that does not
    think or cannot read images (the rollback and judge env files do that). Without --language-model-only vLLM reserved memory for a
    worst-case image and had no room left for the KV cache (measured 2026-11-19: -0.33 GiB, start-up refused)."""
    text = _compose_text()
    assert "${VLLM_REASONING_FLAG---reasoning-parser=qwen3}" in text
    assert "${VLLM_LANGUAGE_MODEL_ONLY_FLAG---language-model-only}" in text


def test_cuda_graphs_and_the_cap_are_the_default():
    """MODEL-01 speed: CUDA graphs (record the GPU steps once, replay them) are on unless VLLM_ENFORCE_EAGER is set; a hybrid model
    needs a cap on answers in progress for graphs (only 47 fixed-size slots fit), measured best at 12."""
    text = _compose_text()
    assert "${VLLM_ENFORCE_EAGER:+--enforce-eager}" in text
    assert not re.search(r"^\s*--enforce-eager\s*$", text, re.M)              # no unconditional eager flag left
    assert "--max-num-seqs ${VLLM_MAX_NUM_SEQS:-12}" in text
    assert ("${VLLM_MTP_TOKENS:+--speculative-config.method mtp --speculative-config.num_speculative_tokens ${VLLM_MTP_TOKENS}}"
            in text)                                                           # measured second best; does not fit with graphs on 8 GB
    assert "--gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" in text


def test_the_api_reranks_on_the_gpu_and_the_worker_on_the_cpu():
    """PERF-03: the API image carries a CUDA PyTorch and gets the GPU; the worker has no GPU and keeps the CPU re-ranker (its scores
    were identical on 100/100 search pools). The GPU's compiled-code cache survives container re-creation (first question 7.7 s -> 1.3 s)."""
    import yaml
    text = _compose_text()
    cfg = yaml.safe_load(text)
    api, worker = cfg["services"]["api"], cfg["services"]["celery_worker"]
    assert api["build"]["args"]["TORCH_CUDA"] == "${TORCH_CUDA:-1}"
    assert "RERANKER_DEVICE=${RERANKER_DEVICE:-cuda}" in api["environment"]
    assert api["deploy"]["resources"]["reservations"]["devices"][0]["capabilities"] == ["gpu"]
    assert "reranker_cuda_cache:/root/.nv" in api["volumes"] and "reranker_cuda_cache" in cfg["volumes"]
    assert "RERANKER_DEVICE=${WORKER_RERANKER_DEVICE:-cpu}" in worker["environment"]
    assert "deploy" not in worker
    assert text.count("RERANKER_BACKEND=${RERANKER_BACKEND:-torch}") == 2
    assert text.count('WITH_ONNX: "${WITH_ONNX:-0}"') == 2
    assert not os.path.exists(os.path.join(ROOT, "docker-compose.gpu-reranker.yml"))   # merged into the main file


def test_the_rollback_env_file_restores_the_previous_stack():
    """models/qwen25-7b.env + `docker compose --env-file .env --env-file models/qwen25-7b.env up -d` = the stack as it was before
    2026-11-19 (Qwen2.5-7B, AWQ, hermes, no reasoning parser, full model, eager, no cap worth hitting, production prompt, CPU re-ranker)."""
    e = _env_file("qwen25-7b.env")
    assert e["GENERATOR_MODEL"] == "Qwen/Qwen2.5-7B-Instruct-AWQ" and e["VLLM_QUANTIZATION"] == "awq"
    assert e["VLLM_TOOL_PARSER"] == "hermes" and e["VLLM_REASONING_FLAG"] == "" and e["VLLM_LANGUAGE_MODEL_ONLY_FLAG"] == ""
    assert e["VLLM_ENFORCE_EAGER"] == "1" and e["GENERATOR_THINKING"] == "none" and e["PROMPT_PROFILE"] == "enterprise"
    assert e["RERANKER_DEVICE"] == "cpu" and e["VLLM_MAX_NUM_SEQS"] == "256"


def test_the_judge_env_file_serves_llama_without_qwen_flags():
    e = _env_file("judge-llama31-8b.env")
    assert e["VLLM_MODEL"] == "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" and e["VLLM_QUANTIZATION"] == "awq"
    assert e["VLLM_TOOL_PARSER"] == "llama3_json" and e["VLLM_REASONING_FLAG"] == "" and e["VLLM_LANGUAGE_MODEL_ONLY_FLAG"] == ""
    assert e["VLLM_MAX_LEN"] == "5120"


def test_vllm_keeps_its_compile_caches_across_container_recreation():
    """Measured 2026-11-19: with CUDA graphs, the first question after vLLM was re-created took 37 s (the next 2.4 s): vLLM compiled GPU
    code (torch.compile, Triton kernels) into /root/.cache/vllm and /root/.triton, which a new container had lost. Named volumes keep them."""
    import yaml
    cfg = yaml.safe_load(_compose_text())
    vols = cfg["services"]["vllm_engine"]["volumes"]
    assert "vllm_compile_cache:/root/.cache/vllm" in vols and "vllm_triton_cache:/root/.triton" in vols
    assert "vllm_compile_cache" in cfg["volumes"] and "vllm_triton_cache" in cfg["volumes"]


def test_vllm_is_healthy_only_once_it_can_generate():
    """Measured 2026-11-19: the first generation of a fresh vLLM process took 33 s, even for 1 token (later ones 0.07 s), whatever the
    caches. The health check asks for one token, so that cost is paid before 'healthy' (the API waits for it) and never by a user;
    if generation is broken, vLLM stays unhealthy and the API does not start (loud). It reads the served model name from /v1/models."""
    import yaml
    hc = yaml.safe_load(_compose_text())["services"]["vllm_engine"]["healthcheck"]
    test = " ".join(hc["test"])
    # the chat endpoint, not /v1/completions: a 1-token /v1/completions warm-up left the first chat question at 33 s (measured)
    assert "/v1/models" in test and "/v1/chat/completions" in test and "max_tokens" in test
    assert int(hc["timeout"].rstrip("s")) >= 60


def test_vllm_does_not_go_online_by_default():
    """Found 2026-11-19: on its first chat request vLLM looked up the model's file list on huggingface.co; with the connection failing
    it retried for ~33 s (the 'slow first question'), and it is a hidden outbound call in a self-hosted stack (alignment_plan ENG-06).
    HF_HUB_OFFLINE=1 makes vLLM use only the model already in the cache; a first-ever start sets HF_HUB_OFFLINE=0 once to download."""
    import yaml
    env = yaml.safe_load(_compose_text())["services"]["vllm_engine"]["environment"]
    assert "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" in env
