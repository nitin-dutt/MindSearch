import os
import re

import pytest

import settings as S

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_defaults_match_the_verified_stack():
    cfg = S.Settings.from_env({})
    assert cfg.generator_model == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert cfg.kg_extractor == "ollama" and cfg.pipeline_preset == "hybrid_graph_chunks" and cfg.retrieval_workers == 4
    assert cfg.qdrant_location == "./qdrant_db"            # embedded folder when no server URL is set


def test_environment_overrides_a_model_or_an_address_without_code_changes():
    cfg = S.Settings.from_env({"EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5", "QDRANT_URL": "http://qdrant:6333",
                               "VLLM_API_BASE": "http://vllm:8000/v1", "RETRIEVAL_WORKERS": "8"})
    assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
    assert cfg.qdrant_location == "http://qdrant:6333"
    assert cfg.vllm_api_base == "http://vllm:8000/v1" and cfg.judge_api_base == "http://vllm:8000/v1"   # judge follows
    assert cfg.retrieval_workers == 8


def test_judge_endpoint_can_differ_from_the_generator_endpoint():
    cfg = S.Settings.from_env({"VLLM_API_BASE": "http://a/v1", "JUDGE_API_BASE": "http://b/v1"})
    assert (cfg.vllm_api_base, cfg.judge_api_base) == ("http://a/v1", "http://b/v1")


def test_the_kg_vllm_model_falls_back_to_the_generator_model():
    assert S.Settings.from_env({"GENERATOR_MODEL": "m1"}).effective_kg_vllm_model == "m1"
    assert S.Settings.from_env({"GENERATOR_MODEL": "m1", "KG_VLLM_MODEL": "m2"}).effective_kg_vllm_model == "m2"


@pytest.mark.parametrize("env", [{"RETRIEVAL_WORKERS": "four"}, {"REWRITE_TIMEOUT_S": "soon"}, {"KG_EXTRACTOR": "gpt"},
                                 {"RETRIEVAL_WORKERS": "0"}, {"HISTORY_MAX_MESSAGES": "-1"}])
def test_bad_values_fail_with_a_clear_message(env):
    with pytest.raises(S.ConfigError):
        S.Settings.from_env(env)


def test_the_config_dump_has_no_secrets_and_covers_every_field():
    secrets = {"NEO4J_PASSWORD": "pw-123", "MINDSEARCH_JWT_SECRET": "jwt-456", "MINDSEARCH_DATA_KEY": "key-789"}
    dumped = str(S.Settings.from_env(secrets).describe())
    assert not any(v in dumped for v in secrets.values())
    assert set(S.Settings.from_env({}).describe()) == set(S.field_names())


def test_get_settings_reads_the_live_environment(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODEL", "some/other-model")
    assert S.get_settings().generator_model == "some/other-model"


def test_the_embedding_size_comes_from_the_model_not_a_constant():
    pytest.importorskip("sentence_transformers")
    import embedder
    assert embedder.EMBEDDING_DIM == len(embedder.model.encode("probe"))


# ---- guard: nobody hard-codes a model name or reads a setting from the environment outside settings.py

MODULES = [f for f in os.listdir(BACKEND) if f.endswith(".py") and f not in ("settings.py", "conftest.py")] + \
          ["evaluation/" + f for f in os.listdir(os.path.join(BACKEND, "evaluation")) if f.endswith(".py")]
SECRET_OR_TOOLING_READS = {"auth.py", "encryptor.py", "kg_builder.py", "manage_users.py", "benchmark_security.py",
                           "prepare_hotpot_dataset.py", "prepare_large_dataset.py", "seed_demo_users.py",
                           "evaluation/generate.py", "evaluation/followup.py"}   # secrets / CLI-only knobs, listed on purpose
HARD_CODED_MODEL = re.compile(r"""["'](Qwen/|qwen2\.5|all-MiniLM|cross-encoder/|en_core_web|BAAI/|meta-llama|hugging-quants)""")


def read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


# the retrieval benchmark deliberately names the candidate models it compares (it is a list of experiments, not configuration)
@pytest.mark.parametrize("rel", [m for m in MODULES if m != "evaluation/retrieval_bench.py"])
def test_no_model_name_is_hard_coded_outside_settings(rel):
    code = "\n".join(line for line in read(rel).splitlines() if not line.lstrip().startswith(("#", '"', "'")))
    hits = [m.group(0) for m in HARD_CODED_MODEL.finditer(code) if "help=" not in code[max(0, m.start() - 80):m.start()]]
    assert not hits, f"{rel} hard-codes a model name {hits}; add it to settings.py instead"


@pytest.mark.parametrize("rel", [m for m in MODULES if m not in SECRET_OR_TOOLING_READS])
def test_application_modules_do_not_read_the_environment_directly(rel):
    assert not re.search(r"os\.(getenv|environ)", read(rel)), f"{rel} reads os.environ; use settings.get_settings()"
