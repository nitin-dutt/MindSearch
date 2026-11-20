"""Prompt profiles (GEN items): every profile the settings accept exists, the production default is unchanged, and the profiles the
chat uses keep the two rules the rest of the system reads back (the exact refusal sentence and numbered citations)."""
import llm
import settings as S
from answers import ABSTAIN_MESSAGE


def test_the_default_profile_is_the_qwen35_one():
    """Since 2026-11-19 the default generator is Qwen3.5-4B, with the profile tuned for it on dev sets (enterprise_q35c).
    The Qwen2.5 rollback (models/qwen25-7b.env) sets PROMPT_PROFILE=enterprise."""
    assert S.Settings.from_env({}).prompt_profile == "enterprise_q35c"
    assert S.Settings.from_env({"PROMPT_PROFILE": "enterprise"}).prompt_profile == "enterprise"


def test_every_accepted_profile_has_a_prompt():
    for name in S.PROMPT_PROFILES:
        S.Settings.from_env({"PROMPT_PROFILE": name})          # accepted by the settings ...
        assert llm.PROMPTS[name]                               # ... and defined
    assert set(llm.PROMPTS) == set(S.PROMPT_PROFILES)


def test_v4_keeps_the_refusal_sentence_and_citations_and_adds_the_two_step_rule():
    """enterprise_v4 (2026-11-19, under measurement): the enterprise rules plus an explicit procedure for questions that go through a
    middle person or thing, and names written as the documents write them."""
    p = llm.PROMPTS["enterprise_v4"]
    assert ABSTAIN_MESSAGE in p and "[1]" in p
    assert "first find" in p.lower() and "exactly as the documents" in p.lower()


def test_every_enterprise_profile_keeps_the_rules_the_app_reads_back():
    """The app recognises a refusal by its exact sentence and maps citations by number; every production-style profile must ask for both."""
    for name in S.PROMPT_PROFILES:
        if name.startswith("enterprise"):
            assert ABSTAIN_MESSAGE in llm.PROMPTS[name] and "[1]" in llm.PROMPTS[name], name


def test_the_qwen35_profiles_keep_the_multi_hop_rule_and_demand_the_bare_refusal_sentence():
    """MODEL-01 #3 (2026-11-19, tuned on dev only): Qwen3.5-4B often refused in its own words ('the documents do not contain ...'),
    which the app cannot recognise, and answered questions about something merely similar to what the documents state.
    q35a fixes the refusal format; q35b also adds the 'exactly the same thing' check. Both keep rule 4 (combine documents)."""
    rule4 = "Some questions need facts from SEVERAL documents"
    for name in ("enterprise_q35a", "enterprise_q35b", "enterprise_q35c"):
        p = llm.PROMPTS[name]
        assert rule4 in p and "nothing else" in p.lower(), name
    assert "exactly" in llm.PROMPTS["enterprise_q35b"].split("4.")[0]      # the same-thing check comes before the multi-hop rule
