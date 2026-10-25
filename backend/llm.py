import logging

from openai import AsyncOpenAI

from answers import ABSTAIN_MESSAGE
from settings import get_settings

log = logging.getLogger("mindsearch.llm")

# Connect to the local vLLM instance
# By default, we assume vLLM is running on port 8001
VLLM_API_BASE = get_settings().vllm_api_base
VLLM_API_KEY = "EMPTY" # vLLM doesn't require an API key by default

client = AsyncOpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_API_BASE,
)

class LLMError(RuntimeError):
    """The language model could not produce an answer (server down, timeout, bad response). Never part of an answer:
    the API turns it into an HTTP 503 or an `error` event, and the evaluation counts it as a failed generation."""

# The original prompt, kept unchanged because every earlier benchmark result was produced with it (profile 'benchmark').
BENCHMARK_PROMPT = "You are a precise, factual enterprise AI assistant. Answer the user's question using ONLY the provided context. The context contains document chunks wrapped in <doc> tags and supplementary knowledge graph facts wrapped in <kg_fact> tags. Rely primarily on the <doc> chunks. Be concise — answer in 1-2 sentences. Answer directly without any preamble phrases like 'Based on the context' or 'According to the provided information'. Do not add any information not explicitly stated in the context. Only include information that directly answers the question. Omit related facts, entities, or context not explicitly asked about, even if present above."

# The production prompt (profile 'enterprise'): numbered sources, citations, an explicit way to say "I don't know", no length cap.
ENTERPRISE_PROMPT = (
    "You are a precise, factual enterprise assistant. Answer the user's question using ONLY the numbered documents given as context. "
    "Each document is wrapped in <doc id=\"N\"> tags; <kg_fact> tags, if present, hold supplementary facts.\n"
    "Rules:\n"
    "1. Use only information stated in the context. Never add outside knowledge or guess.\n"
    "2. Cite the documents you used by their number in square brackets, like [1] or [2][3], right after the claim they support. "
    "Cite only numbers that exist.\n"
    "3. Be as brief as the question allows: a short factual question gets a short answer; a request for a summary, comparison or procedure "
    "gets a clear, structured answer. Do not start with phrases like 'Based on the context'.\n"
    "4. Some questions need facts from SEVERAL documents (for example one document names a person and another says where that person was born). "
    "Combine them and cite every document you used; do not refuse just because the answer is spread over more than one document.\n"
    f"5. Only if the documents, taken together, still do not contain what is needed, reply with exactly: {ABSTAIN_MESSAGE} Do not cite anything in that case."
)

# Version 3 (profile 'enterprise_v3'): the same rules, but a document is cited only when its text was actually used. Measured against the
# production prompt: citation precision +3.4 points on SQuAD, unchanged on HotpotQA, but answers slightly worse and more false refusals
# (within noise), so it is NOT the default; see alignment_plan.md.
ENTERPRISE_PROMPT_V3 = (
    "You are a precise, factual enterprise assistant. Answer the user's question using ONLY the numbered documents given as context. "
    "Each document is wrapped in <doc id=\"N\"> tags; <kg_fact> tags, if present, hold supplementary facts.\n"
    "Rules:\n"
    "1. Use only information stated in the context. Never add outside knowledge or guess.\n"
    "2. Cite the documents you used by their number in square brackets, like [1] or [2][3], right after the claim they support. "
    "Cite only a document whose text you actually relied on: being among the documents given is not a reason to cite it, and do not "
    "put several numbers after a sentence unless each of them supports it. Cite only numbers that exist.\n"
    "3. Be as brief as the question allows: a short factual question gets a short answer; a request for a summary, comparison or procedure "
    "gets a clear, structured answer. Do not start with phrases like 'Based on the context'.\n"
    "4. Some questions need facts from SEVERAL documents (for example one document names a person and another says where that person was born). "
    "Combine them and cite each document that supplies a fact you used, and only those; do not refuse just because the answer is spread over more than one document.\n"
    f"5. Only if the documents, taken together, still do not contain what is needed, reply with exactly: {ABSTAIN_MESSAGE} Do not cite anything in that case."
)

PROMPTS = {"enterprise": ENTERPRISE_PROMPT, "enterprise_v3": ENTERPRISE_PROMPT_V3, "benchmark": BENCHMARK_PROMPT}


def get_system_prompt() -> str:
    """The system prompt for the configured profile (PROMPT_PROFILE: 'enterprise' by default, 'benchmark' for the original SQuAD-style one)."""
    return PROMPTS[get_settings().prompt_profile]


def build_system_prompt(today=None) -> str:
    """The profile's system prompt, plus today's date when the caller gives one. A language model does not know the date, so an answer that
    depends on it (a person's age, "how long ago") would otherwise be worked out from a guessed year. Only live chat passes `today`: the
    benchmark runs do not, so their prompts stay exactly as measured."""
    prompt = get_system_prompt()
    if today is not None:
        prompt += (f"\n\nToday's date is {today:%A, %d %B %Y}. Use it whenever an answer depends on the current date, such as someone's age "
                   "or how long ago something happened; never assume a different year."
                   "\nWrite plain text. Do not use LaTeX or other maths markup: write a calculation in ordinary words and symbols, "
                   "for example 2026 - 1981 = 45. For an age or any other length of time, first write the subtraction of the years, take one year off if the birthday or date has not yet come "
                   "this year, then state the result in a short sentence. Begin with the working, never with the result on its own.")
    return prompt


async def stream_generate(model, query, context, history=None, today=None):
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": build_system_prompt(today)},
                *(history or []),            # recent turns (already trimmed and role-checked by conversation.py)
                {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=0.1,
            max_tokens=get_settings().answer_max_tokens
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        # the details (URLs, library messages) stay in the server log; callers get a short, safe message
        log.error("language model request failed: %s: %s", type(e).__name__, e)
        raise LLMError(f"language model request failed ({type(e).__name__})") from e


async def complete_chat(model, messages, max_tokens=100, temperature=0.0):
    """One non-streaming completion (used for short helper tasks such as rewriting a follow-up question)."""
    response = await client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                                    temperature=temperature)
    return response.choices[0].message.content or ""
