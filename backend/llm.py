import logging

from openai import AsyncOpenAI

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

SYSTEM_PROMPT = "You are a precise, factual enterprise AI assistant. Answer the user's question using ONLY the provided context. The context contains document chunks wrapped in <doc> tags and supplementary knowledge graph facts wrapped in <kg_fact> tags. Rely primarily on the <doc> chunks. Be concise — answer in 1-2 sentences. Answer directly without any preamble phrases like 'Based on the context' or 'According to the provided information'. Do not add any information not explicitly stated in the context. Only include information that directly answers the question. Omit related facts, entities, or context not explicitly asked about, even if present above."

async def stream_generate(model, query, context, history=None):
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
