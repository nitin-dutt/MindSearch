from openai import AsyncOpenAI

from settings import get_settings

# Connect to the local vLLM instance
# By default, we assume vLLM is running on port 8001
VLLM_API_BASE = get_settings().vllm_api_base
VLLM_API_KEY = "EMPTY" # vLLM doesn't require an API key by default

client = AsyncOpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_API_BASE,
)

# Errors are still yielded as text for now (alignment_plan GEN-04). The evaluation harness
# uses this prefix to count failed generations instead of scoring them as answers.
LLM_ERROR_PREFIX = "Error generating response from vLLM:"

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
            max_tokens=1024
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"{LLM_ERROR_PREFIX} {str(e)}"


async def complete_chat(model, messages, max_tokens=100, temperature=0.0):
    """One non-streaming completion (used for short helper tasks such as rewriting a follow-up question)."""
    response = await client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                                    temperature=temperature)
    return response.choices[0].message.content or ""
