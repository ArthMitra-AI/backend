import json
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _cache_key(text: str, tier: str) -> str:
    return "llm:" + hashlib.sha256(f"{tier}:{text}".encode()).hexdigest()[:32]


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        t = "\n".join(lines[1:end])
    return t.strip()


async def call_llm(
    prompt: str,
    system: str = "",
    use_pro: bool = False,
    use_cache: bool = True,
    cache_ttl: int = 3600,
    temperature: float = 0.3,
) -> str:
    from core.cache import get_cache
    key = _cache_key(f"{system[:100]}{prompt}", "pro" if use_pro else "flash")

    if use_cache:
        cached = await get_cache().get(key)
        if cached:
            logger.info("LLM cache hit")
            return cached

    # Try Groq first (free, fast)
    result = await _try_groq(prompt, system, use_pro, temperature)

    # Fallback to Gemini
    if result is None:
        logger.warning("Groq failed — trying Gemini")
        result = await _try_gemini(prompt, system, use_pro, temperature)

    # Fallback to OpenAI
    if result is None:
        logger.warning("Gemini failed — trying OpenAI")
        result = await _try_openai(prompt, system, temperature)

    if result is None:
        raise RuntimeError("All LLM providers failed. Check your API keys.")

    if use_cache:
        await get_cache().set(key, result, ex=cache_ttl)

    return result


async def call_llm_json(
    prompt: str,
    system: str = "",
    use_pro: bool = False,
    use_cache: bool = True,
) -> dict:
    json_system = (
        system
        + "\n\nCRITICAL: Return ONLY valid JSON. "
        + "No markdown fences. No explanation. Just the JSON object or array."
    )

    text = await call_llm(prompt, json_system, use_pro=use_pro, use_cache=use_cache)
    cleaned = _strip_fences(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed — retrying")
        retry = (
            prompt
            + "\n\nReturn ONLY valid JSON starting with { or [. Nothing else."
        )
        text2 = await call_llm(retry, json_system, use_pro=use_pro, use_cache=False)
        return json.loads(_strip_fences(text2))


async def _try_groq(
    prompt: str,
    system: str,
    use_pro: bool,
    temperature: float,
) -> Optional[str]:
    try:
        from groq import Groq
        from core.config import get_settings
        s = get_settings()

        api_key = getattr(s, 'groq_api_key', '') or __import__('os').getenv('GROQ_API_KEY', '')
        if not api_key:
            return None

        client = Groq(api_key=api_key)

        # Use 70b for pro (X-Ray), 8b for everything else
        model = "llama-3.3-70b-versatile" if use_pro else "llama-3.1-8b-instant"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=3000,
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.warning(f"Groq failed: {e}")
        return None


async def _try_gemini(
    prompt: str,
    system: str,
    use_pro: bool,
    temperature: float,
) -> Optional[str]:
    try:
        from core.config import get_settings
        import google.generativeai as genai
        s = get_settings()

        if not getattr(s, 'gemini_api_key', ''):
            return None

        genai.configure(api_key=s.gemini_api_key)
        model_name = getattr(s, 'gemini_pro_model', 'gemini-2.0-flash') if use_pro else getattr(s, 'gemini_flash_model', 'gemini-2.0-flash')

        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": temperature, "max_output_tokens": 3000},
            system_instruction=system if system else None,
        )
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        logger.warning(f"Gemini failed: {e}")
        return None


async def _try_openai(
    prompt: str,
    system: str,
    temperature: float,
) -> Optional[str]:
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        import os

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=api_key,
            max_tokens=3000,
        )
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        response = await llm.ainvoke(messages)
        return response.content

    except Exception as e:
        logger.warning(f"OpenAI failed: {e}")
        return None