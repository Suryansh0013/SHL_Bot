"""
Thin wrapper around the LLM call so agent.py doesn't care which provider is
behind it. Defaults to Groq (OpenAI-compatible endpoint, free tier, and fast
inference -- matters because the evaluator enforces a 30s per-call timeout).

Swap providers by changing LLM_PROVIDER / API key env vars. Gemini and
OpenRouter both also expose OpenAI-compatible endpoints so the same client
code works for all three with just a base_url + model change.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

_PROVIDER_CONFIG = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.1-8b-instant",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
}

_cfg = _PROVIDER_CONFIG[PROVIDER]
_api_key = os.getenv(_cfg["api_key_env"])
_model = os.getenv("LLM_MODEL", _cfg["default_model"])

if not _api_key:

    raise RuntimeError(

        f"Missing {_cfg['api_key_env']} in your .env file."

    )

_client = OpenAI(

    base_url=_cfg["base_url"],

    api_key=_api_key,
    timeout=20,

 ) 


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
    """Calls the LLM and parses a strict JSON object out of the response.
    Retries once with a stricter instruction if parsing fails."""
    if _client is None:
        raise RuntimeError(
            f"No API key set for provider '{PROVIDER}'. "
            f"Set {_cfg['api_key_env']} as an environment variable."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(2):
        resp = _client.chat.completions.create(
            model=_model,
            messages=messages,
            temperature=temperature,
            max_tokens=350,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        parsed = _try_parse_json(raw)
        if parsed is not None:
            return parsed
        # retry with a correction nudge
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": "That was not valid JSON. Respond with ONLY a valid JSON object, no markdown fences, no commentary.",
        })

    raise ValueError(f"LLM did not return valid JSON after retries. Last raw output: {raw[:500]}")


def _try_parse_json(raw: str):
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to salvage the largest {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None
