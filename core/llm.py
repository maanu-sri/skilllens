"""Thin OpenAI-compatible client. Temperature 0, JSON output, one retry, returns None on ANY failure
(callers then fall back to rule-based behaviour, so the app never depends on the API being up)."""
import json
import os
from .config import LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT

LAST_ERROR = ""


def available() -> bool:
    return bool(os.getenv("LLM_API_KEY"))


def chat_json(system: str, user: str, retries: int = 1):
    global LAST_ERROR
    if not available():
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL", LLM_BASE_URL),
                        timeout=LLM_TIMEOUT)
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__}: {e}"
        return None
    for _ in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", LLM_MODEL), temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            LAST_ERROR = f"{type(e).__name__}: {str(e)[:200]}"
    return None
