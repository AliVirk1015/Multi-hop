"""
Minimal OpenAI-compatible LLM client for the RAG pipeline.

Credentials come from `.env` (no python-dotenv dependency):
    LLM_BASE_URL   e.g. http://host:7100/v1, https://openrouter.ai/api/v1
    LLM_API_KEY    (may be "not-needed" for local gateways)
    LLM_MODEL      e.g. gpt-oss-20b / google/gemma-4-31b-it:free
    LLM_TIMEOUT    seconds

If no LLM_BASE_URL is set but OPENAI_API_KEY is present, the client falls back
to OpenAI's API (https://api.openai.com/v1) with model `OPENAI_MODEL` or
gpt-4o-mini — so the pipeline works out of the box with just an OpenAI key.

Transient 429/5xx responses (common on free tiers such as OpenRouter `:free`)
are retried with exponential backoff + Retry-After support.

`llm_reachable()` only checks that base_url + model are configured; it does NOT
ping the server. Actual call failures are raised and handled by callers
(planner falls back to deterministic; main_query prints a note).

The retrieval pipeline does NOT require the LLM to run — it is used only for
optional multi-hop planning and answer synthesis.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import httpx

DOTENV_LOADED = False

# Retry policy for transient gateway errors (rate limits, 5xx).
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "1.5"))


def _load_env(path: str = ".env") -> None:
    """Tiny .env loader (no dependency): KEY=VALUE lines, never overrides set vars."""
    global DOTENV_LOADED
    if DOTENV_LOADED or not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value
    DOTENV_LOADED = True


_load_env()


OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def settings() -> dict:
    base_url = (os.getenv("LLM_BASE_URL", "") or "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "") or ""
    model = (os.getenv("LLM_MODEL", "") or "").strip()

    # No explicit gateway -> fall back to OpenAI directly if a key is present.
    if not base_url and os.getenv("OPENAI_API_KEY"):
        base_url = OPENAI_BASE_URL
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not model:
            model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    return {
        "base_url": base_url,
        "api_key": api_key or "not-needed",
        "model": model,
        "timeout": float(os.getenv("LLM_TIMEOUT", "120")),
    }


def llm_reachable() -> bool:
    s = settings()
    return bool(s["base_url"] and s["model"])


def llm_chat(
    system: str,
    user: str,
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> str:
    """Single OpenAI-compatible chat completion -> assistant text."""
    s = settings()
    if not llm_reachable():
        raise RuntimeError(
            "LLM not configured: set LLM_BASE_URL + LLM_MODEL, or OPENAI_API_KEY in .env"
        )
    payload = {
        "model": s["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = f"{s['base_url']}/chat/completions"
    headers = {"Authorization": f"Bearer {s['api_key']}"}

    # Free-tier gateways (e.g. OpenRouter `:free` models) rate-limit hard.
    # Retry transient 429/5xx with exponential backoff + Retry-After support.
    last_exc: Optional[Exception] = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=s["timeout"])
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = float(resp.headers.get("Retry-After", 0) or 0)
                wait = retry_after or min(LLM_BACKOFF_BASE * (2 ** attempt), 20)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
            last_exc = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(min(LLM_BACKOFF_BASE * (2 ** attempt), 20))
    raise RuntimeError(f"LLM call failed after {LLM_MAX_RETRIES} attempts: {last_exc}")


def llm_chat_json(system: str, user: str, max_tokens: int = 600) -> dict:
    """Chat completion parsed as JSON; returns {} on any failure."""
    try:
        text = llm_chat(system, user, max_tokens=max_tokens)
        return json.loads(text)
    except Exception:
        return {}
