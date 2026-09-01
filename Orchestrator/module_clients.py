"""Thin HTTP clients for STT and Core_LLM.

Pipelines call these instead of talking HTTP directly — keeps the actual
module URLs, timeouts, and request shapes in one place. Every function here
is a straight wrapper over one module endpoint; the *choice* of which model
to use is made by pipeline code (see pipelines/), never by an orchestrator
API caller.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import httpx

import config


def is_api_model(model: Optional[str]) -> bool:
    """True if `model` (a choice made in pipeline code) selects the external API."""
    return bool(model) and model.startswith(config.API_PREFIX)


def strip_api_prefix(model: Optional[str]) -> str:
    return model[len(config.API_PREFIX):] if model else ""


def _client() -> httpx.Client:
    return httpx.Client(timeout=config.HTTP_TIMEOUT)


def _resolve_api_key(api_key: Optional[str]) -> str:
    key = api_key or config.OPENAI_API_KEY
    if not key:
        raise RuntimeError(
            "No API key available for this external call — set OPENAI_API_KEY "
            "in the orchestrator's .env, or pass one in from pipeline code."
        )
    return key


def _resolve_base_url(base_url: Optional[str]) -> str:
    return (base_url or config.OPENAI_BASE_URL).rstrip("/")


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------
def stt_health() -> bool:
    try:
        with _client() as c:
            return c.get(f"{config.STT_URL}/models").status_code == 200
    except httpx.HTTPError:
        return False


def stt_load(model: str) -> None:
    with _client() as c:
        r = c.post(f"{config.STT_URL}/models/{model}/load")
    if r.status_code != 200:
        raise RuntimeError(f"STT load failed ({r.status_code}): {r.text}")


def stt_transcribe(audio: bytes, filename: str = "audio.wav", language: Optional[str] = None) -> str:
    """Transcribe with the local STT service. Caller must have loaded the model already."""
    data = {"language": language} if language else None
    with _client() as c:
        r = c.post(f"{config.STT_URL}/transcribe", files={"file": (filename, audio)}, data=data)
    if r.status_code != 200:
        raise RuntimeError(f"STT transcribe failed ({r.status_code}): {r.text}")
    return r.json()["text"]


def stt_unload() -> None:
    with _client() as c:
        c.post(f"{config.STT_URL}/models/unload")


# ---------------------------------------------------------------------------
# Core_LLM (local) + external API — dispatch is a pipeline's own choice
# ---------------------------------------------------------------------------
def llm_health() -> bool:
    try:
        with _client() as c:
            return c.get(f"{config.LLM_URL}/").status_code == 200
    except httpx.HTTPError:
        return False


def llm_chat(messages: list[dict], model: Optional[str] = None,
             response_format: Optional[dict] = None) -> str:
    """Chat with the local Core_LLM service."""
    payload: dict[str, Any] = {"messages": messages}
    if model:
        payload["model"] = model
    if response_format:
        payload["response_format"] = response_format
    with _client() as c:
        r = c.post(f"{config.LLM_URL}/chat", json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"LLM chat failed ({r.status_code}): {r.text}")
    return r.json()["reply"]


def llm_api_chat(messages: list[dict], model: str, response_format: Optional[dict] = None,
                  api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """Chat with an external, OpenAI-compatible LLM API."""
    key = _resolve_api_key(api_key)
    url = _resolve_base_url(base_url)
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if response_format:
        payload["response_format"] = response_format
    with _client() as c:
        r = c.post(f"{url}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"API LLM chat failed ({r.status_code}): {r.text}")
    return r.json()["choices"][0]["message"]["content"]


def chat(messages: list[dict], model: str, response_format: Optional[dict] = None) -> str:
    """Dispatch to the local Core_LLM or the external API, based on `model` —
    a value pipeline code chose, never one an orchestrator API caller supplied."""
    if is_api_model(model):
        return llm_api_chat(messages, strip_api_prefix(model), response_format=response_format)
    return llm_chat(messages, model=model, response_format=response_format)


def llm_unload(model: Optional[str] = None) -> None:
    if is_api_model(model):
        return  # external API — nothing local to unload
    params = {"model": model} if model else {}
    with _client() as c:
        c.post(f"{config.LLM_URL}/unload", params=params)


def extract_json(text: str) -> Any:
    """Pull a JSON object out of an LLM reply (tolerant of code fences / prose)."""
    text = text.strip()
    if "```" in text:
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in LLM reply")
    return json.loads(text[start:end + 1])
