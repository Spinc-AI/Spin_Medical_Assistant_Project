"""Thin wrapper around the local LLM.

This is the single integration seam for the whole medical-assistant project.
Later, the CT-scan model, STT/TTS, sentiment analysis, and RAG references will
all funnel their output into `messages` here, and the spoken/written answer
comes back out. Keep this file small and stable; build features around it.
"""
import httpx
from openai import OpenAI

import config

_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


def chat(messages, *, model=None, temperature=0.3, stream=False, response_format=None):
    """Send a list of chat messages and return the assistant's reply.

    `messages` is the standard OpenAI format, e.g.:
        [{"role": "system", "content": "..."},
         {"role": "user", "content": "..."}]

    With stream=True this yields text chunks; otherwise returns the full string.
    Low temperature (0.3) by default — for medical use we want consistency,
    not creativity.
    """
    extra = {"response_format": response_format} if response_format else {}
    response = _client.chat.completions.create(
        model=model or config.LLM_MODEL,
        messages=messages,
        temperature=temperature,
        stream=stream,
        **extra,
    )

    if stream:
        return _stream_text(response)
    return response.choices[0].message.content


def _stream_text(response):
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def unload(model=None):
    """Ask Ollama to unload the model from memory (keep_alive=0).

    The OpenAI-compatible /v1 layer has no unload, so we call Ollama's native API.
    """
    payload = {"model": model or config.LLM_MODEL, "keep_alive": 0}
    with httpx.Client(timeout=30) as c:
        c.post(f"{config.OLLAMA_URL}/api/generate", json=payload)
