"""LLMClient over an OpenAI-compatible vLLM server. NOT WIRED INTO ANYTHING.

Mirrors Core_LLM/deployment/llm/vllm_client.py. It exists for one possible
future: a deployment where AI_Service talks to a vLLM server directly instead
of through Core_LLM. That is not today's architecture and choosing it would be
an architectural decision, not a config change -- Core_LLM is currently the
single owner of model loading and VRAM, and bypassing it would mean two
services independently deciding what to keep in GPU memory.

UNTESTED against a real vLLM server (none is deployed in this project). The
test that ships with it only pins the request/response shape.
"""
import time

import httpx

from .interface import LLMClient, LLMResponse

DEFAULT_TIMEOUT = 600.0


class VLLMClient(LLMClient):
    """Example:
        VLLMClient("http://localhost:8000", default_model="google/gemma-4-12B-it")
    """

    def __init__(self, base_url: str, default_model: str, timeout: float = DEFAULT_TIMEOUT,
                 api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.api_key = api_key

    def generate(self, messages: list[dict], temperature: float = 0.3,
                 max_tokens: int | None = None, model: str | None = None) -> LLMResponse:
        payload = {"model": model or self.default_model,
                   "messages": [dict(m) for m in messages],
                   "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"vLLM call failed ({r.status_code}): {r.text}")
        data = r.json()
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", payload["model"]),
            latency=time.perf_counter() - started,
            usage=data.get("usage"),
        )
