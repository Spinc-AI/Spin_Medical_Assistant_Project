"""The only way AI_Service reaches a language model: HTTP to Core_LLM.

There is deliberately no transformers/HuggingFace path in this module. Two
reasons, both firm:
  - CONTRIBUTING.md requires modules to communicate over API only.
  - Core_LLM already holds a model in VRAM. A second in-process copy would
    double GPU memory for identical output.

If offline evaluation ever needs to run without a live Core_LLM, the answer is
a fake client (see evaluation/runner.py, which does exactly that), not a local
model loader here.
"""
import logging
import time

import httpx

import config

from .interface import LLMClient, LLMResponse

# Why this module logs at all: every httpx failure below used to be swallowed
# into a bare `return False` / `return []`, which made "core_llm_reachable:
# false" in GET /health a dead end -- no way to tell a refused connection from
# a DNS miss, a proxy hijack or a timeout without attaching a debugger. The
# one-line WARNING is the default because it is what makes the cause visible
# with no flag to flip (root logger has no handler under uvicorn, so
# logging.lastResort still prints WARNING+ to stderr); the DEBUG line carries
# the full traceback for when the one-liner is not enough.
logger = logging.getLogger(__name__)


def _log_transport_failure(what: str, url: str, exc: Exception) -> None:
    """Record why an httpx call failed. Never swallow the cause silently."""
    logger.warning("Core_LLM %s failed: %s: %s (url=%s)",
                   what, type(exc).__name__, exc, url)
    logger.debug("Core_LLM %s traceback (url=%s)", what, url, exc_info=exc)


def _client(timeout: float) -> httpx.Client:
    """The one place an httpx client for Core_LLM is built.

    `trust_env=False` is load-bearing, not tidiness. httpx honours HTTP_PROXY /
    HTTPS_PROXY by default and -- unlike curl, PowerShell and WinINET, which
    all bypass proxies for local addresses -- applies them to loopback too.
    A developer machine running a local VPN/proxy client (Xray, V2Ray, Clash:
    they export HTTP_PROXY=http://127.0.0.1:<port> without a matching
    NO_PROXY) therefore has every AI_Service -> Core_LLM call swallowed by
    that proxy and timed out, while `curl http://localhost:8001/` from the
    same shell answers 200. That mismatch cost a long debugging session once;
    it should not cost another.

    Proxies are matched by SCHEME, so switching CORE_LLM_URL between
    `localhost` and `127.0.0.1` does NOT dodge this -- only trust_env=False
    does. Core_LLM is an internal service reached over plain HTTP on the same
    host or private network; there is no case where it should be proxied.

    Every call below goes through here so a new call site cannot quietly
    reintroduce the default.
    """
    return httpx.Client(timeout=timeout, trust_env=False)


class CoreLLMClient(LLMClient):
    """Calls Core_LLM's `POST /chat`.

    Example:
        client = CoreLLMClient()
        client.generate([{"role": "user", "content": "سلام"}], model="gemma-4-12b").text
    """

    def __init__(self, base_url: str | None = None, timeout: float | None = None,
                 default_model: str | None = None):
        self.base_url = (base_url or config.CORE_LLM_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else config.HTTP_TIMEOUT
        self.default_model = default_model or config.DEFAULT_MODEL_KEY

    def generate(self, messages: list[dict], temperature: float = 0.3,
                 max_tokens: int | None = None, model: str | None = None) -> LLMResponse:
        payload: dict = {
            "messages": [dict(m) for m in messages],
            "model": model or self.default_model,
            "temperature": temperature,
            # Core_LLM accepts response_format for interface parity but can't
            # enforce JSON mode locally -- the prompt asks for JSON and
            # safety/output_validator.py checks that it got it. Sent anyway so
            # the day Core_LLM gains a backend that *can* enforce it (vLLM's
            # guided decoding, or a cloud model), this starts working for free.
            "response_format": {"type": "json_object"},
        }
        # NOTE: Core_LLM's ChatRequest has no max_tokens field today, so this
        # is not sent -- it would be silently ignored. Generation length is
        # capped by Core_LLM's own MAX_NEW_TOKENS. Plumb it through when
        # Core_LLM's schema grows the field.
        _ = max_tokens

        started = time.perf_counter()
        try:
            with _client(self.timeout) as client:
                r = client.post(f"{self.base_url}/chat", json=payload)
        except httpx.HTTPError as exc:
            _log_transport_failure("POST /chat", f"{self.base_url}/chat", exc)
            raise CoreLLMUnavailable(f"Core_LLM at {self.base_url} is unreachable: "
                                     f"{type(exc).__name__}: {exc}") from exc
        latency = time.perf_counter() - started

        if r.status_code != 200:
            raise CoreLLMError(f"Core_LLM /chat failed ({r.status_code}): {r.text}")
        body = r.json()
        return LLMResponse(
            text=body["reply"],
            model=body.get("model", payload["model"]),
            # Measured client-side: Core_LLM's ChatResponse is {model, reply}
            # and reports no timing of its own, so this includes network time.
            latency=latency,
            # TODO: Core_LLM doesn't report token usage. Left None rather than
            # estimated -- a fabricated token count would quietly corrupt
            # evaluation/metrics.py's `tokens` and `throughput`. Fill this in
            # when Core_LLM's ChatResponse grows a `usage` field.
            usage=None,
        )

    def health(self) -> bool:
        """Is Core_LLM answering? Used by AI_Service's GET /health."""
        try:
            with _client(min(self.timeout, 10.0)) as client:
                return client.get(f"{self.base_url}/").status_code == 200
        except httpx.HTTPError as exc:
            _log_transport_failure("GET /", f"{self.base_url}/", exc)
            return False

    def models(self) -> list[str]:
        """Model keys Core_LLM currently registers; [] if it's unreachable."""
        try:
            with _client(min(self.timeout, 10.0)) as client:
                r = client.get(f"{self.base_url}/models")
            return r.json().get("available", []) if r.status_code == 200 else []
        except httpx.HTTPError as exc:
            _log_transport_failure("GET /models", f"{self.base_url}/models", exc)
            return []


class CoreLLMError(RuntimeError):
    """Core_LLM answered, but with an error."""


class CoreLLMUnavailable(CoreLLMError):
    """Core_LLM couldn't be reached at all (down, wrong URL, timeout)."""
