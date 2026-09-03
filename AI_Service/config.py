"""Central configuration for AI_Service, loaded from environment / .env file.

Same pattern as Core_LLM/deployment/config.py and Orchestrator's .env.example:
nothing below is hard-coded anywhere else in the module.

The five versioning keys (ROUTING_POLICY_VERSION, PROMPT_VERSION,
FEWSHOT_VERSION, SAFETY_VERSION, ROUTER_ENABLED) live HERE and not in
Core_LLM's config, even though the task doc listed them under Core_LLM:
Core_LLM is a pure inference server that knows nothing about domains, prompt
versions or routing policies. AI_Service is the source of truth for them. If
Core_LLM ever wants them for telemetry it can be told at request time; it
should not own them.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # reads a .env file if present; no-op otherwise


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# --- Where Core_LLM is ------------------------------------------------------
# AI_Service NEVER loads a model itself; every completion is an HTTP call to
# Core_LLM's POST /chat (CONTRIBUTING.md: modules talk over API only -- and
# loading a second copy of a model would double VRAM for no benefit).
#
# 127.0.0.1, NOT localhost, and that is not cosmetic. Core_LLM binds
# 0.0.0.0:8001 -- IPv4 only, no [::]:8001 listener -- while "localhost" on
# Windows resolves to ::1 BEFORE 127.0.0.1. Every call therefore tries the
# IPv6 address first, and Windows takes ~2s to return WSAECONNREFUSED on
# loopback rather than refusing immediately, so httpx burns two seconds
# before falling back to IPv4. Measured on this project: 2.07s per call via
# `localhost` vs 0.01s via `127.0.0.1`, on every health check and every
# interview turn. Set CORE_LLM_URL explicitly when Core_LLM is on another
# host (or reachable over IPv6).
CORE_LLM_URL = os.getenv("CORE_LLM_URL", "http://127.0.0.1:8001")

# Seconds to wait on a Core_LLM call. A big local model answering a long
# interview turn can genuinely take minutes, and the first call after a model
# switch also pays the load cost -- same reasoning as Orchestrator's default.
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "600"))

# --- This service's own address --------------------------------------------
# 8000 = STT, 8001 = Core_LLM, 9000 = Orchestrator, so 9100 here.
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9100"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# --- Versioned behaviour ----------------------------------------------------
# Which routing policy runs (router/policy.py registers one function per
# version), which prompt folder is read (prompts/v{n}/), which few-shot file
# from that folder, and which safety rule set applies. All four are stamped
# onto responses and evaluation results so a result can be traced back to
# exactly the behaviour that produced it.
ROUTING_POLICY_VERSION = os.getenv("ROUTING_POLICY_VERSION", "v1")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
FEWSHOT_VERSION = os.getenv("FEWSHOT_VERSION", PROMPT_VERSION)
SAFETY_VERSION = os.getenv("SAFETY_VERSION", "v1")

# Turn the router off to pin every conversation to DEFAULT_DOMAIN /
# DEFAULT_MODEL_KEY -- useful for A/B runs where routing is the variable you
# want held still, and as a kill switch if a policy misbehaves in production.
ROUTER_ENABLED = _flag("ROUTER_ENABLED", "true")

# --- Model selection --------------------------------------------------------
# Registry keys as published by Core_LLM's GET /models.
DEFAULT_MODEL_KEY = os.getenv("DEFAULT_MODEL_KEY", "gemma-4-12b")

# LICENSING -- MUST BE CONFIRMED BEFORE CLINICAL/COMMERCIAL GO-LIVE.
# The Aya Expanse family is CC-BY-NC (non-commercial); every other model in
# Core_LLM's registry is Apache-2.0. With this false (the default), the router
# will not select an Aya model no matter what the policy would otherwise
# prefer. Flip it to true ONLY for research/internal evaluation, and only with
# whoever owns licensing for this project saying so.
ALLOW_NON_COMMERCIAL_MODELS = _flag("ALLOW_NON_COMMERCIAL_MODELS", "false")
NON_COMMERCIAL_MODEL_KEYS = {"aya-expanse-8b", "aya-expanse-32b"}

# --- Generation -------------------------------------------------------------
# Low by default: a triage interview wants consistency, not creativity.
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

# --- Safety bounds (structural, not clinical) -------------------------------
# Character counts, checked by safety/precheck.py and safety/output_validator.py.
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))
MAX_OUTPUT_CHARS = int(os.getenv("MAX_OUTPUT_CHARS", "8000"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "600"))

# --- Sessions ---------------------------------------------------------------
# In-memory only (see main.py's docstring for the trade-off). Oldest sessions
# are evicted past this count so a long-running server can't grow unbounded.
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))
