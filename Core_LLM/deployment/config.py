"""Central configuration, loaded from environment / .env file.

Keeping this in one place means the rest of the code never hard-codes the
server address or model name — handy when you point at different servers.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # reads a .env file if present; no-op otherwise

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "aya-expanse")

# Native Ollama API base (for things the OpenAI-compat /v1 layer doesn't cover,
# e.g. unloading a model). Derived from LLM_BASE_URL by dropping the /v1 suffix.
OLLAMA_URL = os.getenv("OLLAMA_URL", LLM_BASE_URL.rsplit("/v1", 1)[0])

# --- HTTP server (the FastAPI wrapper in main.py) ---
# Where this service listens. The orchestrator and other modules call it here.
# Port 8001 keeps it clear of the STT service (which uses 8000).
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# --- Local multimodal (audio-capable) models — served directly via transformers,
# NOT through Ollama. Ollama doesn't support audio input yet (as of 2026-07),
# even though these models' own weights do, so they need their own serving path.
# Lazy-loaded on first /chat_audio request for a given model, not at startup;
# only one is held in memory at a time (see multimodal.MultimodalManager).
MULTIMODAL_GEMMA_MODEL_ID = os.getenv("MULTIMODAL_GEMMA_MODEL_ID", "google/gemma-4-E4B-it")
# Gemma 4's audio support only exists on E2B/E4B/12B ("Unified", encoder-free)
# -- NOT the 26B-A4B or 31B dense variants, which are image/video/text only.
# 12B is the largest audio-capable one; same architecture family as E4B, so
# it reuses GemmaAudioModel (see multimodal.py) rather than a new class.
MULTIMODAL_GEMMA_12B_MODEL_ID = os.getenv("MULTIMODAL_GEMMA_12B_MODEL_ID", "google/gemma-4-12B-it")
MULTIMODAL_QWEN_MODEL_ID = os.getenv("MULTIMODAL_QWEN_MODEL_ID", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
DEFAULT_MULTIMODAL_MODEL = os.getenv("DEFAULT_MULTIMODAL_MODEL", "gemma-4-e4b")
MULTIMODAL_MAX_NEW_TOKENS = int(os.getenv("MULTIMODAL_MAX_NEW_TOKENS", "2048"))
