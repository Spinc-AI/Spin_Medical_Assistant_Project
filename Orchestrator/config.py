"""Orchestrator config (override via environment / .env)."""
import os
import pathlib

from dotenv import load_dotenv

HERE = pathlib.Path(__file__).parent
load_dotenv(HERE / ".env")

# STT, Core_LLM, and Orchestrator are permanently co-located on the same server.
STT_URL = os.getenv("STT_URL", "http://localhost:8000").rstrip("/")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8001").rstrip("/")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9000"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "600"))  # a large local model's reply can take a while

# External (OpenAI-compatible) API — only relevant to a pipeline that chooses a
# "openai:<model>" model for itself (see module_clients.is_api_model). This is
# never something a caller of the orchestrator's own API can select; it's a
# fallback credential for whichever pipeline code picks a cloud model.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
API_PREFIX = "openai:"
