"""Spin Orchestrator (v1 - if/else).

Reads an instruction (a JSON workflow) and runs it end to end:
    user  <->  [this FastAPI service]  <->  modules (STT, Core_LLM) over HTTP.

Flow for the `01_casebook` instruction:
  1. pick instruction + STT model + LLM model      (POST /session)
  2. this service checks the module servers are up and loads the chosen models
  3. hand it audio OR text                          (POST /run)
  4. if audio -> STT transcribes it -> text
  5. text -> Core_LLM, which fills the patient JSON form
  6. return the filled form

The `02_radiology_report_assist_stt` instruction is richer: audio is
transcribed by up to THREE independently-configurable STT engines ("stt_slot"
steps) — each slot is freely local or external ("openai:..."), with its own
model/key/base_url, chosen per session. An LLM then reconciles whichever
transcripts were actually produced into one corrected report. (An instruction
can still use the older, simpler shapes too — see step types below.)

Local vs. external ("api") is a runtime choice, via a prefix convention:
  - `stt_model` = "openai:" (optionally + a model, e.g. "openai:whisper-1")
    routes a generic "stt" step (e.g. 01_casebook's) to the external STT API
    instead of our local service.
  - Each entry in `stt_slots` (see "stt_slot" steps) works the same way per
    slot — its own `model` field follows the same "openai:" convention.
  - `llm_model` = "openai:<model>" (e.g. "openai:gpt-4o-mini") routes any
    "llm" step to the external API instead of Core_LLM.

External calls need credentials, but never have to be preset on the server —
pass them per-request (session default and/or per-/run override); each only
falls back to OPENAI_API_KEY/OPENAI_BASE_URL in .env if nothing else is given.
Only OpenAI-compatible APIs are supported this way (a differently-shaped
provider like Google Cloud Speech would need its own integration).

Everything talks over HTTP, so any module can be rewritten internally
(e.g. Python -> C++) without touching this file, as long as its API holds.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config (override via environment / .env)
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).parent
load_dotenv(HERE / ".env")  # reads a .env file next to this script if present
INSTRUCTIONS_DIR = pathlib.Path(os.getenv("INSTRUCTIONS_DIR", str(HERE / "instruction")))

# STT, LLM, and Orchestrator are permanently co-located on the same server.
STT_URL = os.getenv("STT_URL", "http://localhost:8000").rstrip("/")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8001").rstrip("/")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9000"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "120"))

# External ("api") provider fallback defaults — used only when a request
# doesn't supply its own credentials. Only OpenAI-compatible APIs are
# supported this way (same request/response shape our own services already
# use; a differently-shaped provider would need its own integration).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
API_PREFIX = "openai:"

# A second external provider shape, alongside the OpenAI-compatible one above:
# Google's Gemini generateContent API (contents/parts/inline_data), used for
# "llm_audio" steps against Gemini's audio-capable models (e.g. via GapGPT's
# Gemini-shaped endpoint) -- these don't speak OpenAI's chat/completions +
# input_audio shape, so they need their own request/response handling.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
GEMINI_PREFIX = "gemini:"
MAX_STT_SLOTS = 3


# ---------------------------------------------------------------------------
# Instructions - an instruction is just a folder with a core_instruction.json
# ---------------------------------------------------------------------------
def load_instructions() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not INSTRUCTIONS_DIR.exists():
        return out
    for folder in sorted(INSTRUCTIONS_DIR.iterdir()):
        core = folder / "core_instruction.json"
        if core.is_file():
            data = json.loads(core.read_text(encoding="utf-8"))
            data["_folder"] = str(folder)
            out[data["id"]] = data
    return out


def load_template(instruction: dict, name: Optional[str] = None) -> dict:
    """Load a JSON file from the instruction's own folder (default: its output template)."""
    name = name or instruction.get("output", {}).get("template")
    if not name:
        return {}
    path = pathlib.Path(instruction["_folder"]) / name
    return json.loads(path.read_text(encoding="utf-8"))


def instruction_uses(instruction: dict, use: str) -> bool:
    return any(step.get("use") == use for step in instruction.get("steps", []))


def stt_slot_indices(instruction: dict) -> list[int]:
    """Which slot indices (0-based) this instruction's "stt_slot" steps reference."""
    return sorted({step.get("slot", 0) for step in instruction.get("steps", [])
                  if step.get("use") == "stt_slot"})


INSTRUCTIONS = load_instructions()


# ---------------------------------------------------------------------------
# Local <-> API dispatch helpers
# ---------------------------------------------------------------------------
def is_api_model(model: Optional[str]) -> bool:
    """True if `model` (an stt_model/llm_model/slot model value) selects the external API."""
    return bool(model) and model.startswith(API_PREFIX)


def strip_api_prefix(model: Optional[str]) -> str:
    return model[len(API_PREFIX):] if model else ""


def is_gemini_model(model: Optional[str]) -> bool:
    """True if `model` selects Gemini's own (non-OpenAI-shaped) API directly."""
    return bool(model) and model.startswith(GEMINI_PREFIX)


def strip_gemini_prefix(model: Optional[str]) -> str:
    return model[len(GEMINI_PREFIX):] if model else ""


def is_cloud_model(model: Optional[str]) -> bool:
    """True if `model` routes to any external provider (OpenAI-shaped or Gemini),
    as opposed to a local (Ollama-served) model."""
    return is_api_model(model) or is_gemini_model(model)


# ---------------------------------------------------------------------------
# Module clients - thin wrappers over each module's HTTP API
# ---------------------------------------------------------------------------
def _client() -> httpx.Client:
    return httpx.Client(timeout=HTTP_TIMEOUT)


def _resolve_api_key(api_key: Optional[str]) -> str:
    """An explicit key (from the request) wins; otherwise fall back to .env."""
    key = api_key or OPENAI_API_KEY
    if not key:
        raise RuntimeError(
            "No API key available for this external call — pass it in POST /session or /run, "
            "or set OPENAI_API_KEY in the orchestrator's .env"
        )
    return key


def _resolve_base_url(base_url: Optional[str]) -> str:
    return (base_url or OPENAI_BASE_URL).rstrip("/")


def stt_health() -> bool:
    try:
        with _client() as c:
            return c.get(f"{STT_URL}/models").status_code == 200
    except httpx.HTTPError:
        return False


def stt_models() -> dict:
    with _client() as c:
        r = c.get(f"{STT_URL}/models")
    r.raise_for_status()
    return r.json()


def stt_load(model: str) -> None:
    with _client() as c:
        r = c.post(f"{STT_URL}/models/{model}/load")
    if r.status_code != 200:
        raise RuntimeError(f"STT load failed ({r.status_code}): {r.text}")


def stt_transcribe(audio: bytes, filename: str = "audio.wav", language: Optional[str] = None) -> str:
    """Transcribe with our own (local) STT service. Caller must have loaded the model already."""
    data = {"language": language} if language else None
    with _client() as c:
        r = c.post(f"{STT_URL}/transcribe", files={"file": (filename, audio)}, data=data)
    if r.status_code != 200:
        raise RuntimeError(f"STT transcribe failed ({r.status_code}): {r.text}")
    return r.json()["text"]


def stt_api_transcribe(audio: bytes, filename: str = "audio.wav", language: Optional[str] = None,
                        api_key: Optional[str] = None, base_url: Optional[str] = None,
                        model: Optional[str] = None) -> str:
    """Transcribe with an external STT API (OpenAI-compatible /audio/transcriptions)."""
    key = _resolve_api_key(api_key)
    url = _resolve_base_url(base_url)
    data = {"model": model or OPENAI_STT_MODEL}
    if language:
        data["language"] = language
    with _client() as c:
        r = c.post(
            f"{url}/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (filename, audio)},
            data=data,
        )
    if r.status_code != 200:
        raise RuntimeError(f"API STT failed ({r.status_code}): {r.text}")
    return r.json()["text"]


def stt_languages() -> dict:
    with _client() as c:
        r = c.get(f"{STT_URL}/languages")
    r.raise_for_status()
    return r.json()


def stt_unload() -> None:
    with _client() as c:
        c.post(f"{STT_URL}/models/unload")


def llm_health() -> bool:
    try:
        with _client() as c:
            return c.get(f"{LLM_URL}/").status_code == 200
    except httpx.HTTPError:
        return False


def llm_chat(messages: list[dict], model: Optional[str] = None,
             response_format: Optional[dict] = None) -> str:
    """Chat with our own (local) Core_LLM service."""
    payload: dict[str, Any] = {"messages": messages}
    if model:
        payload["model"] = model
    if response_format:
        payload["response_format"] = response_format
    with _client() as c:
        r = c.post(f"{LLM_URL}/chat", json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"LLM chat failed ({r.status_code}): {r.text}")
    return r.json()["reply"]


def llm_api_chat(messages: list[dict], model: str, response_format: Optional[dict] = None,
                  api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """Chat with an external LLM API (OpenAI-compatible /chat/completions)."""
    key = _resolve_api_key(api_key)
    url = _resolve_base_url(base_url)
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if response_format:
        payload["response_format"] = response_format
    with _client() as c:
        r = c.post(
            f"{url}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
    if r.status_code != 200:
        raise RuntimeError(f"API LLM chat failed ({r.status_code}): {r.text}")
    return r.json()["choices"][0]["message"]["content"]


# OpenAI's chat-completions audio input (input_audio content-part) only
# accepts these two container formats. Gemini's generateContent (inline_data)
# accepts a wider set. The local path (Core_LLM's /chat_audio, via
# transformers' own audio loader) is generally permissive -- unverified exact
# boundary, so treated the same as Gemini's set for now. Whichever applies is
# picked by which prefix the llm_model uses.
OPENAI_AUDIO_FORMATS = {"wav", "mp3"}
GEMINI_AUDIO_MIME_TYPES = {
    "wav": "audio/wav", "mp3": "audio/mp3", "aac": "audio/aac",
    "ogg": "audio/ogg", "flac": "audio/flac", "aiff": "audio/aiff",
}
LOCAL_AUDIO_FORMATS = set(GEMINI_AUDIO_MIME_TYPES)


def audio_format_from_filename(filename: str, model: Optional[str] = None) -> str:
    """Validate the uploaded audio's container format against whichever
    provider `model` selects, and return its extension (lowercase, no dot)."""
    ext = pathlib.Path(filename or "").suffix.lstrip(".").lower()
    if is_gemini_model(model):
        allowed = set(GEMINI_AUDIO_MIME_TYPES)
    elif is_api_model(model):
        allowed = OPENAI_AUDIO_FORMATS
    else:
        allowed = LOCAL_AUDIO_FORMATS
    if ext not in allowed:
        raise HTTPException(
            400,
            f"multimodal LLM mode only accepts {sorted(allowed)} audio for this provider "
            f"(got '{ext or 'unknown'}') — this is a provider API restriction, not ours.",
        )
    return ext


def llm_api_chat_audio(audio: bytes, audio_format: str, system_prompt: str, user_text: Optional[str],
                        model: str, response_format: Optional[dict] = None,
                        api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """Chat with an OpenAI-compatible external LLM API, feeding it audio directly (no STT step).

    Uses the OpenAI-compatible input_audio content-part shape. Only makes
    sense for a model that actually accepts audio input (e.g. an
    audio-preview-class model) — the caller is responsible for that choice;
    this function doesn't validate model capability.
    """
    content: list[dict] = [{
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(audio).decode("ascii"), "format": audio_format},
    }]
    if user_text:
        content.append({"type": "text", "text": user_text})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    return llm_api_chat(messages, model, response_format=response_format, api_key=api_key, base_url=base_url)


def _resolve_gemini_api_key(api_key: Optional[str]) -> str:
    key = api_key or GEMINI_API_KEY
    if not key:
        raise RuntimeError(
            "No Gemini API key available for this call — pass it in POST /session or /run "
            "(llm_api_key), or set GEMINI_API_KEY in the orchestrator's .env"
        )
    return key


def _resolve_gemini_base_url(base_url: Optional[str]) -> str:
    return (base_url or GEMINI_BASE_URL).rstrip("/")


def llm_gemini_chat_audio(audio: bytes, audio_format: str, system_prompt: str, user_text: Optional[str],
                          model: str, json_response: bool = True,
                          api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """Chat with Gemini's own generateContent API, feeding it audio directly.

    Uses Gemini's native request shape (contents/parts/inline_data +
    systemInstruction) — NOT the OpenAI-compatible shape, since Gemini's
    audio-capable ("live"/"native-audio-dialog") models don't speak that.
    `base_url` should point at wherever this Gemini-shaped endpoint actually
    lives (Google's own API, or a proxy like GapGPT that exposes one) —
    verify the exact path/auth against your provider before relying on this
    in production; this follows Google's own documented v1beta shape.

    The instructions go in BOTH `system_instruction` AND as the first text
    part of the user turn — some proxies drop `system_instruction` silently,
    which leaves the model with nothing but the audio and no task, so it
    falls back to a generic "transcribe and translate" response instead of
    the requested JSON. Putting the instructions in the user turn too means
    they survive even when that happens.
    """
    key = _resolve_gemini_api_key(api_key)
    url = _resolve_gemini_base_url(base_url)
    mime_type = GEMINI_AUDIO_MIME_TYPES.get(audio_format, f"audio/{audio_format}")
    instructions = system_prompt + (f"\n\n{user_text}" if user_text else "")
    parts: list[dict] = [
        {"text": instructions},
        {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(audio).decode("ascii")}},
    ]
    payload: dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": parts}],
    }
    if json_response:
        payload["generationConfig"] = {"response_mime_type": "application/json"}
    with _client() as c:
        r = c.post(f"{url}/models/{model}:generateContent",
                   headers={"x-goog-api-key": key}, json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini LLM call failed ({r.status_code}): {r.text}")
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"unexpected Gemini response shape: {data}")


def llm_local_chat_audio(audio: bytes, audio_format: str, system_prompt: str,
                         user_text: Optional[str], filename: str = "audio.wav") -> str:
    """Chat with our own (local) Core_LLM service's audio-capable model.

    Unlike the local *text* path (llm_chat -> /chat, Ollama-backed), this
    goes to Core_LLM's separate /chat_audio endpoint, served directly via
    transformers -- Ollama doesn't support audio input. There's currently
    only one local multimodal model, so no model name is passed; whatever
    llm_model string the caller used is irrelevant here (any non-"openai:"/
    "gemini:" model just means "use the local audio-capable model").
    """
    with _client() as c:
        r = c.post(
            f"{LLM_URL}/chat_audio",
            files={"file": (filename, audio)},
            data={"system_prompt": system_prompt, "text": user_text or ""},
        )
    if r.status_code != 200:
        raise RuntimeError(f"Local multimodal LLM call failed ({r.status_code}): {r.text}")
    return r.json()["reply"]


def llm_chat_audio(audio: bytes, audio_format: str, system_prompt: str, user_text: Optional[str],
                   model: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """Dispatch an audio-input LLM call to whichever provider `model` selects:
    "openai:..." (OpenAI-shaped), "gemini:..." (Gemini's own shape), or
    anything else -> our local Core_LLM service's audio-capable model."""
    if is_gemini_model(model):
        return llm_gemini_chat_audio(audio, audio_format, system_prompt, user_text,
                                     strip_gemini_prefix(model), api_key=api_key, base_url=base_url)
    if is_api_model(model):
        return llm_api_chat_audio(audio, audio_format, system_prompt, user_text,
                                  strip_api_prefix(model), response_format={"type": "json_object"},
                                  api_key=api_key, base_url=base_url)
    return llm_local_chat_audio(audio, audio_format, system_prompt, user_text,
                                filename=f"audio.{audio_format}")


def chat(messages: list[dict], model: Optional[str], response_format: Optional[dict] = None,
         api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """Dispatch to the local Core_LLM or an external API, based on `model`.

    A model name prefixed with API_PREFIX ("openai:gpt-4o-mini") is routed to
    the external API (with the prefix stripped); anything else goes to our
    local Core_LLM service.
    """
    if is_api_model(model):
        return llm_api_chat(messages, strip_api_prefix(model), response_format=response_format,
                            api_key=api_key, base_url=base_url)
    return llm_chat(messages, model=model, response_format=response_format)


def llm_unload(model: Optional[str] = None) -> None:
    if is_api_model(model):
        return  # external API — nothing local to unload
    params = {"model": model} if model else {}
    with _client() as c:
        c.post(f"{LLM_URL}/unload", params=params)


# ---------------------------------------------------------------------------
# STT slots — up to MAX_STT_SLOTS independently-configurable STT engines
# ---------------------------------------------------------------------------
class SttSlotConfig(BaseModel):
    """One STT engine's config: local model name, or "openai:<model>" for cloud."""
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    language: Optional[str] = None  # overrides the session/run language for this slot only


def transcribe_slot(audio: bytes, slot: SttSlotConfig, default_language: Optional[str]) -> str:
    """Run one STT slot: loads+calls the local service, or calls the external API."""
    language = slot.language or default_language
    if is_api_model(slot.model):
        return stt_api_transcribe(audio, language=language, api_key=slot.api_key,
                                  base_url=slot.base_url, model=strip_api_prefix(slot.model) or None)
    stt_load(slot.model)  # STT holds one model at a time — (re)load right before use
    return stt_transcribe(audio, language=language)


# ---------------------------------------------------------------------------
# Session state (single active session for now - this is a v1 scaffold)
# ---------------------------------------------------------------------------
class Session(BaseModel):
    instruction: str
    stt_model: str
    llm_model: str
    language: Optional[str] = None      # default STT language for this session's audio steps
    stt_api_key: Optional[str] = None   # default key for an external STT call this session
    stt_base_url: Optional[str] = None  # default base URL for an external STT call
    llm_api_key: Optional[str] = None   # default key for an external LLM call this session
    llm_base_url: Optional[str] = None  # default base URL for an external LLM call
    stt_slots: Optional[list[Optional[SttSlotConfig]]] = None  # up to 3 independent STT engine configs; a slot may be null (unused)
    stt_mode: str = "separate"          # "separate" (STT then LLM) or "multimodal" (audio -> LLM directly, no STT)
    stt_ready: bool = False
    llm_ready: bool = False


SESSION: Optional[Session] = None

_SECRET_FIELDS = {"stt_api_key", "llm_api_key"}


def _redact_session(session: Session) -> dict:
    data = session.model_dump(exclude=_SECRET_FIELDS)
    if data.get("stt_slots"):
        data["stt_slots"] = [
            ({k: v for k, v in slot.items() if k != "api_key"} if slot is not None else None)
            for slot in data["stt_slots"]
        ]
    return data


# ---------------------------------------------------------------------------
# Pipeline - read the instruction's steps and run them (the "if/else")
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> Any:
    """Pull a JSON object out of the LLM reply (tolerant of code fences / prose)."""
    text = text.strip()
    if "```" in text:
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in LLM reply")
    return json.loads(text[start:end + 1])


def run_instruction(instruction: dict, *, audio: Optional[bytes], text: Optional[str],
                    stt_model: str, llm_model: str, language: Optional[str] = None,
                    stt_api_key: Optional[str] = None, stt_base_url: Optional[str] = None,
                    llm_api_key: Optional[str] = None, llm_base_url: Optional[str] = None,
                    stt_slots: Optional[list[Optional[SttSlotConfig]]] = None,
                    stt_mode: str = "separate", audio_filename: Optional[str] = None) -> dict:
    """Execute an instruction's steps in order.

    Each step's output is stored under its declared "output" name. An "llm"
    step reads its input from "inputs" (a list of prior output names) if
    given — using whichever of those are actually present (a slot may have
    been skipped) — otherwise falls back to the single most recent
    transcript/text (this keeps simple single-STT instructions like
    01_casebook unchanged). An "llm" step's own JSON reply is merged with any
    outputs it lists under "include_in_output" (e.g. to echo the raw
    transcripts alongside the LLM's verdict) to form the instruction's final
    result.

    Step "use" values:
      "stt"       -- generic; respects stt_model's local-vs-"openai:" choice.
      "stt_local" -- explicit; always local, regardless of stt_model.
      "stt_api"   -- explicit; always the external API, regardless of stt_model.
      "stt_slot"  -- reads its config from stt_slots[step["slot"]] (0-based);
                    skipped if that slot isn't configured. Independently
                    local or "openai:..." per slot.
      "llm"       -- respects llm_model's local-vs-"openai:" choice.
      "llm_audio" -- skips STT entirely; sends the audio straight to the LLM
                    (as an OpenAI-compatible input_audio content part). Only
                    works with an "openai:"-prefixed llm_model whose provider
                    actually accepts audio input — Ollama-served local models
                    can't (see AUDIO_INPUT_FORMATS / audio_format_from_filename).

    A step may also declare "run_when_stt_mode": [...] — a list of the
    `stt_mode` values it should run under ("separate" and/or "multimodal").
    Omitting the key means "always run", so instructions that don't use
    stt_mode at all (e.g. 01_casebook) are unaffected.

    Credentials may be None (falls back to the orchestrator's .env default).
    """
    outputs: dict[str, str] = {}
    last_transcript = text
    stt_slots = stt_slots or []

    for step in instruction.get("steps", []):
        allowed_modes = step.get("run_when_stt_mode")
        if allowed_modes is not None and stt_mode not in allowed_modes:
            continue
        use = step.get("use")

        if use == "stt":
            if audio is None:
                continue  # text was supplied -> skip transcription
            if is_api_model(stt_model):
                result = stt_api_transcribe(audio, language=language, api_key=stt_api_key,
                                            base_url=stt_base_url, model=strip_api_prefix(stt_model) or None)
            else:
                result = stt_transcribe(audio, language=language)
            outputs[step.get("output", "transcript")] = result
            last_transcript = result

        elif use == "stt_local":
            if audio is None:
                continue
            result = stt_transcribe(audio, language=language)
            outputs[step.get("output", "transcript_local")] = result
            last_transcript = result

        elif use == "stt_api":
            if audio is None:
                continue
            result = stt_api_transcribe(audio, language=language, api_key=stt_api_key, base_url=stt_base_url)
            outputs[step.get("output", "transcript_api")] = result
            last_transcript = result

        elif use == "stt_slot":
            if audio is None:
                continue
            idx = step.get("slot", 0)
            if idx >= len(stt_slots) or stt_slots[idx] is None:
                continue  # this slot wasn't configured for this run -> skip it
            result = transcribe_slot(audio, stt_slots[idx], language)
            outputs[step.get("output", f"transcript_{idx + 1}")] = result
            last_transcript = result

        elif use == "llm_audio":
            if audio is None:
                continue
            audio_format = audio_format_from_filename(audio_filename or "", model=llm_model)
            llm_template = load_template(instruction, step.get("template"))
            system_prompt = (step.get("system_prompt", "")
                             + "\n\nJSON template to fill:\n"
                             + json.dumps(llm_template, ensure_ascii=False, indent=2))
            try:
                reply = llm_chat_audio(
                    audio, audio_format, system_prompt, text, llm_model,
                    api_key=llm_api_key, base_url=llm_base_url,
                )
            except Exception as exc:
                raise HTTPException(502, f"LLM call failed: {exc}")
            try:
                result = _extract_json(reply)
            except ValueError:
                raise HTTPException(502, f"LLM did not return valid JSON:\n{reply}")

            include = step.get("include_in_output", [])
            merged = {name: outputs[name] for name in include if name in outputs}
            merged.update(result)
            return merged

        elif use == "llm":
            inputs = step.get("inputs")
            if inputs:
                present = [name for name in inputs if name in outputs]
                if not present:
                    raise HTTPException(400, f"none of this step's inputs were produced: {inputs}")
                labels = step.get("input_labels", {})
                user_content = "\n\n".join(
                    f"{labels.get(name, name)}:\n{outputs[name]}" for name in present
                )
            else:
                if not last_transcript:
                    raise HTTPException(400, "no text available to give the LLM")
                user_content = last_transcript

            llm_template = load_template(instruction, step.get("template"))
            messages = [
                {"role": "system",
                 "content": step.get("system_prompt", "")
                 + "\n\nJSON template to fill:\n"
                 + json.dumps(llm_template, ensure_ascii=False, indent=2)},
                {"role": "user", "content": user_content},
            ]
            try:
                reply = chat(messages, model=llm_model, response_format={"type": "json_object"},
                            api_key=llm_api_key, base_url=llm_base_url)
            except Exception as exc:
                raise HTTPException(502, f"LLM call failed: {exc}")
            try:
                result = _extract_json(reply)
            except ValueError:
                raise HTTPException(502, f"LLM did not return valid JSON:\n{reply}")

            include = step.get("include_in_output", [])
            merged = {name: outputs[name] for name in include if name in outputs}
            merged.update(result)
            return merged

    raise HTTPException(500, "instruction produced no output")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="Spin Orchestrator")


class SessionRequest(BaseModel):
    instruction: str
    stt_model: str = ""                 # used by "stt"/"stt_local" steps; irrelevant for pure "stt_slot" instructions
    llm_model: str
    language: Optional[str] = None      # e.g. "fa" or "en" — default STT language, see GET /languages
    stt_api_key: Optional[str] = None   # default key for an external STT call this session
    stt_base_url: Optional[str] = None  # default base URL for an external STT call
    llm_api_key: Optional[str] = None   # default key for an external LLM call this session
    llm_base_url: Optional[str] = None  # default base URL for an external LLM call
    stt_slots: Optional[list[Optional[SttSlotConfig]]] = None  # up to 3 independent STT engine configs; a slot may be null (unused)
    stt_mode: str = "separate"          # "separate" (STT then LLM) or "multimodal" (audio -> LLM directly, no STT)


@app.get("/instructions")
def list_instructions():
    """List the instructions this orchestrator knows about."""
    return {"instructions": [
        {"id": i["id"], "name": i.get("name"), "description": i.get("description")}
        for i in INSTRUCTIONS.values()
    ]}


@app.get("/instructions/{instruction_id}")
def get_instruction(instruction_id: str):
    """Full instruction detail (input.accepts, output.type, etc.) for building a UI.

    Adds computed hints so a client doesn't need to re-parse `steps`:
      "stt_model_is_choice" -- stt_model is a genuine local-vs-"openai:" pick
                               (the instruction has a generic "stt" step).
      "always_uses_stt_api" -- the instruction calls the external STT API
                               unconditionally (an explicit "stt_api" step),
                               regardless of stt_model.
      "stt_slot_count"      -- how many independent STT slots (0-3) this
                               instruction's "stt_slot" steps use; 0 if none.
      "supports_multimodal_llm" -- the instruction has an "llm_audio" step,
                               i.e. it can skip STT and give a cloud LLM the
                               audio directly (stt_mode="multimodal").
    """
    if instruction_id not in INSTRUCTIONS:
        raise HTTPException(404, f"unknown instruction '{instruction_id}'")
    instruction = INSTRUCTIONS[instruction_id]
    data = {k: v for k, v in instruction.items() if k != "_folder"}
    data["stt_model_is_choice"] = instruction_uses(instruction, "stt")
    data["always_uses_stt_api"] = instruction_uses(instruction, "stt_api")
    slots = stt_slot_indices(instruction)
    data["stt_slot_count"] = (max(slots) + 1) if slots else 0
    data["supports_multimodal_llm"] = instruction_uses(instruction, "llm_audio")
    return data


@app.get("/models")
def list_models():
    """Proxy STT's available local models (for building a model picker)."""
    try:
        return stt_models()
    except Exception as exc:
        raise HTTPException(502, f"Could not fetch models from STT: {exc}")


@app.get("/health")
def health():
    return {"orchestrator": "ok", "stt": stt_health(), "llm": llm_health(),
            "openai_default_key_configured": bool(OPENAI_API_KEY)}


@app.get("/languages")
def list_languages():
    """Proxy STT's supported language codes (for building a language picker)."""
    try:
        return stt_languages()
    except Exception as exc:
        raise HTTPException(502, f"Could not fetch languages from STT: {exc}")


@app.get("/status")
def status():
    if SESSION is None:
        return {"active": False}
    return {"active": True, **_redact_session(SESSION)}


@app.post("/session")
def start_session(req: SessionRequest):
    """Choose instruction + models, bring the models up, report status.

    No external credential has to be configured on the server — pass it here
    (or per-call in /run) instead. Session start only checks that the module
    SERVERS this session actually needs are reachable; a missing key only
    surfaces when a call that needs it runs. For "stt_slot"-based instructions,
    local models are loaded per-step during /run (not eagerly here), since
    different slots may need different local models in sequence.
    """
    global SESSION
    if req.instruction not in INSTRUCTIONS:
        raise HTTPException(404, f"unknown instruction '{req.instruction}'")
    instruction = INSTRUCTIONS[req.instruction]

    if req.stt_mode == "multimodal":
        if not instruction_uses(instruction, "llm_audio"):
            raise HTTPException(400, f"instruction '{req.instruction}' has no multimodal-LLM "
                                     "path (no 'llm_audio' step)")
        if not is_cloud_model(req.llm_model) and not llm_health():
            raise HTTPException(503, f"LLM server not reachable at {LLM_URL} (needed for the local "
                                     "multimodal model's /chat_audio endpoint)")
        # No STT involved at all in this mode -- nothing to check/load here.
        SESSION = Session(instruction=req.instruction, stt_model=req.stt_model, llm_model=req.llm_model,
                          language=req.language, llm_api_key=req.llm_api_key, llm_base_url=req.llm_base_url,
                          stt_mode=req.stt_mode, stt_ready=True, llm_ready=True)
        return status()

    slots_needed = stt_slot_indices(instruction)
    if slots_needed:
        # New-style: one or more independently-configured STT slots (0-3).
        slots = req.stt_slots or []
        if not any(idx < len(slots) and slots[idx] is not None for idx in slots_needed):
            raise HTTPException(400, "this instruction needs at least one configured STT slot "
                                     "(stt_slots)")
        if len(slots) > MAX_STT_SLOTS:
            raise HTTPException(400, f"at most {MAX_STT_SLOTS} STT slots are supported")
        any_local = any(not is_api_model(s.model) for s in slots if s is not None)
        if any_local and not stt_health():
            raise HTTPException(503, f"STT server not reachable at {STT_URL}")
        # Local models are (re)loaded per-slot, per-step during /run — not here.
    else:
        # Old-style: a single stt_model (generic "stt", or fixed "stt_local"/"stt_api").
        if instruction_uses(instruction, "stt_local") and is_api_model(req.stt_model):
            raise HTTPException(400, "this instruction has an unconditional local STT step "
                                     "— stt_model must be a real local model, not 'openai:...'")
        needs_local_stt = not is_api_model(req.stt_model) or instruction_uses(instruction, "stt_local")
        if needs_local_stt and not stt_health():
            raise HTTPException(503, f"STT server not reachable at {STT_URL}")
        if needs_local_stt:
            stt_load(req.stt_model)

    # A local llm_model needs Core_LLM reachable; an "openai:" one doesn't.
    if not is_api_model(req.llm_model) and not llm_health():
        raise HTTPException(503, f"LLM server not reachable at {LLM_URL}")

    SESSION = Session(instruction=req.instruction, stt_model=req.stt_model, llm_model=req.llm_model,
                      language=req.language, stt_api_key=req.stt_api_key, stt_base_url=req.stt_base_url,
                      llm_api_key=req.llm_api_key, llm_base_url=req.llm_base_url,
                      stt_slots=req.stt_slots, stt_mode=req.stt_mode, stt_ready=True, llm_ready=True)
    return status()


@app.post("/run")
def run(file: Optional[UploadFile] = File(default=None),
        text: Optional[str] = Form(default=None),
        language: Optional[str] = Form(default=None),
        stt_api_key: Optional[str] = Form(default=None),
        stt_base_url: Optional[str] = Form(default=None),
        llm_api_key: Optional[str] = Form(default=None),
        llm_base_url: Optional[str] = Form(default=None),
        stt_slots_json: Optional[str] = Form(default=None)):
    """Run the active instruction on an audio file OR a text string.

    `language`, `stt_api_key`/`stt_base_url`, and `llm_api_key`/`llm_base_url`
    override the session's defaults for this call. `stt_slots_json` (a
    JSON-encoded array of {model, api_key?, base_url?, language?}, same shape
    as POST /session's `stt_slots`) overrides the session's slot configs for
    this call — multipart form fields can't carry nested JSON directly, hence
    the string-encoded form here.
    """
    if SESSION is None:
        raise HTTPException(409, "no active session - call POST /session first")
    if file is None and not text:
        raise HTTPException(400, "provide either an audio 'file' or 'text'")
    instruction = INSTRUCTIONS[SESSION.instruction]
    audio = file.file.read() if file is not None else None

    stt_slots = SESSION.stt_slots
    if stt_slots_json:
        try:
            stt_slots = [SttSlotConfig(**s) if s is not None else None
                        for s in json.loads(stt_slots_json)]
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"invalid stt_slots_json: {exc}")

    try:
        result = run_instruction(
            instruction, audio=audio, text=text,
            stt_model=SESSION.stt_model, llm_model=SESSION.llm_model,
            language=language or SESSION.language,
            stt_api_key=stt_api_key or SESSION.stt_api_key,
            stt_base_url=stt_base_url or SESSION.stt_base_url,
            llm_api_key=llm_api_key or SESSION.llm_api_key,
            llm_base_url=llm_base_url or SESSION.llm_base_url,
            stt_slots=stt_slots,
            stt_mode=SESSION.stt_mode,
            audio_filename=file.filename if file is not None else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, str(exc))
    return {"instruction": SESSION.instruction, "result": result}


@app.post("/session/unload")
def unload():
    """Unload the models from the modules, then drop the active session."""
    global SESSION
    llm_model = SESSION.llm_model if SESSION else None
    try:
        stt_unload()
    except Exception:
        pass  # best-effort: module may already be down
    try:
        llm_unload(llm_model)
    except Exception:
        pass
    SESSION = None
    return {"active": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator:app", host=HOST, port=PORT)
