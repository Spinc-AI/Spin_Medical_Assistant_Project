"""Local multimodal (audio-capable) LLM -- Gemma 4 E4B served directly via
``transformers``, NOT through Ollama.

Ollama's API doesn't support audio input (as of 2026-07:
https://github.com/ollama/ollama/issues/11798), even though this model's own
weights do -- so this is a separate serving path from ``llm_client.py``/the
Ollama-backed ``/chat`` endpoint, used only for audio-in requests
(Orchestrator's BuAli "multimodal LLM mode").

Lazy-loaded singleton: there's currently only one local multimodal model, so
this mirrors the STT module's model-loading pattern but without its
multi-model swapping machinery -- nothing to swap between yet.
"""
import gc
import tempfile
import threading

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

import config

_lock = threading.Lock()
_model = None
_processor = None


def _ensure_loaded():
    global _model, _processor
    if _model is not None:
        return
    with _lock:
        if _model is not None:  # lost the race while waiting for the lock
            return
        _processor = AutoProcessor.from_pretrained(config.MULTIMODAL_MODEL_ID, padding_side="left")
        _model = AutoModelForMultimodalLM.from_pretrained(
            config.MULTIMODAL_MODEL_ID, device_map="auto", attn_implementation="sdpa"
        )


def chat_audio(audio: bytes, audio_format: str, system_prompt: str, user_text: str | None = None) -> str:
    """Generate a reply from the audio-capable model given raw audio bytes.

    `audio_format` is a file extension (e.g. "wav", "mp3") used only to give
    the temp file the right suffix so the processor's audio loader can infer
    the container format.
    """
    _ensure_loaded()
    content: list[dict] = []
    if user_text:
        content.append({"type": "text", "text": user_text})
    with tempfile.NamedTemporaryFile(suffix=f".{audio_format}") as f:
        f.write(audio)
        f.flush()
        content.append({"type": "audio", "url": f.name})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        with _lock:
            inputs = _processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(_model.device, dtype=_model.dtype)
            input_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                outputs = _model.generate(**inputs, max_new_tokens=config.MULTIMODAL_MAX_NEW_TOKENS)
            # skip_special_tokens=True (unlike the reference docs example) --
            # callers parse this as JSON, so stray special-token text would break that.
            return _processor.decode(outputs[0][input_len:], skip_special_tokens=True)


def unload():
    """Release the model from memory (e.g. to free VRAM for another workload)."""
    global _model, _processor
    with _lock:
        _model = None
        _processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
