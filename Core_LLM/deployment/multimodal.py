"""Local multimodal (audio-capable) LLMs -- served directly via ``transformers``,
NOT through Ollama.

Ollama's API doesn't support audio input (as of 2026-07:
https://github.com/ollama/ollama/issues/11798), even though these models'
own weights do -- so this is a separate serving path from ``llm_client.py``/
the Ollama-backed ``/chat`` endpoint, used only for audio-in requests
(Orchestrator's BuAli "multimodal LLM mode" / "hybrid" mode).

Two models are registered, each with its own loading/message-format quirks
(different transformers classes, different chat-template content-key
conventions) -- this mirrors the STT module's swappable BaseSTTModel/
ModelManager pattern, holding at most one model in memory at a time.
"""
import gc
import tempfile
import threading
from abc import ABC, abstractmethod

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

import config


class BaseMultimodalModel(ABC):
    """An audio-capable local model that can be loaded and chatted with."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._processor = None

    @abstractmethod
    def load(self):
        """Pull weights and processor into memory."""

    @abstractmethod
    def chat_audio(self, audio_path: str, system_prompt: str, user_text: str | None) -> str:
        """Return the model's text reply given a path to an audio file on disk."""

    def unload(self):
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GemmaAudioModel(BaseMultimodalModel):
    """Gemma 4 E4B via AutoModelForMultimodalLM."""

    def load(self):
        self._processor = AutoProcessor.from_pretrained(self.model_id, padding_side="left")
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id, device_map="auto", attn_implementation="sdpa"
        )

    def chat_audio(self, audio_path, system_prompt, user_text=None):
        content: list[dict] = []
        if user_text:
            content.append({"type": "text", "text": user_text})
        content.append({"type": "audio", "url": audio_path})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device, dtype=self._model.dtype)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=config.MULTIMODAL_MAX_NEW_TOKENS)
        # skip_special_tokens=True (unlike the reference docs example) -- callers
        # parse this as JSON, so stray special-token text would break that.
        return self._processor.decode(outputs[0][input_len:], skip_special_tokens=True)


class QwenOmniAudioModel(BaseMultimodalModel):
    """Qwen3-Omni, Thinker-only (text output, no speech generation -- we don't
    need audio-out, and this skips loading the Talker's audio-codec weights).

    Confirmed via PARSA-Bench (an independent Persian audio-language-model
    benchmark) as the strongest tested option for Persian ASR/understanding
    among locally-runnable models (0.358 WER, vs. 6-9 WER for Gemma-3n-class
    models) -- this is the model to reach for specifically for Persian audio.

    Note: transformers' own docs flag that MoE inference through `transformers`
    (as opposed to vLLM) can be slow. Fine for now since it matches Core_LLM's
    existing serving pattern; revisit if latency becomes a real problem.
    """

    def load(self):
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id)
        self._model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto"
        )

    def chat_audio(self, audio_path, system_prompt, user_text=None):
        user_content = [{"type": "audio", "path": audio_path}]
        if user_text:
            user_content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        inputs = self._processor.apply_chat_template(
            messages, load_audio_from_video=True, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt", padding=True,
        ).to(self._model.device)
        # Slicing off input_len (unlike the reference docs snippet, which decodes
        # the full sequence) so the reply doesn't echo the prompt back -- same
        # reasoning as GemmaAudioModel above.
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            text_ids = self._model.generate(**inputs, max_new_tokens=config.MULTIMODAL_MAX_NEW_TOKENS)
        return self._processor.batch_decode(
            text_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# ============================================================
# Registry
# ============================================================
MODEL_REGISTRY = {
    "gemma-4-e4b": (GemmaAudioModel, config.MULTIMODAL_GEMMA_MODEL_ID),
    "qwen3-omni-30b": (QwenOmniAudioModel, config.MULTIMODAL_QWEN_MODEL_ID),
}


class MultimodalManager:
    """Holds at most one loaded multimodal model, swapping as needed.

    A single lock guards both loading and generation so concurrent requests
    can't swap the model out from under an in-flight generation.
    """

    def __init__(self):
        self._current_key = None
        self._current_model = None
        self._lock = threading.Lock()

    def available(self):
        return list(MODEL_REGISTRY.keys())

    @property
    def loaded(self):
        return self._current_key

    def chat_audio(self, key: str, audio: bytes, audio_format: str,
                   system_prompt: str, user_text: str | None = None) -> str:
        if key not in MODEL_REGISTRY:
            raise KeyError(f"unknown multimodal model '{key}' -- available: {self.available()}")
        with self._lock:
            if self._current_key != key:
                if self._current_model is not None:
                    self._current_model.unload()
                cls, model_id = MODEL_REGISTRY[key]
                model = cls(model_id)
                model.load()
                self._current_model = model
                self._current_key = key
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}") as f:
                f.write(audio)
                f.flush()
                return self._current_model.chat_audio(f.name, system_prompt, user_text)

    def unload(self):
        with self._lock:
            if self._current_model is not None:
                self._current_model.unload()
                self._current_model = None
                self._current_key = None


MANAGER = MultimodalManager()
