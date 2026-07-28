"""Desktop demo client for STT, Core_LLM, and the Orchestrator.

One tab per service. Every model selector offers a **connection-type choice**:
"Remote Local Model" (talks to our deployed service, as before) or
"Custom Cloud API" (talks straight to an OpenAI-compatible cloud API from
THIS app — no dependency on our servers at all for that call).

On the STT and Core_LLM tabs the choice sits above Host/Port, since it decides
whether Host/Port even matters. The Orchestrator always needs its own server
(coordinating a multi-step instruction is literally its job), so instead the
choice sits separately before each of its two models (STT / LLM) — they can
even use different cloud providers/keys, since STT and LLM external calls are
fully independent.

The Orchestrator tab's input/output layout is built dynamically from
GET /instructions/{id} (input.accepts, output.type), so it adapts
automatically as new instructions are added.

Run:  python app.py
(Needs: pip install -r requirements.txt. On Linux, tkinter itself is a system
package: sudo apt install python3-tk)
"""
import json
import os
import queue
import re
import tempfile
import threading
import wave
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import requests
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement

try:
    import sounddevice as sd
    MIC_ERROR = None
except Exception as exc:  # missing PortAudio / no audio device — keep the app usable
    sd = None
    MIC_ERROR = exc

DEFAULT_HOST = "94.184.177.150"
TIMEOUT_SHORT = 15
TIMEOUT_LOAD = 900   # model loading / inference can be slow -- must exceed Orchestrator's own
                      # internal HTTP_TIMEOUT (see Orchestrator/.env), or this becomes the
                      # bottleneck instead: a big local model (e.g. aya-expanse:32b) can take
                      # several minutes to reconcile a BuAli report, and Orchestrator's timeout
                      # to Core_LLM needs raising to match (default 120s is too short for that).
AUDIO_FILETYPES = [("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")]
MIC_SAMPLE_RATE = 16000  # matches the STT service's target rate

# The two connection-type choices offered by every model selector.
REMOTE_LOCAL_LABEL = "Remote Local Model"
CUSTOM_CLOUD_LABEL = "Custom Cloud API"
DEFAULT_CLOUD_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CLOUD_STT_MODEL = "whisper-1"
DEFAULT_CLOUD_LLM_MODEL = "gpt-4o-mini"
FALLBACK_LANGUAGES = {"fa": "Persian", "en": "English"}  # used before/without a server round-trip
MAX_STT_SLOTS = 3  # mirrors the orchestrator's MAX_STT_SLOTS — multi-STT instructions (e.g. radiology)
# Core_LLM's registered local models (Core_LLM/deployment/model.py) -- all served
# directly via transformers, not Ollama. One model is loaded at a time and serves
# BOTH text-only chat and (if it supports_audio) BuAli's multimodal/hybrid pipelines,
# so this single list covers every "Remote Local Model" dropdown in the app.
LOCAL_LLM_MODELS = ["aya-expanse-8b", "aya-expanse-32b", "gemma-4-31b",
                    "gemma-4-e4b", "gemma-4-12b", "qwen3-omni-30b"]
# Subset that actually accepts audio input -- used to filter the dropdown down
# to valid choices whenever the pipeline mode needs an audio-capable model.
# qwen3-omni-30b: best tested option for Persian audio (see Core_LLM's README).
LOCAL_AUDIO_MODELS = ["gemma-4-e4b", "gemma-4-12b", "qwen3-omni-30b"]

# The Orchestrator's per-instruction STT pipeline choice: "separate" (STT-slots
# only), "multimodal" (audio straight to an audio-capable LLM, no STT), or
# "hybrid" (both -- STT slot(s) run AND the LLM hears the audio directly, with
# the STT transcript(s) folded in as reference material) -- see
# supports_multimodal_llm from GET /instructions/{id}.
SEPARATE_STT_LABEL = "Separate STT model(s)"
MULTIMODAL_LLM_LABEL = "Multimodal LLM mode"
HYBRID_LABEL = "Hybrid (STT + Multimodal LLM)"


def cloud_transcribe(base_url, api_key, model, audio_bytes, filename, language=None):
    """Transcribe directly against an OpenAI-compatible /audio/transcriptions endpoint."""
    data = {"model": model or DEFAULT_CLOUD_STT_MODEL}
    if language:
        data["language"] = language
    r = requests.post(
        f"{base_url.rstrip('/')}/audio/transcriptions",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": (filename, audio_bytes)},
        data=data,
        timeout=TIMEOUT_LOAD,
    )
    r.raise_for_status()
    return r.json()["text"]


def cloud_chat(base_url, api_key, model, messages):
    """Chat directly against an OpenAI-compatible /chat/completions endpoint."""
    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model or DEFAULT_CLOUD_LLM_MODEL, "messages": messages},
        timeout=TIMEOUT_LOAD,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run_bg(fn, *args, **kwargs):
    """Fire fn(*args, **kwargs) on a background thread so the GUI never freezes."""
    threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()


class ModeSelector(ttk.Frame):
    """Label + dropdown choosing "Remote Local Model" vs "Custom Cloud API"."""

    def __init__(self, parent, label_text):
        super().__init__(parent)
        ttk.Label(self, text=label_text).grid(row=0, column=0, sticky="w")
        self.mode = tk.StringVar(value=REMOTE_LOCAL_LABEL)
        self.box = ttk.Combobox(self, textvariable=self.mode, width=20, state="readonly",
                                values=[REMOTE_LOCAL_LABEL, CUSTOM_CLOUD_LABEL])
        self.box.grid(row=0, column=1, sticky="w", padx=(6, 0))

    def is_cloud(self):
        return self.mode.get() == CUSTOM_CLOUD_LABEL


class CloudFieldsFrame(ttk.Frame):
    """Cloud model / API key / base URL fields. Grid this frame as one unit
    and show/hide it as a whole via grid()/grid_remove()."""

    def __init__(self, parent, default_model):
        super().__init__(parent)
        ttk.Label(self, text="Cloud model:").grid(row=0, column=0, sticky="w")
        self.model = tk.StringVar(value=default_model)
        ttk.Entry(self, textvariable=self.model, width=18).grid(row=0, column=1, sticky="w", padx=(0, 12))

        ttk.Label(self, text="API key:").grid(row=0, column=2, sticky="w")
        self.api_key = tk.StringVar()
        ttk.Entry(self, textvariable=self.api_key, show="*", width=24).grid(row=0, column=3, sticky="w")

        ttk.Label(self, text="Base URL:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.base_url = tk.StringVar(value=DEFAULT_CLOUD_BASE_URL)
        ttk.Entry(self, textvariable=self.base_url, width=40).grid(
            row=1, column=1, columnspan=3, sticky="w", pady=(4, 0))


class SttSlotWidget(ttk.Frame):
    """One independently local-or-cloud STT engine slot: an explicit "Use this
    slot" checkbox, a mode selector, and local dropdown / cloud fields. Up to
    MAX_STT_SLOTS of these are used together for multi-STT instructions (e.g.
    the radiology one) — each can be a totally different engine/provider."""

    def __init__(self, parent, label_text, local_models_getter, on_refresh=None):
        super().__init__(parent)
        self._local_models_getter = local_models_getter  # callable -> current local model list
        self._on_refresh = on_refresh                    # callable -> re-fetch from the server

        self.enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Use this slot", variable=self.enabled,
                        command=self._update_visibility).grid(row=0, column=0, sticky="w")

        self.mode = ModeSelector(self, label_text)
        self.mode.grid(row=0, column=1, columnspan=3, sticky="w", padx=(10, 0))
        self.mode.mode.trace_add("write", lambda *a: self._update_visibility())

        self.local_model = tk.StringVar()
        self.local_box = ttk.Combobox(self, textvariable=self.local_model, width=18, state="readonly")
        self.refresh_btn = ttk.Button(self, text="Refresh", command=self._refresh_clicked)

        self.cloud = CloudFieldsFrame(self, DEFAULT_CLOUD_STT_MODEL)

        self._update_visibility()

    def refresh_local_models(self):
        models = self._local_models_getter()
        self.local_box["values"] = models
        if self.local_model.get() not in models and models:
            self.local_model.set(models[0])

    def _refresh_clicked(self):
        if self._on_refresh:
            self._on_refresh()

    def _update_visibility(self):
        if not self.enabled.get():
            self.local_box.grid_remove()
            self.refresh_btn.grid_remove()
            self.cloud.grid_remove()
            return
        if self.mode.is_cloud():
            self.local_box.grid_remove()
            self.refresh_btn.grid_remove()
            self.cloud.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        else:
            self.cloud.grid_remove()
            self.local_box.grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.refresh_btn.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=(4, 0))

    def is_cloud(self):
        return self.mode.is_cloud()

    def effective_model(self):
        if self.is_cloud():
            return "openai:" + self.cloud.model.get().strip()
        return self.local_model.get().strip()

    def as_slot_config(self):
        """Return a dict matching the backend's stt_slots entry shape, or None
        if this slot is unchecked (skipped)."""
        if not self.enabled.get():
            return None
        cfg = {"model": self.effective_model()}
        if self.is_cloud():
            key = self.cloud.api_key.get().strip()
            base = self.cloud.base_url.get().strip()
            if key:
                cfg["api_key"] = key
            if base:
                cfg["base_url"] = base
        return cfg


class MicRecorder:
    """Records mono 16-bit PCM from the default microphone until stopped."""

    def __init__(self, samplerate=MIC_SAMPLE_RATE, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self._queue = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        self._queue.put(bytes(indata))

    def start(self):
        self._queue = queue.Queue()
        self._stream = sd.RawInputStream(
            samplerate=self.samplerate, channels=self.channels,
            dtype="int16", callback=self._callback,
        )
        self._stream.start()

    def stop_and_save(self):
        """Stop recording and write the captured audio to a temp .wav file. Returns its path."""
        self._stream.stop()
        self._stream.close()
        chunks = []
        while not self._queue.empty():
            chunks.append(self._queue.get())
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16 -> 2 bytes
            wf.setframerate(self.samplerate)
            wf.writeframes(b"".join(chunks))
        return path


class RecordButton(ttk.Button):
    """Toggle button: records from the mic, writes a temp WAV, and stores its
    path in `file_path_var` — the same variable a Browse button would set."""

    def __init__(self, parent, file_path_var):
        super().__init__(parent, text="Record mic", command=self._toggle)
        self.file_path_var = file_path_var
        self._recorder = None

    def _toggle(self):
        if self._recorder is None:
            self._start()
        else:
            self._stop()

    def _start(self):
        if sd is None:
            messagebox.showerror("Microphone", f"Microphone support unavailable: {MIC_ERROR}")
            return
        try:
            self._recorder = MicRecorder()
            self._recorder.start()
        except Exception as exc:
            messagebox.showerror("Microphone", f"Could not start recording: {exc}")
            self._recorder = None
            return
        self.config(text="Stop recording")

    def _stop(self):
        try:
            path = self._recorder.stop_and_save()
            self.file_path_var.set(path)
        except Exception as exc:
            messagebox.showerror("Microphone", f"Could not save recording: {exc}")
        finally:
            self._recorder = None
            self.config(text="Record mic")


class ConnectionBar(ttk.Frame):
    """Host/Port fields + a Check button + a colored connected/disconnected indicator."""

    def __init__(self, parent, default_port, health_path="/"):
        super().__init__(parent)
        self.health_path = health_path

        ttk.Label(self, text="Host:").grid(row=0, column=0, padx=(0, 4))
        self.host = tk.StringVar(value=DEFAULT_HOST)
        ttk.Entry(self, textvariable=self.host, width=16).grid(row=0, column=1, padx=(0, 8))

        ttk.Label(self, text="Port:").grid(row=0, column=2, padx=(0, 4))
        self.port = tk.StringVar(value=str(default_port))
        ttk.Entry(self, textvariable=self.port, width=6).grid(row=0, column=3, padx=(0, 8))

        ttk.Button(self, text="Check connection", command=self.check).grid(row=0, column=4, padx=(0, 8))
        self.indicator = ttk.Label(self, text="● unknown", foreground="gray")
        self.indicator.grid(row=0, column=5)

    @property
    def base_url(self):
        return f"http://{self.host.get().strip()}:{self.port.get().strip()}"

    def check(self):
        self._set("● checking...", "gray")
        run_bg(self._check_bg)

    def _check_bg(self):
        try:
            ok = requests.get(self.base_url + self.health_path, timeout=TIMEOUT_SHORT).status_code == 200
        except requests.RequestException:
            ok = False
        self.after(0, self._set, ("● connected" if ok else "● unreachable"),
                   ("green" if ok else "red"))

    def _set(self, text, color):
        self.indicator.config(text=text, foreground=color)


def error_detail(exc):
    """Best-effort extraction of a JSON {"detail": ...} body from a requests error."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            return resp.json().get("detail", str(exc))
        except ValueError:
            pass
    return str(exc)


# ---------------------------------------------------------------------------
# Transcript -> Word (.docx) export
# ---------------------------------------------------------------------------
# Persian/Arabic-block characters. A line containing any of these is treated
# as an RTL paragraph; Word's own bidi algorithm still renders embedded Latin
# words/numbers within it left-to-right, so mixed Persian/English lines are
# fine without per-run splitting.
_RTL_CHAR_RE = re.compile(r"[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]")

# Characters illegal in Windows filenames, plus "/" and ":" which show up in
# HF model ids (e.g. "MohammadReza-Halakoo/persian-whisper-large-v3...") and
# in our "openai:gpt-4o-mini" convention.
_FILENAME_UNSAFE_RE = re.compile(r'[\\/:*?"<>|]')


def _looks_rtl(text):
    return bool(_RTL_CHAR_RE.search(text))


def sanitize_filename_part(name):
    """Turn a model id/name into something safe to put in a filename."""
    if not name:
        return ""
    name = name[len("openai:"):] if name.startswith("openai:") else name
    name = _FILENAME_UNSAFE_RE.sub("-", name)
    return name.strip("-_")


def build_transcript_filename(audio_path, model_parts, ext=".docx"):
    """<audio-stem>_<model1>_<model2>_..._<modelN>.ext, skipping empty parts."""
    stem = os.path.splitext(os.path.basename(audio_path))[0] if audio_path else "transcript"
    parts = [stem] + [sanitize_filename_part(m) for m in model_parts if m]
    return "_".join(p for p in parts if p) + ext


def save_text_as_docx(text, path):
    """Write `text` to a .docx at `path`, one paragraph per line. Each line's
    direction (RTL/LTR) is auto-detected so mixed Persian/English transcripts
    render correctly in Word. python-docx has no high-level RTL API, so the
    <w:bidi>/<w:rtl> elements are added directly.
    """
    doc = Document()
    for line in (text or "").split("\n"):
        p = doc.add_paragraph()
        run = p.add_run(line)
        if _looks_rtl(line):
            p._p.get_or_add_pPr().append(OxmlElement("w:bidi"))
            run._r.get_or_add_rPr().append(OxmlElement("w:rtl"))
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    doc.save(path)


class TranscriptSaver(ttk.Frame):
    """'Save to: <dir>  [Browse...]  [Save Transcript (.docx)]' row.

    Defaults the save directory to the source audio's own directory; a
    Browse... button lets the user override it (the override sticks until
    the app restarts). `get_text_and_parts()` is supplied by the owning tab
    and returns (text_to_save, audio_path, [model names in filename order]).
    """

    def __init__(self, parent, get_text_and_parts):
        super().__init__(parent)
        self._get_text_and_parts = get_text_and_parts
        self._overridden = False
        self.save_dir = tk.StringVar()

        ttk.Label(self, text="Save to:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.save_dir, width=40, state="readonly").grid(
            row=0, column=1, sticky="w")
        ttk.Button(self, text="Browse...", command=self._browse).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Button(self, text="Save Transcript (.docx)", command=self._save).grid(
            row=0, column=3, sticky="w", padx=(8, 0))

    def note_audio_path(self, audio_path):
        """Called by the owning tab whenever a new audio file is selected/used."""
        if audio_path and not self._overridden:
            self.save_dir.set(os.path.dirname(os.path.abspath(audio_path)))

    def _browse(self):
        d = filedialog.askdirectory(title="Choose where to save the transcript")
        if d:
            self.save_dir.set(d)
            self._overridden = True

    def _save(self):
        text, audio_path, model_parts = self._get_text_and_parts()
        if not text or not text.strip():
            messagebox.showwarning("Save Transcript", "Nothing to save yet — run a transcription first.")
            return
        save_dir = self.save_dir.get().strip() or (
            os.path.dirname(os.path.abspath(audio_path)) if audio_path else os.getcwd()
        )
        filename = build_transcript_filename(audio_path, model_parts)
        path = os.path.join(save_dir, filename)
        try:
            os.makedirs(save_dir, exist_ok=True)
            save_text_as_docx(text, path)
            messagebox.showinfo("Save Transcript", f"Saved to:\n{path}")
        except OSError as exc:
            messagebox.showerror("Save Transcript", f"Could not save: {exc}")


class OutputBox(scrolledtext.ScrolledText):
    """A read-only text area used for API output."""

    def __init__(self, parent, height=12):
        super().__init__(parent, width=80, height=height, state="disabled", wrap="word")

    def write(self, text):
        self.config(state="normal")
        self.delete("1.0", tk.END)
        self.insert(tk.END, text)
        self.config(state="disabled")


# ---------------------------------------------------------------------------
# STT tab
# ---------------------------------------------------------------------------
class STTTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self._local_models = []

        self.mode = ModeSelector(self, "Connection type:")
        self.mode.grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 8))
        self.mode.mode.trace_add("write", lambda *a: self._update_mode_visibility())

        # --- Remote Local Model group: Host/Port + local model controls ---
        self.local_frame = ttk.Frame(self)
        self.conn = ConnectionBar(self.local_frame, default_port=8000, health_path="/models")
        self.conn.grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 8))

        ttk.Label(self.local_frame, text="Model:").grid(row=1, column=0, sticky="w")
        self.model = tk.StringVar()
        self.model_box = ttk.Combobox(self.local_frame, textvariable=self.model, width=18, state="readonly")
        self.model_box.grid(row=1, column=1, sticky="w")
        ttk.Button(self.local_frame, text="Refresh", command=self.refresh_models).grid(
            row=1, column=2, sticky="w", padx=4)
        ttk.Button(self.local_frame, text="Load", command=self.load_model).grid(row=1, column=3, sticky="w")
        ttk.Button(self.local_frame, text="Unload", command=self.unload_model).grid(
            row=1, column=4, sticky="w", padx=4)
        self.model_status = ttk.Label(self.local_frame, text="loaded: -")
        self.model_status.grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))

        # --- Custom Cloud API group ---
        self.cloud = CloudFieldsFrame(self, DEFAULT_CLOUD_STT_MODEL)

        # --- Always visible: language + audio input + action + output ---
        ttk.Label(self, text="Language:").grid(row=2, column=0, sticky="w", pady=(8, 4))
        self.language = tk.StringVar()
        self.language_box = ttk.Combobox(self, textvariable=self.language, width=14, state="readonly",
                                         values=[f"{c} - {n}" for c, n in FALLBACK_LANGUAGES.items()])
        self.language_box.current(0)
        self.language_box.grid(row=2, column=1, sticky="w", pady=(8, 4))
        self._language_codes = list(FALLBACK_LANGUAGES.keys())

        ttk.Label(self, text="Audio file:").grid(row=3, column=0, sticky="w")
        self.file_path = tk.StringVar()
        ttk.Entry(self, textvariable=self.file_path, width=45, state="readonly").grid(
            row=3, column=1, columnspan=2, sticky="w")
        ttk.Button(self, text="Browse...", command=self.browse).grid(row=3, column=3, sticky="w")
        RecordButton(self, self.file_path).grid(row=3, column=4, sticky="w", padx=4)
        ttk.Button(self, text="Transcribe", command=self.transcribe).grid(row=4, column=0, sticky="w", pady=10)

        ttk.Label(self, text="Output:").grid(row=5, column=0, sticky="nw")
        self.output = OutputBox(self)
        self.output.grid(row=6, column=0, columnspan=5, pady=4)

        self._last_audio_path = ""
        self._last_stt_model = ""
        self.saver = TranscriptSaver(self, self._transcript_parts)
        self.saver.grid(row=7, column=0, columnspan=5, sticky="w", pady=(4, 0))

        self._update_mode_visibility()
        self.refresh_models()
        self.refresh_languages()

    def _transcript_parts(self):
        return self.output.get("1.0", tk.END).strip(), self._last_audio_path, [self._last_stt_model]

    def _is_cloud(self):
        return self.mode.is_cloud()

    def _update_mode_visibility(self):
        """Remote Local Model: Host/Port + local model controls. Custom Cloud API: the
        opposite — and that path never touches Host/Port/our server at all."""
        if self._is_cloud():
            self.local_frame.grid_remove()
            self.cloud.grid(row=1, column=0, columnspan=5, sticky="w", pady=(0, 8))
        else:
            self.cloud.grid_remove()
            self.local_frame.grid(row=1, column=0, columnspan=5, sticky="w", pady=(0, 8))

    def refresh_models(self):
        run_bg(self._refresh_models_bg)

    def _refresh_models_bg(self):
        try:
            r = requests.get(f"{self.conn.base_url}/models", timeout=TIMEOUT_SHORT)
            r.raise_for_status()
            self._local_models = r.json().get("available", [])
            self.after(0, self._apply_models, r.json())
        except requests.RequestException as exc:
            self.after(0, messagebox.showerror, "STT", f"Could not fetch models: {error_detail(exc)}")

    def _apply_models(self, data):
        self.model_box["values"] = self._local_models
        loaded = data.get("loaded")
        self.model_status.config(text=f"loaded: {loaded or '-'}")
        if loaded:
            self.model.set(loaded)
        elif self._local_models:
            self.model_box.current(0)

    def refresh_languages(self):
        run_bg(self._refresh_languages_bg)

    def _refresh_languages_bg(self):
        try:
            r = requests.get(f"{self.conn.base_url}/languages", timeout=TIMEOUT_SHORT)
            r.raise_for_status()
            self.after(0, self._apply_languages, r.json())
        except requests.RequestException:
            pass  # keep the FALLBACK_LANGUAGES already seeded — no server needed for this

    def _apply_languages(self, data):
        available = data.get("available", {})
        if not available:
            return
        self._language_codes = list(available.keys())
        self.language_box["values"] = [f"{c} - {n}" for c, n in available.items()]
        default = data.get("default")
        idx = self._language_codes.index(default) if default in self._language_codes else 0
        self.language_box.current(idx)

    def _selected_language(self):
        idx = self.language_box.current()
        return self._language_codes[idx] if 0 <= idx < len(self._language_codes) else None

    def load_model(self):
        key = self.model.get()
        if not key:
            messagebox.showwarning("STT", "Pick a model first.")
            return
        self.model_status.config(text=f"loading {key}...")
        run_bg(self._load_model_bg, key)

    def _load_model_bg(self, key):
        try:
            r = requests.post(f"{self.conn.base_url}/models/{key}/load", timeout=TIMEOUT_LOAD)
            r.raise_for_status()
            self.after(0, self.model_status.config, {"text": f"loaded: {r.json().get('model')}"})
        except requests.RequestException as exc:
            self.after(0, messagebox.showerror, "STT", f"Load failed: {error_detail(exc)}")
            self.after(0, self.model_status.config, {"text": "loaded: -"})

    def unload_model(self):
        run_bg(self._unload_model_bg)

    def _unload_model_bg(self):
        try:
            r = requests.post(f"{self.conn.base_url}/models/unload", timeout=TIMEOUT_SHORT)
            r.raise_for_status()
            self.after(0, self.model_status.config, {"text": "loaded: -"})
        except requests.RequestException as exc:
            self.after(0, messagebox.showerror, "STT", f"Unload failed: {error_detail(exc)}")

    def browse(self):
        path = filedialog.askopenfilename(title="Choose an audio file", filetypes=AUDIO_FILETYPES)
        if path:
            self.file_path.set(path)

    def transcribe(self):
        path = self.file_path.get()
        if not path:
            messagebox.showwarning("STT", "Choose an audio file first.")
            return
        language = self._selected_language()
        self.output.write("transcribing...")
        self._last_audio_path = path
        self.saver.note_audio_path(path)
        if self._is_cloud():
            self._last_stt_model = "openai:" + self.cloud.model.get().strip()
            run_bg(self._transcribe_cloud_bg, path, language)
        else:
            self._last_stt_model = self.model.get().strip()
            run_bg(self._transcribe_bg, path, language)

    def _transcribe_bg(self, path, language):
        try:
            with open(path, "rb") as f:
                data = {"language": language} if language else None
                r = requests.post(f"{self.conn.base_url}/transcribe",
                                  files={"file": (os.path.basename(path), f)}, data=data, timeout=TIMEOUT_LOAD)
            r.raise_for_status()
            self.after(0, self.output.write, r.json().get("text", ""))
        except requests.RequestException as exc:
            self.after(0, self.output.write, f"Error: {error_detail(exc)}")

    def _transcribe_cloud_bg(self, path, language):
        """Calls the cloud API directly — no Host/Port, no our-server dependency."""
        api_key = self.cloud.api_key.get().strip()
        if not api_key:
            self.after(0, self.output.write, "Error: enter an API key first.")
            return
        try:
            with open(path, "rb") as f:
                text = cloud_transcribe(self.cloud.base_url.get().strip(), api_key,
                                        self.cloud.model.get().strip(), f.read(),
                                        os.path.basename(path), language)
            self.after(0, self.output.write, text)
        except requests.RequestException as exc:
            self.after(0, self.output.write, f"Error: {error_detail(exc)}")


# ---------------------------------------------------------------------------
# Core_LLM tab
# ---------------------------------------------------------------------------
class LLMTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)

        self.mode = ModeSelector(self, "Connection type:")
        self.mode.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.mode.mode.trace_add("write", lambda *a: self._update_mode_visibility())

        # --- Remote Local Model group ---
        self.local_frame = ttk.Frame(self)
        self.conn = ConnectionBar(self.local_frame, default_port=8001, health_path="/")
        self.conn.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(self.local_frame, text="Model:").grid(row=1, column=0, sticky="w")
        self.model = tk.StringVar(value=LOCAL_LLM_MODELS[0])
        ttk.Combobox(self.local_frame, textvariable=self.model, width=20,
                    values=LOCAL_LLM_MODELS).grid(row=1, column=1, sticky="w")
        self.unload_btn = ttk.Button(self.local_frame, text="Unload model", command=self.unload_model)
        self.unload_btn.grid(row=1, column=2, sticky="w", padx=4)

        # --- Custom Cloud API group ---
        self.cloud = CloudFieldsFrame(self, DEFAULT_CLOUD_LLM_MODEL)

        ttk.Label(self, text="Message:").grid(row=2, column=0, sticky="nw", pady=(10, 0))
        self.input = scrolledtext.ScrolledText(self, width=70, height=6, wrap="word")
        self.input.grid(row=2, column=1, columnspan=2, pady=(10, 0), sticky="w")

        ttk.Button(self, text="Send", command=self.send).grid(row=3, column=0, sticky="w", pady=10)

        ttk.Label(self, text="Output:").grid(row=4, column=0, sticky="nw")
        self.output = OutputBox(self)
        self.output.grid(row=5, column=0, columnspan=3, pady=4)

        self._update_mode_visibility()

    def _is_cloud(self):
        return self.mode.is_cloud()

    def _update_mode_visibility(self):
        if self._is_cloud():
            self.local_frame.grid_remove()
            self.cloud.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))
        else:
            self.cloud.grid_remove()
            self.local_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))

    def send(self):
        text = self.input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Core_LLM", "Type a message first.")
            return
        self.output.write("thinking...")
        if self._is_cloud():
            run_bg(self._send_cloud_bg, text)
        else:
            run_bg(self._send_bg, text)

    def _send_bg(self, text):
        payload = {"messages": [{"role": "user", "content": text}]}
        model = self.model.get().strip()
        if model:
            payload["model"] = model
        try:
            r = requests.post(f"{self.conn.base_url}/chat", json=payload, timeout=TIMEOUT_LOAD)
            r.raise_for_status()
            self.after(0, self.output.write, r.json().get("reply", ""))
        except requests.RequestException as exc:
            self.after(0, self.output.write, f"Error: {error_detail(exc)}")

    def _send_cloud_bg(self, text):
        """Calls the cloud API directly — no Host/Port, no our-server dependency."""
        api_key = self.cloud.api_key.get().strip()
        if not api_key:
            self.after(0, self.output.write, "Error: enter an API key first.")
            return
        try:
            reply = cloud_chat(self.cloud.base_url.get().strip(), api_key, self.cloud.model.get().strip(),
                               [{"role": "user", "content": text}])
            self.after(0, self.output.write, reply)
        except requests.RequestException as exc:
            self.after(0, self.output.write, f"Error: {error_detail(exc)}")

    def unload_model(self):
        run_bg(self._unload_model_bg)

    def _unload_model_bg(self):
        model = self.model.get().strip() or None
        try:
            r = requests.post(f"{self.conn.base_url}/unload",
                              params=({"model": model} if model else {}), timeout=TIMEOUT_SHORT)
            r.raise_for_status()
            self.after(0, messagebox.showinfo, "Core_LLM", "Model unloaded.")
        except requests.RequestException as exc:
            self.after(0, messagebox.showerror, "Core_LLM", f"Unload failed: {error_detail(exc)}")


# ---------------------------------------------------------------------------
# Orchestrator tab
# ---------------------------------------------------------------------------
class OrchestratorTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self._instruction_ids = []
        self._accepts = []
        self.language_var = tk.StringVar()
        self.language_box = None  # only created when the active instruction accepts audio
        self._language_codes = []
        self._language_labels = []
        self._language_default_idx = 0
        self._always_uses_stt_api = False   # instruction has an unconditional "stt_api" step
        self._stt_model_is_choice = True    # instruction has a generic "stt" step (local-vs-cloud matters)
        self._stt_slot_count = 0            # >0 for multi-STT instructions (e.g. radiology) — see stt_slot_widgets
        self._local_stt_models = []
        self._supports_multimodal_llm = False  # instruction has an "llm_audio" step (see pipeline dropdown)

        self.conn = ConnectionBar(self, default_port=9000, health_path="/health")
        self.conn.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

        ttk.Label(self, text="Instruction:").grid(row=1, column=0, sticky="w")
        self.instruction = tk.StringVar()
        self.instruction_box = ttk.Combobox(self, textvariable=self.instruction, width=28, state="readonly")
        self.instruction_box.grid(row=1, column=1, sticky="w")
        self.instruction_box.bind("<<ComboboxSelected>>", lambda e: self.on_instruction_change())
        ttk.Button(self, text="Refresh", command=self.refresh_instructions).grid(row=1, column=2, sticky="w", padx=4)

        # Only shown for instructions that support it (supports_multimodal_llm):
        # skip STT entirely and feed audio straight to a cloud audio-capable LLM.
        self.pipeline_label = ttk.Label(self, text="Pipeline:")
        self.pipeline_mode = tk.StringVar(value=SEPARATE_STT_LABEL)
        self.pipeline_box = ttk.Combobox(self, textvariable=self.pipeline_mode, width=22, state="readonly",
                                         values=[SEPARATE_STT_LABEL, MULTIMODAL_LLM_LABEL, HYBRID_LABEL])
        self.pipeline_label.grid(row=1, column=3, sticky="w", padx=(12, 0))
        self.pipeline_box.grid(row=1, column=4, sticky="w")
        self.pipeline_label.grid_remove()
        self.pipeline_box.grid_remove()
        self.pipeline_mode.trace_add("write", lambda *a: self._on_pipeline_mode_change())

        # --- STT section: source choice comes BEFORE the model itself, per-instruction ---
        self.stt_frame = ttk.Frame(self)
        self.stt_frame.grid(row=2, column=0, columnspan=4, sticky="w", pady=(10, 0))

        self.stt_mode = ModeSelector(self.stt_frame, "STT source:")
        self.stt_mode.mode.trace_add("write", lambda *a: self._update_stt_section())

        self.stt_info_label = ttk.Label(
            self.stt_frame, foreground="gray",
            text="(this instruction always uses BOTH local and cloud STT)")

        self.stt_local_model = tk.StringVar(value="whisper")
        self.stt_local_box = ttk.Combobox(self.stt_frame, textvariable=self.stt_local_model,
                                          width=18, state="readonly")
        self.stt_local_refresh_btn = ttk.Button(self.stt_frame, text="Refresh", command=self.refresh_models)

        self.stt_cloud = CloudFieldsFrame(self.stt_frame, DEFAULT_CLOUD_STT_MODEL)

        # Multi-STT instructions (stt_slot_count > 0, e.g. radiology): up to
        # MAX_STT_SLOTS independent slots instead of the single choice above.
        self.stt_slot_widgets = [
            SttSlotWidget(self.stt_frame, f"STT slot {i + 1} source:",
                         lambda: self._local_stt_models, on_refresh=self.refresh_models)
            for i in range(MAX_STT_SLOTS)
        ]

        # --- LLM section: same pattern, always a real choice (no "always both" case) ---
        self.llm_frame = ttk.Frame(self)
        self.llm_frame.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))

        self.llm_mode = ModeSelector(self.llm_frame, "LLM source:")
        self.llm_mode.grid(row=0, column=0, columnspan=4, sticky="w")
        self.llm_mode.mode.trace_add("write", lambda *a: self._update_llm_section())

        # One dropdown for every "Remote Local Model" case -- separate mode
        # (any of the 6), multimodal/hybrid mode (filtered to the audio-capable
        # 3 in _update_llm_section, since only one model is loaded at a time
        # and it serves both text and audio roles regardless of which endpoint
        # a request came in through).
        self.llm_local_model = tk.StringVar(value=LOCAL_LLM_MODELS[0])
        self.llm_local_box = ttk.Combobox(self.llm_frame, textvariable=self.llm_local_model,
                                          width=18, state="readonly", values=LOCAL_LLM_MODELS)

        self.llm_cloud = CloudFieldsFrame(self.llm_frame, DEFAULT_CLOUD_LLM_MODEL)
        self.multimodal_hint = ttk.Label(
            self.llm_frame, foreground="gray",
            text="Local: served via Core_LLM's /chat_audio (not Ollama, which can't take audio) — "
                 "qwen3-omni-30b is the best tested option for Persian. Cloud: for a Gemini audio "
                 "model (e.g. via GapGPT), prefix Cloud model with \"gemini:\" — e.g. "
                 "gemini:gemini-2.5-flash-preview-native-audio-dialog")

        ttk.Button(self, text="Start session", command=self.start_session).grid(row=4, column=0, sticky="w", pady=10)
        ttk.Button(self, text="Unload session", command=self.unload_session).grid(row=4, column=1, sticky="w")
        self.session_status = ttk.Label(self, text="session: none")
        self.session_status.grid(row=5, column=0, columnspan=4, sticky="w")

        # Rebuilt per-instruction: audio Browse and/or a text box, per input.accepts.
        self.input_area = ttk.Frame(self)
        self.input_area.grid(row=6, column=0, columnspan=4, sticky="w", pady=10)
        self.file_path = tk.StringVar()
        self.text_input = None

        ttk.Button(self, text="Run", command=self.run).grid(row=7, column=0, sticky="w", pady=(0, 10))

        ttk.Label(self, text="Output:").grid(row=8, column=0, sticky="nw")
        self.output = OutputBox(self, height=14)
        self.output.grid(row=9, column=0, columnspan=4, pady=4)

        self._last_audio_path = ""
        self._last_model_parts = []
        self.saver = TranscriptSaver(self, self._transcript_parts)
        self.saver.grid(row=10, column=0, columnspan=4, sticky="w", pady=(4, 0))

        self._update_stt_section()
        self._update_llm_section()
        self.refresh_instructions()
        self.refresh_languages()
        self.refresh_models()

    def _transcript_parts(self):
        return self.output.get("1.0", tk.END).strip(), self._last_audio_path, self._last_model_parts

    # --- STT section: multi-slot (if the instruction supports it), else the
    # older mode-selector-or-"always both" single-choice UI. ---
    def _is_multimodal(self):
        return self._supports_multimodal_llm and self.pipeline_mode.get() == MULTIMODAL_LLM_LABEL

    def _is_hybrid(self):
        return self._supports_multimodal_llm and self.pipeline_mode.get() == HYBRID_LABEL

    def _uses_llm_audio(self):
        """Multimodal or hybrid -- either way the LLM needs audio-capable handling."""
        return self._is_multimodal() or self._is_hybrid()

    def _on_pipeline_mode_change(self):
        self._update_stt_section()
        self._update_llm_section()

    def _update_stt_section(self):
        for w in (self.stt_mode, self.stt_info_label, self.stt_local_box,
                 self.stt_local_refresh_btn, self.stt_cloud):
            w.grid_remove()
        for w in self.stt_slot_widgets:
            w.grid_remove()

        if self._is_multimodal():
            return  # no STT at all in this mode — audio goes straight to the LLM

        if self._stt_slot_count > 0:
            for i in range(self._stt_slot_count):
                self.stt_slot_widgets[i].grid(row=i, column=0, columnspan=4, sticky="w",
                                              pady=(0 if i == 0 else 6, 0))
        elif self._stt_model_is_choice:
            self.stt_mode.grid(row=0, column=0, columnspan=4, sticky="w")
            if self.stt_mode.is_cloud():
                self.stt_cloud.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
            else:
                self.stt_local_box.grid(row=1, column=0, sticky="w", pady=(4, 0))
                self.stt_local_refresh_btn.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=(4, 0))
        else:
            # This instruction has an unconditional "stt_local" + "stt_api" pair —
            # no real choice, both always run.
            self.stt_info_label.grid(row=0, column=0, columnspan=4, sticky="w")
            self.stt_local_box.grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.stt_local_refresh_btn.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=(4, 0))
            self.stt_cloud.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))

    def _update_llm_section(self):
        # Multimodal/hybrid modes can use EITHER a local audio-capable model
        # (Core_LLM's /chat_audio) or a cloud one ("openai:"/"gemini:" prefix)
        # — Ollama itself still can't take audio, but Core_LLM's own local
        # models are served via transformers directly, not Ollama, so local
        # is a real option here.
        self.llm_mode.box.config(state="readonly")

        # Same one dropdown either way -- just filter its values down to the
        # audio-capable subset when the pipeline mode needs that, so you can't
        # pick a combination that'll just error out server-side.
        wanted_values = LOCAL_AUDIO_MODELS if self._uses_llm_audio() else LOCAL_LLM_MODELS
        if list(self.llm_local_box["values"]) != wanted_values:
            self.llm_local_box["values"] = wanted_values
            if self.llm_local_model.get() not in wanted_values:
                self.llm_local_model.set(wanted_values[0])

        if self.llm_mode.is_cloud():
            self.llm_local_box.grid_remove()
            self.llm_cloud.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        else:
            self.llm_cloud.grid_remove()
            self.llm_local_box.grid(row=1, column=0, sticky="w", pady=(4, 0))

        if self._uses_llm_audio():
            self.multimodal_hint.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        else:
            self.multimodal_hint.grid_remove()

    def _stt_cloud_active(self):
        """Whether the STT cloud fields are relevant right now (chosen, or always-on)."""
        return (not self._stt_model_is_choice) or self.stt_mode.is_cloud()

    def _llm_cloud_active(self):
        return self.llm_mode.is_cloud()

    def _effective_stt_model(self):
        if self._stt_model_is_choice and self.stt_mode.is_cloud():
            return "openai:" + self.stt_cloud.model.get().strip()
        return self.stt_local_model.get().strip()

    def _effective_llm_model(self):
        if self.llm_mode.is_cloud():
            model = self.llm_cloud.model.get().strip()
            # Multimodal LLM mode may need a non-OpenAI-shaped provider (e.g.
            # "gemini:..." for Gemini's own audio API) — an explicit prefix
            # passes through untouched; anything else still defaults to "openai:".
            if model.startswith("openai:") or model.startswith("gemini:"):
                return model
            return "openai:" + model
        return self.llm_local_model.get().strip()

    def refresh_models(self):
        run_bg(self._refresh_models_bg)

    def _refresh_models_bg(self):
        try:
            r = requests.get(f"{self.conn.base_url}/models", timeout=TIMEOUT_SHORT)
            r.raise_for_status()
            self._local_stt_models = r.json().get("available", [])
            self.after(0, self._apply_stt_model_options)
        except requests.RequestException as exc:
            self.after(0, messagebox.showerror, "Orchestrator",
                      f"Could not fetch local STT models: {error_detail(exc)}")

    def _apply_stt_model_options(self):
        self.stt_local_box["values"] = self._local_stt_models
        if self.stt_local_model.get() not in self._local_stt_models and self._local_stt_models:
            self.stt_local_model.set(self._local_stt_models[0])
        for w in self.stt_slot_widgets:
            w.refresh_local_models()

    def refresh_languages(self):
        run_bg(self._refresh_languages_bg)

    def _refresh_languages_bg(self):
        try:
            r = requests.get(f"{self.conn.base_url}/languages", timeout=TIMEOUT_SHORT)
            r.raise_for_status()
            data = r.json()
            available = data.get("available", {})
            self._language_codes = list(available.keys())
            self._language_labels = [f"{c} - {n}" for c, n in available.items()]
            default = data.get("default")
            self._language_default_idx = (
                self._language_codes.index(default) if default in self._language_codes else 0
            )
            self.after(0, self._apply_language_widget)
        except requests.RequestException:
            pass  # language picker is optional — STT falls back to its own default

    def _apply_language_widget(self):
        if self.language_box is not None:
            self.language_box["values"] = self._language_labels
            if self._language_labels:
                self.language_box.current(self._language_default_idx)

    def _selected_language(self):
        if self.language_box is None:
            return None
        idx = self.language_box.current()
        return self._language_codes[idx] if 0 <= idx < len(self._language_codes) else None

    def refresh_instructions(self):
        run_bg(self._refresh_instructions_bg)

    def _refresh_instructions_bg(self):
        try:
            r = requests.get(f"{self.conn.base_url}/instructions", timeout=TIMEOUT_SHORT)
            r.raise_for_status()
            self.after(0, self._apply_instructions, r.json().get("instructions", []))
        except requests.RequestException as exc:
            self.after(0, messagebox.showerror, "Orchestrator", f"Could not fetch instructions: {error_detail(exc)}")

    def _apply_instructions(self, items):
        self._instruction_ids = [i["id"] for i in items]
        self.instruction_box["values"] = [f"{i['id']} - {i.get('name', '')}" for i in items]
        if self._instruction_ids:
            self.instruction_box.current(0)
            self.on_instruction_change()

    def on_instruction_change(self):
        idx = self.instruction_box.current()
        if 0 <= idx < len(self._instruction_ids):
            run_bg(self._fetch_instruction_bg, self._instruction_ids[idx])

    def _fetch_instruction_bg(self, instr_id):
        try:
            r = requests.get(f"{self.conn.base_url}/instructions/{instr_id}", timeout=TIMEOUT_SHORT)
            r.raise_for_status()
            self.after(0, self._rebuild_input_area, r.json())
        except requests.RequestException as exc:
            self.after(0, messagebox.showerror, "Orchestrator", f"Could not fetch instruction: {error_detail(exc)}")

    def _rebuild_input_area(self, instruction):
        """Show a Browse button and/or a text box, matching input.accepts for this instruction."""
        for w in self.input_area.winfo_children():
            w.destroy()
        self.text_input = None
        self.language_box = None
        self._accepts = instruction.get("input", {}).get("accepts", [])
        self._always_uses_stt_api = instruction.get("always_uses_stt_api", False)
        self._stt_model_is_choice = instruction.get("stt_model_is_choice", True)
        self._stt_slot_count = min(instruction.get("stt_slot_count", 0), MAX_STT_SLOTS)
        self._supports_multimodal_llm = instruction.get("supports_multimodal_llm", False)
        if self._supports_multimodal_llm:
            self.pipeline_label.grid()
            self.pipeline_box.grid()
        else:
            self.pipeline_mode.set(SEPARATE_STT_LABEL)
            self.pipeline_label.grid_remove()
            self.pipeline_box.grid_remove()
        self._apply_stt_model_options()
        self._update_stt_section()
        self._update_llm_section()

        row = 0
        if "audio" in self._accepts:
            self.file_path.set("")
            ttk.Label(self.input_area, text="Audio file:").grid(row=row, column=0, sticky="w")
            ttk.Entry(self.input_area, textvariable=self.file_path, width=40, state="readonly").grid(
                row=row, column=1, sticky="w")
            ttk.Button(self.input_area, text="Browse...", command=self.browse).grid(row=row, column=2, sticky="w")
            RecordButton(self.input_area, self.file_path).grid(row=row, column=3, sticky="w", padx=4)
            ttk.Label(self.input_area, text="Language:").grid(row=row, column=4, sticky="w", padx=(8, 0))
            self.language_box = ttk.Combobox(self.input_area, textvariable=self.language_var,
                                             width=14, state="readonly")
            self.language_box.grid(row=row, column=5, sticky="w")
            self._apply_language_widget()
            row += 1
        if "text" in self._accepts:
            ttk.Label(self.input_area, text="Text:").grid(row=row, column=0, sticky="nw")
            self.text_input = scrolledtext.ScrolledText(self.input_area, width=60, height=4, wrap="word")
            self.text_input.grid(row=row, column=1, columnspan=2, sticky="w")
            row += 1
        if not self._accepts:
            ttk.Label(self.input_area, text="(this instruction takes no direct input)").grid(row=0, column=0, sticky="w")

    def browse(self):
        path = filedialog.askopenfilename(title="Choose an audio file", filetypes=AUDIO_FILETYPES)
        if path:
            self.file_path.set(path)

    def start_session(self):
        idx = self.instruction_box.current()
        if idx < 0:
            messagebox.showwarning("Orchestrator", "Pick an instruction first.")
            return
        payload = {"instruction": self._instruction_ids[idx], "llm_model": self._effective_llm_model()}

        if self._is_multimodal():
            payload["stt_mode"] = "multimodal"
            key = self.llm_cloud.api_key.get().strip()
            if key:
                payload["llm_api_key"] = key
            base = self.llm_cloud.base_url.get().strip()
            if base:
                payload["llm_base_url"] = base
            self.session_status.config(text="starting session...")
            run_bg(self._start_session_bg, payload)
            return

        if self._is_hybrid():
            # Same as separate mode below (STT slots + generic LLM key/base
            # handling) -- just tags the session so Orchestrator ALSO feeds
            # the audio itself to the LLM, alongside whichever transcripts
            # the STT slot(s) below produce. No early return.
            payload["stt_mode"] = "hybrid"

        if self._stt_slot_count > 0:
            slots = [w.as_slot_config() for w in self.stt_slot_widgets[:self._stt_slot_count]]
            if not any(slots):
                messagebox.showwarning("Orchestrator", "Enable at least one STT slot.")
                return
            payload["stt_slots"] = slots
            payload["stt_model"] = ""  # unused by slot-based instructions
        else:
            payload["stt_model"] = self._effective_stt_model()
            if self._stt_cloud_active():
                key = self.stt_cloud.api_key.get().strip()
                if key:
                    payload["stt_api_key"] = key
                base = self.stt_cloud.base_url.get().strip()
                if base:
                    payload["stt_base_url"] = base

        if self._llm_cloud_active():
            key = self.llm_cloud.api_key.get().strip()
            if key:
                payload["llm_api_key"] = key
            base = self.llm_cloud.base_url.get().strip()
            if base:
                payload["llm_base_url"] = base
        self.session_status.config(text="starting session...")
        run_bg(self._start_session_bg, payload)

    def _start_session_bg(self, payload):
        try:
            r = requests.post(f"{self.conn.base_url}/session", json=payload, timeout=TIMEOUT_LOAD)
            r.raise_for_status()
            data = r.json()
            text = (f"session: {data.get('instruction')}  "
                    f"(stt={data.get('stt_model')}, llm={data.get('llm_model')})")
            self.after(0, self.session_status.config, {"text": text})
        except requests.RequestException as exc:
            self.after(0, self.session_status.config, {"text": f"session failed: {error_detail(exc)}"})

    def run(self):
        path = self.file_path.get() if "audio" in self._accepts else ""
        text = self.text_input.get("1.0", tk.END).strip() if self.text_input else ""
        language = self._selected_language()

        if not path and not text:
            messagebox.showwarning("Orchestrator", "Provide audio or text first.")
            return

        if self._is_multimodal():
            # No STT at all — audio goes straight to the LLM (local or cloud,
            # per _update_llm_section). Nothing here needs stt_slots_json.
            # (Hybrid mode falls through to the STT-slot logic below instead,
            # since it needs both stt_slots_json AND the audio file.)
            llm_key = self.llm_cloud.api_key.get().strip()
            llm_base = self.llm_cloud.base_url.get().strip()
            self.output.write("running...")
            self._last_audio_path = path
            self._last_model_parts = [self._effective_llm_model()]
            if path:
                self.saver.note_audio_path(path)
            run_bg(self._run_bg, path, text, language, "", "", llm_key, llm_base, None)
            return

        stt_key = stt_base = ""
        stt_slots_json = None
        stt_model_parts = []
        if self._stt_slot_count > 0:
            slots = [w.as_slot_config() for w in self.stt_slot_widgets[:self._stt_slot_count]]
            stt_slots_json = json.dumps(slots)
            stt_model_parts = [s.get("model") if s else None for s in slots]
        else:
            stt_key = self.stt_cloud.api_key.get().strip() if self._stt_cloud_active() else ""
            stt_base = self.stt_cloud.base_url.get().strip() if self._stt_cloud_active() else ""
            stt_model_parts = [self._effective_stt_model()]

        llm_key = self.llm_cloud.api_key.get().strip() if self._llm_cloud_active() else ""
        llm_base = self.llm_cloud.base_url.get().strip() if self._llm_cloud_active() else ""
        self.output.write("running...")
        self._last_audio_path = path
        self._last_model_parts = stt_model_parts + [self._effective_llm_model()]
        if path:
            self.saver.note_audio_path(path)
        run_bg(self._run_bg, path, text, language, stt_key, stt_base, llm_key, llm_base, stt_slots_json)

    def _run_bg(self, path, text, language, stt_key, stt_base, llm_key, llm_base, stt_slots_json):
        f = None
        try:
            files = None
            data = {}
            if path:
                f = open(path, "rb")
                files = {"file": (os.path.basename(path), f)}
            else:
                data["text"] = text
            if language:
                data["language"] = language
            if stt_key:
                data["stt_api_key"] = stt_key
            if stt_base:
                data["stt_base_url"] = stt_base
            if llm_key:
                data["llm_api_key"] = llm_key
            if llm_base:
                data["llm_base_url"] = llm_base
            if stt_slots_json:
                data["stt_slots_json"] = stt_slots_json
            r = requests.post(f"{self.conn.base_url}/run", files=files, data=data or None, timeout=TIMEOUT_LOAD)
            r.raise_for_status()
            pretty = json.dumps(r.json(), indent=2, ensure_ascii=False)
            self.after(0, self.output.write, pretty)
        except requests.RequestException as exc:
            self.after(0, self.output.write, f"Error: {error_detail(exc)}")
        finally:
            if f:
                f.close()

    def unload_session(self):
        run_bg(self._unload_session_bg)

    def _unload_session_bg(self):
        try:
            r = requests.post(f"{self.conn.base_url}/session/unload", timeout=TIMEOUT_SHORT)
            r.raise_for_status()
            self.after(0, self.session_status.config, {"text": "session: none"})
        except requests.RequestException as exc:
            self.after(0, messagebox.showerror, "Orchestrator", f"Unload failed: {error_detail(exc)}")


class DemoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spin Medical Assistant — Demo")
        self.geometry("900x760")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        notebook.add(STTTab(notebook), text="STT")
        notebook.add(LLMTab(notebook), text="Core_LLM")
        notebook.add(OrchestratorTab(notebook), text="Orchestrator")


if __name__ == "__main__":
    DemoApp().mainloop()
