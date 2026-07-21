# Demo App

A Tkinter desktop client for manually exercising **STT**, **Core_LLM**, and the
**Orchestrator** — one tab each, no code needed.

## Run
```bash
pip install -r requirements.txt
python app.py          # or: run.bat (Windows) / ./run.sh (Linux)
```
Linux only: tkinter and mic recording need system packages —
`sudo apt install python3-tk libportaudio2`.

## Connection type: Remote Local Model vs. Custom Cloud API

Every model has a **"Connection type"** choice: **Remote Local Model** (talks
to *our* deployed service, as before) or **Custom Cloud API** (talks
**straight from this app to an OpenAI-compatible cloud API** — reveals
**Cloud model** / **API key** / **Base URL** fields, and that call never
touches Host/Port or any of our servers at all).

- **STT and Core_LLM tabs**: the choice sits **above Host/Port**, since it
  decides whether Host/Port even matters — pick Custom Cloud API and
  Host/Port becomes irrelevant for that tab's action.
- **Orchestrator tab**: Host/Port always stays at the top — coordinating a
  multi-step instruction (e.g. transcribe-then-reconcile) is literally the
  orchestrator's job, so that part can't be bypassed without duplicating its
  logic here. Instead, the choice sits **separately before each of its two
  models** (STT source / LLM source) — they can even use **different cloud
  providers and API keys**, since the STT and LLM external calls are fully
  independent on the backend.

## Tabs

**STT** — Connection type · (local) Host/Port + Model dropdown +
pick/load/unload, **or** (cloud) Cloud model/API key/Base URL · language
dropdown (seeded with `fa`/`en` up front, no server needed; refined from
`GET /languages` if reachable) · browse an audio file **or record from the
mic** · Transcribe · text output.

**Core_LLM** — Connection type · (local) Host/Port + Model + Unload, **or**
(cloud) Cloud model/API key/Base URL · multiline message box · Send · reply
output.

**Orchestrator** — Host/Port + connection check · instruction dropdown ·
**STT source** (mode + local model dropdown or cloud fields) · **LLM source**
(same pattern) · Start/Unload session · **input area rebuilds itself** per the
selected instruction's `input.accepts` (audio Browse/Record + language
dropdown, and/or a text box, fetched via `GET /instructions/{id}`) · Run ·
JSON output. The language dropdown only appears when the instruction accepts
audio, since it only affects the STT step.

The STT side adapts to what the selected instruction actually supports
(`GET /instructions/{id}`'s `stt_model_is_choice` / `stt_slot_count`):

- **Single choice** (e.g. `01_casebook`) — one **STT source** selector:
  local model dropdown or cloud fields.
- **Multi-STT** (e.g. `02_radiology_report_assist_stt`) — up to **3
  independent STT slots**, each with its own **"Use this slot"** checkbox,
  its own local-vs-cloud choice, and its own model/key/base URL. Slots can
  mix freely — e.g. slot 1 = local `whisper`, slot 2 = OpenAI, slot 3 =
  disabled — and each cloud slot can be a **completely different provider**
  with its own key. Unchecked slots are simply skipped; the reconciling LLM
  step only sees whichever transcripts were actually produced. At least one
  slot must be enabled to start a session.

Every local STT model dropdown (the single-choice one and each slot's) has
its own **Refresh** button next to it — click it to (re)fetch the list from
`GET /models` on demand, rather than only relying on the automatic fetch at
tab-open. If the list stays empty, an error dialog now explains why (e.g.
the Orchestrator can't reach its configured `STT_URL`), instead of failing
silently.
- **Fixed, no choice** (older instructions with unconditional STT steps) — an
  info line plus both the local dropdown and cloud fields shown together,
  since both always run regardless of any selection.

### Pipeline: Separate STT model(s) vs. Multimodal LLM mode

For instructions that support it (BuAli does; `GET /instructions/{id}`'s
`supports_multimodal_llm`), a **"Pipeline:"** dropdown appears right next to
the Instruction dropdown:

- **Separate STT model(s)** (default) — the STT-slot pipeline described
  above: transcribe first, then the LLM reconciles the transcript(s).
- **Multimodal LLM mode** — skips STT entirely; the audio is sent straight to
  the LLM. Picking this mode hides the STT slot widgets; **LLM source stays a
  free choice** (Remote Local Model or Custom Cloud API), three provider
  shapes total:
  - **Remote Local Model** — Core_LLM's own audio-capable model (Gemma 4 E4B,
    served via a separate `/chat_audio` endpoint using `transformers`
    directly — **not** Ollama, which still can't take audio input at all).
    The Ollama-tag dropdown is replaced by a fixed label in this mode, since
    there's only one local multimodal model and its selection doesn't matter.
  - **Custom Cloud API**, Cloud model = `openai:<model>` (or no prefix,
    defaults to this) — OpenAI-compatible `input_audio`; accepts `.wav`/`.mp3`
    audio only.
  - **Custom Cloud API**, Cloud model = `gemini:<model>` — Google's own
    Gemini API shape (needed for their "live"/native-audio-dialog models,
    e.g. via GapGPT); accepts `.wav`/`.mp3`/`.aac`/`.ogg`/`.flac`/`.aiff`. A
    gray hint under the LLM fields reminds you of this once Custom Cloud API
    is chosen in this mode.

  Whichever path, an unsupported audio format is rejected with a clear error
  before any API call is made.

API key fields are never required to be preset on a server: type them in and
they're sent with the request. On the Orchestrator tab, STT's and LLM's keys
are session defaults that can be changed again before a later Run to override
them for that call; leaving either blank falls back to `OPENAI_API_KEY` in
the orchestrator's `.env`, if set. On the STT/Core_LLM tabs there is no
server-side fallback — a key is required whenever Custom Cloud API is chosen.

**Confirmed working cloud provider:** [GapGPT](https://gapgpt.app) (Iran-based,
Rial-payable, proxies GPT/Claude/Gemini through the exact OpenAI request
shape). For **Base URL** use `https://api.gapgpt.app/v1` (or
`https://api.gapapi.com/v1` for their external-CDN route), your GapGPT key as
**API key**, and any model they proxy as **Cloud model** — e.g. `gpt-4o`,
`gemini-2.5-pro`.

Any audio input (wherever a "Record mic" button appears) records mono 16kHz
PCM to a temp `.wav` file and uses it exactly like a browsed file.

Defaults assume all three services run on the same server
(`94.184.177.150`, ports `8000` / `8001` / `9000`) — change Host/Port per tab
to point elsewhere. Host/Port is irrelevant on the STT/Core_LLM tabs whenever
Custom Cloud API is selected.

The radiology instruction (`02_radiology_report_assist_stt` on disk) is now
shown in the UI as **"BuAli"** — only its display name changed, its `id` and
folder are the same.

## Save Transcript (.docx)

Both the STT tab and the Orchestrator tab have a **Save Transcript (.docx)**
row under Output: a "Save to:" folder (defaults to the source audio file's own
folder, sticky-overridable via **Browse...**) and the save button itself. It
writes whatever is currently in Output to a Word document, auto-detecting
RTL/LTR per line so mixed Persian/English content (e.g. "بیمار MRI شد") reads
correctly in Word — no manual right-to-left toggling needed.

Filename: `<audio file name>_<model 1>_<model 2>_..._<model N>.docx`, using
whichever model(s) actually produced the transcript, in order (STT slot(s)
first, then the LLM if one was involved) — e.g. for a BuAli run with all 3
STT slots filled: `case04_seamless_whisper_gpt-4o-mini.docx` (the `openai:`
provider prefix and any `/` in an HF model id are stripped/sanitized for the
filename). Empty/unused slots are simply skipped — in **Multimodal LLM
mode** there are no STT slots at all, so the filename is just
`<audio file name>_<llm model>.docx`. Saving before running a transcription
(or with an empty Output) shows a warning instead of writing an empty file.
