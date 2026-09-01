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

On the **STT and Core_LLM tabs**, every model has a **"Connection type"**
choice, sitting **above Host/Port**: **Remote Local Model** (talks to *our*
deployed service) or **Custom Cloud API** (talks straight from this app to an
OpenAI-compatible cloud API — reveals **Cloud model** / **API key** /
**Base URL** fields, and that call never touches Host/Port at all). Pick
Custom Cloud API and Host/Port becomes irrelevant for that tab's action.

The **Orchestrator tab** has no such choice — it only ever talks to the
orchestrator itself at Host/Port; which model (local or cloud) a pipeline
uses internally is entirely that pipeline's own business.

## Tabs

**STT** — Connection type · (local) Host/Port + Model dropdown +
pick/load/unload, **or** (cloud) Cloud model/API key/Base URL · language
dropdown (seeded with `fa`/`en` up front, no server needed; refined from
`GET /languages` if reachable) · browse an audio file **or record from the
mic** · Transcribe · text output.

**Core_LLM** — Connection type · (local) Host/Port + Model + Unload, **or**
(cloud) Cloud model/API key/Base URL · multiline message box · Send · reply
output.

**Orchestrator** — Host/Port + connection check · a **Pipeline** dropdown
(fetched from `GET /pipelines` — id/name/description only, no model info) ·
a running **Conversation** transcript · a message box + **Send** · a
**Result** box for whatever structured JSON the pipeline hands back once it
has one. Picking a pipeline (or **New conversation**) starts fresh by posting
an empty history and showing the pipeline's opening reply. Every reply after
that is appended to a client-side `history` list and sent back on the next
call — the orchestrator itself holds no session state for this. There is no
model picker anywhere on this tab: which model backs a pipeline is chosen by
that pipeline's own code on the server, not by this UI or any other caller.

API key fields are never required to be preset on a server: on the STT/
Core_LLM tabs, type them in and they're sent with the request — a key is
required whenever Custom Cloud API is chosen there, since those calls go
straight from this app to the cloud API with no server-side fallback.

**Confirmed working cloud provider:** [GapGPT](https://gapgpt.app) (Iran-based,
Rial-payable, proxies GPT/Claude/Gemini through the exact OpenAI request
shape). For **Base URL** use `https://api.gapgpt.app/v1` (or
`https://api.gapapi.com/v1` for their external-CDN route), your GapGPT key as
**API key**, and any model they proxy as **Cloud model** — e.g. `gpt-4o`,
`gemini-2.5-pro`.

Any audio input (wherever a "Record mic" button appears) records mono 16kHz
PCM to a temp `.wav` file and uses it exactly like a browsed file.

Defaults assume all three services run on `localhost` (ports `8000` / `8001`
/ `9000`) — change Host/Port per tab to point elsewhere (e.g. a remote GPU
server). Host/Port is irrelevant on the STT/Core_LLM tabs whenever
Custom Cloud API is selected.

## Save Transcript (.docx)

The STT tab has a **Save Transcript (.docx)** row under Output: a "Save to:"
folder (defaults to the source audio file's own folder, sticky-overridable
via **Browse...**) and the save button itself. It writes whatever is
currently in Output to a Word document, auto-detecting RTL/LTR per line so
mixed Persian/English content (e.g. "بیمار MRI شد") reads correctly in Word —
no manual right-to-left toggling needed.

Filename: `<audio file name>_<model>.docx`. Saving before running a
transcription (or with an empty Output) shows a warning instead of writing an
empty file.
