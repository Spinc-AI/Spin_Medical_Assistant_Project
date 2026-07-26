# Orchestrator

The brain of the assistant (v1, if/else). Talks to the user over HTTP and to the
modules (STT, Core_LLM) over their HTTP APIs. It reads an **instruction** (a JSON
workflow at `instruction/<NN_Name>/core_instruction.json`) and runs it: pick models
-> load them -> take audio or text -> (STT if audio) -> LLM fills the form.

## Run
```bash
pip install -r requirements.txt
./run.sh          # Linux;  run.bat on Windows
```
Serves on `0.0.0.0:9000` (docs at `/docs`). Set `STT_URL` / `LLM_URL` in `.env`.
The module servers must already be running/reachable.

## API
| Method & path | Purpose |
|---|---|
| `GET /instructions` | list available instructions |
| `GET /instructions/{id}` | full instruction detail (input.accepts, output.type) |
| `GET /models` | proxies STT's available local models |
| `GET /languages` | proxies STT's supported language codes |
| `GET /health` | orchestrator + STT + LLM reachability |
| `POST /session` | body `{instruction, stt_model, llm_model, language?, stt_api_key?, stt_base_url?, llm_api_key?, llm_base_url?, stt_slots?, stt_mode?}` -> load models, return status |
| `GET /status` | current session / chosen models (never echoes any `*_api_key`, incl. inside `stt_slots`) |
| `POST /run` | multipart: `file` (audio) **or** `text`, optional `language`/`stt_api_key`/`stt_base_url`/`llm_api_key`/`llm_base_url`/`stt_slots_json` override -> filled JSON |
| `POST /session/unload` | unload models on the modules, end the session |

`language` (e.g. `"fa"`/`"en"`) only matters when the input is audio — it's
forwarded to STT's `/transcribe`. Set a default in `/session`, optionally
override per call in `/run`.

`stt_api_key`/`stt_base_url` and `llm_api_key`/`llm_base_url` are for
external ("api") calls only — see below. They're **independent of each
other** (STT and LLM can use different providers/accounts). Same pattern: a
session default, optionally overridden per `/run` call.

## Typical sequence
```bash
curl -X POST http://193.93.169.134:9000/session -H "Content-Type: application/json"   -d '{"instruction":"01_casebook","stt_model":"whisper","llm_model":"aya-expanse"}'

curl -X POST http://193.93.169.134:9000/run -F "file=@clip.wav"
# or:  curl -X POST http://193.93.169.134:9000/run -F "text=patient is Ali, 45, male ..."

curl -X POST http://193.93.169.134:9000/session/unload
```

## Instructions
Each instruction is a folder under `instruction/` holding a `core_instruction.json`
(the workflow) plus its template(s).

| Folder | Displayed name | Does |
|---|---|---|
| `01_Casebook` | Casebook | audio-or-text -> LLM fills a patient form |
| `02_Radiology_Report_Assist_STT` | **BuAli** | audio -> up to **three** independently local-or-cloud STT transcripts -> LLM reconciles whichever were produced into one corrected report |

The folder name is the on-disk id and doesn't change; `"name"` inside each
`core_instruction.json` is what the UI actually shows — the radiology
instruction's `"name"` is `"BuAli"`.

## Local vs. API — models and providers

`stt_model` and `llm_model` support the same convention: prefix with `openai:`
to route that step to the external API instead of the local service — e.g.
`"stt_model": "openai:whisper-1"`, `"llm_model": "openai:gpt-4o-mini"`.
Leaving off the prefix (`"whisper"`, `"aya-expanse"`) uses the local module.
This only applies to a **generic** step (`"use": "stt"` or `"use": "llm"` in
the instruction's JSON) — `01_Casebook`'s STT step is generic, so its
`stt_model` choice matters there.

### Multi-STT instructions: independent slots

An instruction can instead declare up to **3 independent STT slots**
(`"use": "stt_slot"`, `"slot": 0`/`1`/`2` in its JSON — this is what
`02_Radiology_Report_Assist_STT` uses). Each slot is configured separately via
`stt_slots` — a list of `{model, api_key?, base_url?, language?}` — and can
independently be local (a real model name) or external (`"openai:<model>"`),
with its own provider/account. **1 to 3 slots may be configured**; an
unconfigured or `null` slot is simply skipped, and the reconciling LLM step
only sees whichever transcripts were actually produced.

```json
{
  "stt_slots": [
    {"model": "whisper"},
    {"model": "openai:whisper-1", "api_key": "sk-...", "base_url": "https://api.gapgpt.app/v1"},
    null
  ]
}
```
(here slot 3 is skipped — only transcript_1 and transcript_2 get produced and reconciled)

Local STT models are (re)loaded per-slot, per-call — since STT only holds one
model in memory at a time, using two different local models across slots
means it swaps between them during a single `/run` (slower, but correct).

Only OpenAI-compatible APIs are supported this way for now (same shape our own
services already use). A differently-shaped provider (e.g. Google Cloud Speech)
would need its own integration, not just a new key.

### Multimodal LLM mode: skip STT, feed audio to the LLM directly

BuAli (`02_Radiology_Report_Assist_STT`) also supports a second pipeline via
`"stt_mode": "multimodal"` (default is `"separate"`, i.e. the STT-slots
pipeline above) — no STT service call at all; the audio goes straight to an
audio-capable LLM.

Three providers are supported, picked by `llm_model`'s prefix (or lack of
one) — their call shapes are too different to share one path:

| `llm_model` | Provider shape | Example |
|---|---|---|
| *(no prefix)* | **Local** — Core_LLM's `POST /chat_audio`, one of its registered models served directly via `transformers` (NOT Ollama — see below) | `"gemma-4-e4b"` or `"qwen3-omni-30b"` |
| `openai:<model>` | OpenAI-compatible `POST /chat/completions` + `input_audio` content part | `"openai:gpt-4o-audio-preview"` |
| `gemini:<model>` | Google's own `generateContent` API (`contents`/`parts`/`inline_data` + `systemInstruction`) | `"gemini:gemini-2.5-flash-preview-native-audio-dialog"` |

**Why the local path isn't Ollama:** Ollama doesn't support audio input at
all ([ollama/ollama#11798](https://github.com/ollama/ollama/issues/11798)),
even though these models' own weights do. So Core_LLM runs a second, separate
model-serving path just for this — `Core_LLM/deployment/multimodal.py` — with
its own small registry (mirrors the STT module's swappable-model pattern,
one model in memory at a time), exposed at `POST /chat_audio`. **The
`llm_model` string here is the registry key that gets forwarded to Core_LLM**
(unlike the old single-model version, it now matters which one you pick):

| Key | Model | Notes |
|---|---|---|
| `gemma-4-e4b` (default) | `google/gemma-4-E4B-it` | Lighter, faster |
| `gemma-4-12b` | `google/gemma-4-12B-it` | Largest **audio-capable** Gemma 4 (26B-A4B/31B have no audio input at all — image/video/text only) |
| `qwen3-omni-30b` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | **Best tested option for Persian** — confirmed via [PARSA-Bench](https://arxiv.org/html/2603.14456), an independent Persian audio-LM benchmark (0.358 WER vs. 6-9 for Gemma-3n-class models). Needs real VRAM headroom. |

This needs real GPU VRAM headroom; see `Core_LLM/deployment/requirements.txt`
for the extra dependencies (`torch`, `transformers`, `accelerate`, etc.) this
path pulls in beyond Core_LLM's normal Ollama-only footprint.

Requirements, enforced at `POST /session` and again at `POST /run`:
- For the two cloud prefixes, that provider's model must actually accept
  audio input. For the local path, Core_LLM must be reachable (checked the
  same way a normal local `llm_model` would be), and the key must be one
  Core_LLM actually has registered (`GET /chat_audio/models` on Core_LLM).
- The audio file's container format must be accepted by the chosen
  provider — `openai:` only accepts `.wav`/`.mp3`; `gemini:` and the local
  path accept a wider set (`.wav`/`.mp3`/`.aac`/`.ogg`/`.flac`/`.aiff` —
  the local boundary is unverified against `transformers`' actual audio
  loader, treated the same as Gemini's for now). Anything else is rejected
  with a 400 before any API call is made.
- `llm_base_url` must point at wherever the chosen cloud provider's endpoint
  actually lives — for `gemini:`, that's Google's own API by default
  (`GEMINI_BASE_URL`, defaults to `https://generativelanguage.googleapis.com/v1beta`)
  or a proxy's Gemini-shaped endpoint (e.g. GapGPT's, if they expose one —
  **verify the exact path/auth against the provider's docs**; this follows
  Google's own documented request shape, but a proxy may differ). Not
  applicable to the local path, which always talks to `LLM_URL`.

```json
{ "instruction": "02_radiology_report_assist_stt", "stt_mode": "multimodal",
  "llm_model": "qwen3-omni-30b" }
```

`GET /instructions/{id}` reports `"supports_multimodal_llm": true/false` so a
client knows whether to offer this mode at all (only instructions with an
`"llm_audio"` step support it; `01_Casebook` doesn't).

### Hybrid mode: STT transcript(s) *and* audio, both given to the LLM

`"stt_mode": "hybrid"` combines the two pipelines above: one or more STT
slots run (same as `"separate"` — at least one must be configured), **and**
the audio-capable LLM is given the audio directly (same as `"multimodal"`) —
but this time, whichever `transcript_N` the STT slot(s) produced is folded
into the LLM's prompt as labeled reference material, explicitly framed as
"may contain errors, cross-check against what you hear, don't treat as
ground truth." This is for helping the LLM's own listening/judgment with a
second opinion, not for replacing it — the LLM's own transcription is still
what gets returned; the STT transcript(s) are advisory input, not the
answer.

```json
{ "instruction": "02_radiology_report_assist_stt", "stt_mode": "hybrid",
  "llm_model": "qwen3-omni-30b",
  "stt_slots": [{"model": "whisper"}] }
```

Same requirements as `"multimodal"` (audio-capable `llm_model`, format
restrictions) *plus* the same STT-slot requirements as `"separate"` (at
least one configured slot, `MAX_STT_SLOTS` cap, local-STT reachability).
`01_Casebook` and other `stt_mode`-unaware instructions are unaffected —
this only applies to instructions with both `"stt_slot"` and `"llm_audio"`
steps tagged for it (see below).

Adding either mode to another instruction: give it a `"use": "llm_audio"`
step tagged `"run_when_stt_mode": ["multimodal", "hybrid"]`, tag its
`"stt_slot"` steps `"run_when_stt_mode": ["separate", "hybrid"]`, and its
purely-text `"llm"` reconcile step `"run_when_stt_mode": ["separate"]`
(steps without this key always run, so instructions that don't use
`stt_mode` at all are unaffected).

**Confirmed working:** [GapGPT](https://gapgpt.app) — an Iran-based, Rial-payable
gateway that proxies GPT/Claude/Gemini through the exact OpenAI request shape
(same `openai` Python SDK, `client.chat.completions.create(model=..., messages=...)`).
Use `llm_base_url = "https://api.gapgpt.app/v1"` (or `https://api.gapapi.com/v1`
for their external-CDN route) with a GapGPT API key, and any model name they
proxy — e.g. `"llm_model": "openai:gpt-4o"`, `"openai:gemini-2.5-pro"`. No code
changes needed; this is just `OPENAI_BASE_URL`/`llm_base_url` pointed elsewhere.

This "openai:" path is confirmed for plain text chat. GapGPT's catalog also
lists Gemini "live"/"native-audio-dialog" models (their `GET /v1/models`
shows `"supported_endpoint_types": ["gemini", "openai"]` for those) — these
are the ones relevant to BuAli's `gemini:` multimodal path above, but the
exact base URL/auth GapGPT expects for their Gemini-shaped endpoint (as
opposed to the OpenAI-shaped one used for everything else) **hasn't been
verified against their docs** — confirm before relying on it.

### API keys — not required to be preset, and independent per role

External calls need a key, but it **does not have to live on the server**.
Pass `stt_api_key`/`stt_base_url` and/or `llm_api_key`/`llm_base_url` in
`POST /session` (session defaults) and/or `POST /run` (per-call overrides) —
that's the normal way to supply them. They're kept **separate** on purpose:
STT's external call and the LLM's external call can be entirely different
providers or accounts (e.g. a different cloud STT service for STT, OpenAI for
the LLM) — one is not required just because the other is set.

`OPENAI_API_KEY`/`OPENAI_BASE_URL` in `.env` are only fallback defaults (e.g.
for a shared admin key), used independently for whichever side is missing its
own. If neither an explicit key nor that fallback is present when an external
call actually runs, it fails with a clear error at that point — `POST
/session` itself never requires a key, since starting a session doesn't need
one yet.

None of the `*_api_key` fields are ever echoed back by `GET /status`.
