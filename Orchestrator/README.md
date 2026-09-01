# Orchestrator

The HTTP front door in front of the assistant's **pipelines**. A pipeline is
a plain Python module — no JSON instruction files, no generic step-loop
reading them. Each one picks its own model(s) internally; that choice is
never exposed to a caller of this API. STT and Core_LLM stay reachable on
their own ports for direct testing, same as always — this service just
coordinates them for a given pipeline.

BuAli and Casebook, which used to live here as JSON instructions, have been
extracted into their own standalone projects (`Spin_BuAli`, `Spin_CaseBook`).
This engine is for other, future use cases.

## Run
```bash
pip install -r requirements.txt
./run.sh          # Linux;  run.bat on Windows
```
Serves on `0.0.0.0:9000` (docs at `/docs`). Set `STT_URL` / `LLM_URL` in `.env`
if the module servers aren't at the defaults (`:8000` / `:8001`).

## API
| Method & path | Purpose |
|---|---|
| `GET /health` | orchestrator + STT + LLM reachability |
| `GET /pipelines` | which pipelines exist, and what each does — no model details |
| `POST /pipelines/{id}/run` | advance one pipeline's conversation by a turn |

`POST /pipelines/{id}/run` takes `{"history": [...], "text": "..."}` and
returns whatever that pipeline defines as its result — see the pipeline's own
module for the exact shape (`pipelines/greeting.py`'s is documented below).
`history` is the conversation so far, in the exact shape the previous call
returned it in (`[]` to start a new one); `text` is the newest message from
the caller — omit it on the very first call to just get the pipeline's
opening reply. **No field here selects a model.** That's deliberate: which
model backs a pipeline is decided in that pipeline's own code
(`pipelines/<name>.py`), not by whoever is calling this API.

## Adding a pipeline

Write `pipelines/<name>.py` exposing:
- `ID`, `NAME`, `DESCRIPTION` — what `GET /pipelines` reports.
- `run(messages: list[dict]) -> dict` — `messages` is the conversation so
  far (`{"role": "user"|"assistant", "content": str}` dicts); return
  whatever JSON-serializable result this pipeline produces. Raise on
  failure — the API layer turns that into a `502`.

Then list its module in `pipelines/__init__.py`'s `_MODULES`. That's the only
other file that needs to change; `main.py` and `module_clients.py` are
generic across every pipeline.

`module_clients.py` has the thin STT/Core_LLM HTTP wrappers a pipeline needs
(`stt_transcribe`, `chat`, `llm_api_chat`, ...) plus the `"openai:<model>"`
local-vs-cloud convention (`is_api_model`) for a pipeline that wants to pick
a cloud model for itself. `OPENAI_API_KEY`/`OPENAI_BASE_URL` in `.env` are
only a fallback default for that case — nothing here is caller-supplied.

## `greeting` — the example pipeline

Greets the patient and asks how it can help. It will only engage with two
kinds of requests: a question about the Spin platform itself, or a medical
concern. Anything else is politely refused. For a medical concern, it asks
follow-up questions (symptoms, onset, duration, severity, ...) until it has
enough to describe the situation, then returns a structured summary instead
of another question.

Each call returns:
```json
{
  "reply": "<text to show the patient>",
  "status": "in_progress" | "refused" | "complete",
  "patient_situation": null | {
    "chief_complaint": "...", "symptoms": ["..."], "onset": "...",
    "duration": "...", "severity": "...", "associated_symptoms": ["..."],
    "relevant_history": "...", "notes": "..."
  }
}
```
`patient_situation` is only ever non-null once `status` is `"complete"`.

```bash
curl -X POST http://localhost:9000/pipelines/greeting/run \
  -H "Content-Type: application/json" -d '{"history": []}'
# -> {"reply": "سلام! ...", "status": "in_progress", "patient_situation": null}

curl -X POST http://localhost:9000/pipelines/greeting/run \
  -H "Content-Type: application/json" \
  -d '{"history": [], "text": "I have had a headache since this morning"}'
```
Send the `reply` back as an `"assistant"` turn appended to `history` on the
next call, to keep the conversation going.
