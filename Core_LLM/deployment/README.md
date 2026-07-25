# Core_LLM Service

Wraps the local LLM (**Aya Expanse 8B** via Ollama, OpenAI-compatible) behind a
small HTTP API. `llm_client.chat()` is the seam; `main.py` exposes it over HTTP.

## Run
```bash
pip install -r requirements.txt
./run.sh          # Linux;  run.bat on Windows
```
Serves on `0.0.0.0:8001` (docs at `/docs`). Set the Ollama address + model in `.env`.

## API
| Method & path | Purpose |
|---|---|
| `GET /` | health + the configured model |
| `POST /chat` | body `{messages, model?, temperature?, response_format?}` -> `{model, reply}` |
| `POST /chat/stream` | same input, streams the reply as plain text |
| `POST /unload?model=` | unload the model from Ollama memory |
| `GET /chat_audio/models` | list local audio-capable model keys + which is loaded |
| `POST /chat_audio` | multipart: `file` (audio) + `system_prompt` + `text?` + `model?` -> `{model, reply}` |
| `POST /chat_audio/unload` | unload the currently-loaded local multimodal model, freeing its VRAM |

Set `response_format` to `{"type":"json_object"}` to force valid-JSON output.

## Example
```bash
curl -X POST http://193.93.169.134:8001/chat -H "Content-Type: application/json"   -d '{"messages":[{"role":"user","content":"hello"}]}'
```

## Local multimodal (audio) models

`/chat_audio` is a **separate path from Ollama** — it loads one of these
directly via `transformers`, since Ollama doesn't support audio input yet.
Only one is held in memory at a time; picking a different `model` unloads
the current one first (same swap pattern as the STT module).

| `model` key | Model | Notes |
|---|---|---|
| `gemma-4-e4b` (default) | `google/gemma-4-E4B-it` | Lighter, faster |
| `qwen3-omni-30b` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | **Best tested option for Persian** — confirmed via [PARSA-Bench](https://arxiv.org/html/2603.14456), a Persian audio-LM benchmark (0.358 WER vs. 6-9 for Gemma-3n-class models). Bigger — needs real VRAM headroom (MoE, ~3B active params, but full weights are much larger). |

Lazy-loaded on first request for a given model (slow the first time, fast
after). Needs a real GPU with VRAM headroom; install the CUDA build of
`torch` (see [pytorch.org](https://pytorch.org) for the right command),
otherwise it falls back to CPU.

```bash
curl http://localhost:8001/chat_audio/models

curl -X POST http://localhost:8001/chat_audio \
  -F "file=@report.wav" \
  -F "system_prompt=Transcribe this radiology report as JSON." \
  -F "text=Optional extra instructions" \
  -F "model=qwen3-omni-30b"
```
