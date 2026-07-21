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
| `POST /chat_audio` | multipart: `file` (audio) + `system_prompt` + `text?` -> `{model, reply}` |
| `POST /chat_audio/unload` | unload the local multimodal model, freeing its VRAM |

Set `response_format` to `{"type":"json_object"}` to force valid-JSON output.

## Example
```bash
curl -X POST http://193.93.169.134:8001/chat -H "Content-Type: application/json"   -d '{"messages":[{"role":"user","content":"hello"}]}'
```

## Local multimodal (audio) model

`/chat_audio` is a **separate path from Ollama** — it loads `google/gemma-4-E4B-it`
directly via `transformers` (`AutoModelForMultimodalLM`), since Ollama doesn't
support audio input yet. Lazy-loaded on first request (slow the first time,
fast after — stays in memory until `/chat_audio/unload`). Needs a real GPU
with VRAM headroom; install the CUDA build of `torch` (see
[pytorch.org](https://pytorch.org) for the right command), otherwise it falls
back to CPU. There's only one multimodal model right now, so no `model` field.

```bash
curl -X POST http://localhost:8001/chat_audio \
  -F "file=@report.wav" \
  -F "system_prompt=Transcribe this radiology report as JSON." \
  -F "text=Optional extra instructions"
```
