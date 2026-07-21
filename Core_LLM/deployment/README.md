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

Set `response_format` to `{"type":"json_object"}` to force valid-JSON output.

## Example
```bash
curl -X POST http://193.93.169.134:8001/chat -H "Content-Type: application/json"   -d '{"messages":[{"role":"user","content":"hello"}]}'
```
