"""HTTP layer for Core_LLM — a thin FastAPI wrapper around llm_client.chat().

The orchestrator (and other modules) call this service over HTTP instead of
importing Core_LLM directly. Internally it just forwards to the same
`llm_client` seam that chat.py uses, which in turn talks to the Ollama
OpenAI-compatible API.

Run:
    python main.py            # or: uvicorn main:app --host 0.0.0.0 --port 8001
Interactive docs at http://<host>:8001/docs
"""
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import config
import llm_client
import multimodal
from schemas import ChatRequest, ChatResponse, HealthResponse

app = FastAPI(title="Core LLM Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health():
    """Liveness check, and which model this service is configured to talk to."""
    return HealthResponse(status="ok", model=config.LLM_MODEL)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send chat messages (OpenAI format) and get the assistant's full reply."""
    messages = [m.model_dump() for m in req.messages]
    try:
        reply = await run_in_threadpool(
            llm_client.chat,
            messages,
            model=req.model,
            temperature=req.temperature,
            response_format=req.response_format,
        )
    except Exception as exc:  # backend (Ollama) unreachable or errored
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}")
    return ChatResponse(model=req.model or config.LLM_MODEL, reply=reply)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Same as /chat, but streams the reply as plain-text chunks as they arrive."""
    messages = [m.model_dump() for m in req.messages]
    try:
        # Build the stream off the event loop; the generator does blocking I/O.
        stream = await run_in_threadpool(
            llm_client.chat,
            messages,
            model=req.model,
            temperature=req.temperature,
            stream=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}")
    return StreamingResponse(stream, media_type="text/plain")


@app.post("/unload")
async def unload(model: str | None = None):
    """Unload a model from Ollama memory (frees GPU/CPU). Defaults to the configured model."""
    m = model or config.LLM_MODEL
    try:
        await run_in_threadpool(llm_client.unload, m)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}")
    return {"status": "unloaded", "model": m}


@app.post("/chat_audio")
async def chat_audio(
    file: UploadFile = File(...),
    system_prompt: str = Form(...),
    text: str | None = Form(default=None),
):
    """Local multimodal chat: give the audio-capable model (Gemma 4 E4B) the
    audio directly, no STT step. Separate from /chat -- this goes through
    `transformers` directly, not Ollama, since Ollama doesn't support audio
    input yet. There's currently only one such model, so no `model` param."""
    audio_bytes = await file.read()
    audio_format = (file.filename or "").rsplit(".", 1)[-1].lower() or "wav"
    try:
        reply = await run_in_threadpool(
            multimodal.chat_audio, audio_bytes, audio_format, system_prompt, text
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Multimodal LLM error: {exc}")
    return {"model": config.MULTIMODAL_MODEL_ID, "reply": reply}


@app.post("/chat_audio/unload")
async def chat_audio_unload():
    """Unload the local multimodal model, freeing its VRAM."""
    await run_in_threadpool(multimodal.unload)
    return {"status": "unloaded", "model": config.MULTIMODAL_MODEL_ID}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT)
