"""Spin Orchestrator — the HTTP API in front of the assistant's pipelines.

A pipeline is a plain Python module (pipelines/<name>.py), not a JSON
instruction — it picks its own model(s) internally, so nothing about model
choice is ever exposed through this API. STT and Core_LLM stay reachable on
their own ports for direct testing; this service is only the coordinated
front door callers actually talk to.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

import config
import module_clients
import pipelines
from schemas import PipelineInfo, PipelineRunRequest

app = FastAPI(title="Spin Orchestrator")


@app.get("/health")
def health():
    return {"orchestrator": "ok", "stt": module_clients.stt_health(), "llm": module_clients.llm_health()}


@app.get("/pipelines", response_model=list[PipelineInfo])
def list_pipelines():
    """Which pipelines exist and what they do — no model details, that's internal."""
    return pipelines.list_pipelines()


@app.post("/pipelines/{pipeline_id}/run")
def run_pipeline(pipeline_id: str, request: PipelineRunRequest) -> dict:
    """Advance one pipeline's conversation by a turn.

    Send back the previous response's own `history`-shaped turns as `history`
    (empty to start a new conversation) plus the patient's newest `text`
    (omit it on the very first call to just receive the opening message).
    """
    pipeline = pipelines.PIPELINES.get(pipeline_id)
    if pipeline is None:
        raise HTTPException(404, f"unknown pipeline '{pipeline_id}'")

    history = [turn.model_dump() for turn in request.history]
    if request.text:
        history.append({"role": "user", "content": request.text})

    try:
        return pipeline.run(history)
    except Exception as exc:
        raise HTTPException(502, str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT)
