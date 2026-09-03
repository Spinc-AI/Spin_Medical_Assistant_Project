"""Request/response shapes for the orchestrator's HTTP API.

None of these carry a model field — which model backs a pipeline is chosen
by that pipeline's own code (see pipelines/), never by the caller.
"""
from typing import Literal, Optional

from pydantic import BaseModel


class ChatTurn(BaseModel):
    """One turn of a conversation, as the caller sends and receives it."""
    role: Literal["user", "assistant"]
    content: str


class PipelineInfo(BaseModel):
    """What GET /pipelines reports about one available pipeline — no model details."""
    id: str
    name: str
    description: str


class PipelineRunRequest(BaseModel):
    """Body of POST /pipelines/{id}/run.

    `history` is the conversation so far, in the shape returned by the
    previous call (empty to start a new one). `text` is the patient's newest
    message — omit it on the very first call to just get the opening
    greeting. No model field exists here on purpose: which model backs a
    pipeline is not something a caller gets to pick.
    """
    history: list[ChatTurn] = []
    text: Optional[str] = None


PipelineStatus = Literal["in_progress", "refused", "complete"]
