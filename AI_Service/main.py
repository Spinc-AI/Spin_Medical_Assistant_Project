"""HTTP layer for AI_Service — the interview/routing service in front of Core_LLM.

Style mirrors Core_LLM/deployment/main.py and Orchestrator/orchestrator.py: a
thin FastAPI wrapper, CORS from config, a health check, and one real endpoint.

WHERE CONVERSATION STATE LIVES
------------------------------
Server-side, in memory, keyed by `session_id`. The caller sends a message plus
the session id it got back last time; AI_Service remembers the rest.

The trade-off, stated plainly because it will matter later:

  + The client stays trivial -- a Backend integrating this needs to keep one
    string, not mirror a schema it doesn't own.
  + State stays consistent even if slot extraction changes shape.
  - It does not survive a restart, and it does not work behind more than one
    replica without sticky sessions. Both are real limits.

Chosen anyway because no persistence layer exists anywhere in this repo yet
(Orchestrator's sessions are in-memory too), and inventing one here would be a
larger decision than this module should make on its own. The escape hatch is
already in the API: pass `conversation_state` explicitly in the request and
AI_Service uses it instead of its own copy, so a caller that wants to own the
state -- or a deployment that grows a database -- needs no change here.

Run:
    python main.py            # or: uvicorn main:app --host 0.0.0.0 --port 9100
Interactive docs at http://<host>:9100/docs
"""
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from llm.core_llm_client import CoreLLMClient
from orchestrator import interview
from prompts.prompt_loader import UnknownPromptVersion, available_versions
from router.policy import UnknownPolicyVersion, routable_domains
from schemas.response import InterviewRequest, InterviewResponse, Message
from schemas.state import ConversationState

app = FastAPI(title="AI Service — Interview Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIENT = CoreLLMClient()


@dataclass
class Session:
    """One conversation: its state plus the transcript sent as prompt history."""

    state: ConversationState = field(default_factory=ConversationState)
    history: list[Message] = field(default_factory=list)


class SessionStore:
    """In-memory sessions, oldest evicted past config.MAX_SESSIONS.

    Bounded on purpose: an unbounded dict on a long-running server is a slow
    leak, and dropping the least-recently-used conversation is the least-bad
    failure -- that caller can start a new one.
    """

    def __init__(self, limit: int = config.MAX_SESSIONS):
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._limit = limit

    def get_or_create(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = Session()
            self._sessions[session_id] = session
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._limit:
            self._sessions.popitem(last=False)
        return session

    def drop(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._sessions)


SESSIONS = SessionStore()


class HealthResponse(BaseModel):
    status: str
    core_llm_url: str
    core_llm_reachable: bool
    active_sessions: int
    prompt_version: str
    policy_version: str
    safety_version: str
    router_enabled: bool


@app.get("/")
def root():
    """Liveness, and the versioned behaviour this instance is running."""
    return {
        "status": "ok",
        "service": "AI_Service",
        "prompt_versions_available": available_versions(),
        "domains": routable_domains(),
        "prompt_version": config.PROMPT_VERSION,
        "policy_version": config.ROUTING_POLICY_VERSION,
        "safety_version": config.SAFETY_VERSION,
        "router_enabled": config.ROUTER_ENABLED,
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Is AI_Service up, and can it actually reach Core_LLM?

    Mirrors Orchestrator's GET /health, which reports STT/LLM reachability --
    a service that is 'up' but can't reach its dependency is not usable, and
    saying so here saves a round of debugging at the caller.
    """
    reachable = await run_in_threadpool(CLIENT.health)
    return HealthResponse(
        status="ok" if reachable else "degraded",
        core_llm_url=config.CORE_LLM_URL,
        core_llm_reachable=reachable,
        active_sessions=len(SESSIONS),
        prompt_version=config.PROMPT_VERSION,
        policy_version=config.ROUTING_POLICY_VERSION,
        safety_version=config.SAFETY_VERSION,
        router_enabled=config.ROUTER_ENABLED,
    )


@app.post("/interview", response_model=InterviewResponse)
async def run_interview(req: InterviewRequest):
    """One interview turn.

    Send `user_message` with no `session_id` to start; the response carries a
    `session_id` to send back on every following turn. Pass
    `conversation_state` instead if you'd rather hold the state yourself --
    it takes precedence over the server's copy.

    `model_key` / `prompt_version` / `policy_version` override the router and
    the configured defaults. They exist for evaluation and debugging; normal
    callers omit them.
    """
    session_id = req.session_id or uuid.uuid4().hex
    session = SESSIONS.get_or_create(session_id)
    request = req.model_copy(update={"session_id": session_id})

    try:
        response, state = await run_in_threadpool(
            interview.run_turn, request, session.state, CLIENT, session.history)
    except UnknownPromptVersion as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except UnknownPolicyVersion as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 -- surface as 502, same as Core_LLM does
        raise HTTPException(status_code=502, detail=f"interview failed: {exc}")

    session.state = state
    session.history.append(Message(role="user", content=req.user_message))
    session.history.append(Message(role="assistant", content=response.question))
    return response


@app.delete("/interview/{session_id}")
def end_interview(session_id: str):
    """Forget a conversation.

    Worth having even with in-memory state: a finished interview holds symptom
    text, and CONTRIBUTING.md is explicit that patient data is not something
    this project keeps lying around.
    """
    return {"status": "deleted" if SESSIONS.drop(session_id) else "not_found",
            "session_id": session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT)
