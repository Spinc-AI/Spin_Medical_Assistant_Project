"""The public request/response pair for POST /interview, plus the chat Message
shape shared by prompt_loader and the LLM clients.
"""
from enum import Enum

from pydantic import BaseModel, Field

from .state import ConversationState


class Message(BaseModel):
    """One turn in the standard OpenAI chat format -- the same shape Core_LLM's
    POST /chat takes, so a list of these goes over the wire unchanged."""

    role: str  # "system" | "user" | "assistant"
    content: str


class Urgency(str, Enum):
    """How soon this needs a clinician.

    Only the two levels the task spec actually names. Intermediate tiers (a
    "within 24h" band, say) are a clinical-triage decision and must not be
    invented here -- whoever owns clinical safety for this project signs off
    before this enum grows.
    """

    ROUTINE = "routine"
    URGENT = "urgent"


# Least to most severe. postcheck.py escalates along this order and never
# de-escalates, so the ordering is load-bearing -- keep it sorted.
URGENCY_ORDER: list[Urgency] = [Urgency.ROUTINE, Urgency.URGENT]


class InterviewRequest(BaseModel):
    """One user turn.

    Example:
        {"session_id": "3f9c...", "user_message": "از دیروز چشم راستم قرمز است"}

    State handling: pass `session_id` and let AI_Service hold the state
    (the default -- see main.py), or pass `conversation_state` explicitly and
    keep it yourself. If both are given, the explicit state wins.

    The three `*_version`/`model_key` overrides exist for the evaluation
    harness, which has to force a specific model/prompt combination rather
    than let the router choose. Normal callers leave them unset.
    """

    user_message: str
    session_id: str | None = None
    conversation_state: ConversationState | None = None
    model_key: str | None = None
    prompt_version: str | None = None
    policy_version: str | None = None
    temperature: float = 0.3


class InterviewResponse(BaseModel):
    """The next interview question, plus everything needed to continue.

    Example:
        {"domain": "eye", "model": "gemma-4-12b",
         "question": "آیا همراه قرمزی، کاهش بینایی هم دارید؟",
         "urgency": "routine", "session_id": "3f9c...", "complete": false,
         "conversation_state": {"domain": "eye", "slots": {...}, "turn_count": 1},
         "policy_version": "v1", "prompt_version": "v1", "safety_version": "v1",
         "latency": 1.94, "notes": []}

    The first four fields are the task spec's contract; the rest are what a
    stateful multi-turn API needs on top of it. `notes` carries anything safety
    changed or flagged (e.g. an urgency the model under-called and postcheck
    escalated) -- empty on a clean turn.
    """

    domain: str
    model: str
    question: str
    urgency: Urgency = Urgency.ROUTINE
    session_id: str | None = None
    conversation_state: ConversationState = Field(default_factory=ConversationState)
    complete: bool = False
    policy_version: str | None = None
    prompt_version: str | None = None
    safety_version: str | None = None
    latency: float | None = None
    notes: list[str] = Field(default_factory=list)
