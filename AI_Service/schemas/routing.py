"""What the Router is asked, and what it answers."""
from pydantic import BaseModel, Field

from .state import ConversationState


class RouterInput(BaseModel):
    """Example:
        {"user_message": "چشمم از دیروز قرمز شده",
         "conversation_state": {"domain": null, "slots": {}, "turn_count": 0}}
    """

    user_message: str
    conversation_state: ConversationState = Field(default_factory=ConversationState)


class RoutingDecision(BaseModel):
    """Which domain this conversation is in and which model should answer it.

    `policy_version` is stamped on every decision so a line in the logs or a
    row in an evaluation result can be traced back to the policy that produced
    it -- without it, comparing two evaluation runs tells you nothing about why
    they differ.

    Example:
        {"domain": "eye", "model_key": "gemma-4-12b", "policy_version": "v1",
         "reason": "keyword match: چشم"}
    """

    domain: str
    model_key: str
    policy_version: str
    reason: str | None = None


# The task spec calls this RouterOutput; the roadmap calls it RoutingDecision.
# Same model, both names exported so neither document reads as wrong.
RouterOutput = RoutingDecision
