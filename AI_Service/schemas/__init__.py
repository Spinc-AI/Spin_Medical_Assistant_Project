"""Pydantic models for AI_Service.

Split three ways: `state` (what the interview knows), `routing` (what the
router decides), `response` (what the HTTP API takes and returns).
"""
from .response import (
    URGENCY_ORDER,
    InterviewRequest,
    InterviewResponse,
    Message,
    Urgency,
)
from .routing import RouterInput, RouterOutput, RoutingDecision
from .state import DEFAULT_DOMAIN, DOMAIN_SLOTS, ConversationState, slots_for

__all__ = [
    "ConversationState", "DOMAIN_SLOTS", "DEFAULT_DOMAIN", "slots_for",
    "RouterInput", "RouterOutput", "RoutingDecision",
    "InterviewRequest", "InterviewResponse", "Message", "Urgency", "URGENCY_ORDER",
]
