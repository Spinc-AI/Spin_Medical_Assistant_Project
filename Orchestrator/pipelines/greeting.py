"""Greeting & triage — a simple pipeline, meant as a template for the next one.

Greets the patient, refuses anything that isn't about the Spin platform or a
medical concern, and — once it understands the concern well enough — hands
back a structured summary instead of more conversation.

A pipeline module needs exactly this shape: ID, NAME, DESCRIPTION, and a
run() function. The registry in pipelines/__init__.py is the only other
place that needs to know this pipeline exists.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ValidationError

import module_clients

ID = "greeting"
NAME = "Greeting & Triage"
DESCRIPTION = (
    "Greets the patient and asks how it can help. If the concern is medical, "
    "it asks follow-up questions until it can summarize the situation as "
    "structured JSON. Refuses anything unrelated to the Spin platform or to "
    "a medical concern."
)

# This pipeline's own choice of model — orchestrator API callers never see or
# pick this; only the pipeline's own code does.
LLM_MODEL = "aya-expanse-8b"

GREETING_REPLY = "سلام! من دستیار پزشکی اسپین هستم. چطور می‌تونم کمکتون کنم؟"

SYSTEM_PROMPT = """You are the front-desk greeting agent for Spin, a Persian-language medical decision-support platform. You are mid-conversation with a patient; your job this turn is to move that conversation forward correctly.

You may ONLY engage with two kinds of requests:
  (a) a question about the Spin platform itself — what it is, what it can do;
  (b) a medical or health concern the patient wants help with — symptoms, an illness, how they're feeling.
Anything else (small talk, unrelated topics, requests for advice or information outside medicine and the platform) must be politely refused — do not answer the off-topic part even partially, just explain you can only help with medical concerns or questions about Spin, and invite them to ask about one of those instead.

When the patient describes a medical concern, ask focused follow-up questions — one or two at a time, conversationally, not a clinical checklist — to understand: the chief complaint, the main symptoms, when it started (onset), how long it has lasted (duration), how severe it is, and anything else relevant (associated symptoms, relevant history). Use your judgment about when you have enough to describe the situation clearly; you do not need every field filled before finishing.

You must always reply with exactly one JSON object and nothing else, matching this shape:
{
  "reply": "<what to say to the patient, in the language they are using>",
  "status": "in_progress" | "refused" | "complete",
  "patient_situation": null | {
    "chief_complaint": string or null,
    "symptoms": [string, ...],
    "onset": string or null,
    "duration": string or null,
    "severity": string or null,
    "associated_symptoms": [string, ...],
    "relevant_history": string or null,
    "notes": string or null
  }
}

Rules:
- status = "refused": you declined an off-topic request this turn. "reply" explains, briefly and politely, what you can actually help with. patient_situation is null.
- status = "in_progress": you are still greeting the patient, or still gathering information about a medical concern. patient_situation is null.
- status = "complete": ONLY once you have gathered enough to describe the patient's situation. "reply" briefly acknowledges you understand, and patient_situation is filled in — leave an individual field null/empty only if the patient genuinely never mentioned it.
- Never invent a symptom, a date, or any fact the patient did not actually say.
- Reply in whatever language the patient's messages are in."""


class PatientSituation(BaseModel):
    chief_complaint: Optional[str] = None
    symptoms: list[str] = []
    onset: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    associated_symptoms: list[str] = []
    relevant_history: Optional[str] = None
    notes: Optional[str] = None


class GreetingReply(BaseModel):
    reply: str
    status: str  # "in_progress" | "refused" | "complete"
    patient_situation: Optional[PatientSituation] = None


def run(messages: list[dict]) -> dict:
    """`messages` is the conversation so far — role/content dicts, ending with
    the patient's latest turn. Empty means "start a new conversation": that's
    answered directly, with no LLM call needed."""
    if not messages:
        return {"reply": GREETING_REPLY, "status": "in_progress", "patient_situation": None}

    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    try:
        raw_reply = module_clients.chat(llm_messages, model=LLM_MODEL,
                                        response_format={"type": "json_object"})
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}")

    try:
        parsed = module_clients.extract_json(raw_reply)
        result = GreetingReply.model_validate(parsed)
    except (ValueError, ValidationError) as exc:
        raise RuntimeError(f"LLM did not return a valid reply:\n{raw_reply}\n({exc})")

    return result.model_dump()
