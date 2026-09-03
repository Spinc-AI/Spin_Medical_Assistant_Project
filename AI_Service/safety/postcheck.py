"""Checks run on the parsed response AFTER output_validator says it's well-formed.

The one job here: don't take the model's word for how urgent something is.
A model that has just been told a patient lost their vision overnight and
answers `"urgency": "routine"` is wrong in a way that matters, and the fix is
a short explicit table -- not a bigger prompt and not a second model.

The table is deliberately small, readable, and one-directional: it can only
raise urgency, never lower it. A model that says `urgent` keeps `urgent` even
if no rule fires, because a false alarm costs a wasted clinic visit while a
missed one costs sight.

CLINICAL SIGN-OFF REQUIRED on the rules themselves. The two below follow
directly from the task spec's own example (sudden vision loss => urgent);
they are a starting point, not a triage protocol.
"""
from typing import Any

from pydantic import BaseModel, Field

from schemas.response import URGENCY_ORDER, InterviewResponse, Urgency
from schemas.state import ConversationState


class EscalationRule(BaseModel):
    """One state-pattern -> minimum-urgency rule.

    `when` maps a slot name to the values that trigger it; ALL entries must
    match for the rule to fire. `min_urgency` is a floor, never a ceiling.

    Example:
        {"domain": "eye", "when": {"vision_change": ["sudden_loss"]},
         "min_urgency": "urgent", "why": "sudden vision loss"}
    """

    domain: str
    when: dict[str, list[Any]]
    min_urgency: Urgency
    why: str
    version: str = "v1"


ESCALATION_RULES: list[EscalationRule] = [
    EscalationRule(
        domain="eye",
        when={"vision_change": ["sudden_loss"]},
        min_urgency=Urgency.URGENT,
        why="از دست رفتن ناگهانی بینایی",
    ),
    EscalationRule(
        domain="eye",
        when={"vision_change": ["decreased"], "onset": ["sudden"]},
        min_urgency=Urgency.URGENT,
        why="کاهش دید با شروع ناگهانی",
    ),
]


class PostcheckResult(BaseModel):
    """The response as it should be returned, plus what was changed and why."""

    response: InterviewResponse
    escalated: bool = False
    notes: list[str] = Field(default_factory=list)


def _rank(urgency: Urgency | str) -> int:
    value = Urgency(urgency) if not isinstance(urgency, Urgency) else urgency
    return URGENCY_ORDER.index(value)


def _matches(rule: EscalationRule, state: ConversationState) -> bool:
    if rule.domain != state.domain:
        return False
    return all(state.slots.get(slot) in values for slot, values in rule.when.items())


def required_urgency(state: ConversationState,
                     rules: list[EscalationRule] | None = None) -> tuple[Urgency, list[str]]:
    """The highest urgency the collected slots demand, and the reasons why.

    Example:
        required_urgency(ConversationState(domain="eye",
                                           slots={"vision_change": "sudden_loss"}))
        -> (Urgency.URGENT, ["از دست رفتن ناگهانی بینایی"])
    """
    floor, reasons = Urgency.ROUTINE, []
    for rule in (rules if rules is not None else ESCALATION_RULES):
        if _matches(rule, state) and _rank(rule.min_urgency) >= _rank(floor):
            if _rank(rule.min_urgency) > _rank(floor):
                floor, reasons = rule.min_urgency, []
            reasons.append(rule.why)
    return floor, reasons


def postcheck(response: InterviewResponse, state: ConversationState,
              rules: list[EscalationRule] | None = None) -> PostcheckResult:
    """Raise `response.urgency` to whatever the state demands, and say so.

    The discrepancy is recorded in `notes` (and returned to the caller in
    InterviewResponse.notes) rather than silently corrected: a model that
    regularly under-calls urgency is a prompt problem someone needs to see.
    """
    floor, reasons = required_urgency(state, rules)
    notes: list[str] = []
    escalated = False

    if _rank(floor) > _rank(response.urgency):
        notes.append(
            f"postcheck escalated urgency '{response.urgency.value}' -> "
            f"'{floor.value}' ({'; '.join(reasons)}) -- the model under-called it"
        )
        response = response.model_copy(update={"urgency": floor})
        escalated = True

    return PostcheckResult(response=response, escalated=escalated, notes=notes)
