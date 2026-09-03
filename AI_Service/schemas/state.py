"""Conversation state: which clinical domain we're in, and what we've learned.

Deliberately NOT a fixed set of eye fields. `slots` is an open dict keyed by
slot name, and `DOMAIN_SLOTS` below is the single place a new domain gets
declared -- adding `skin` or `chest_pain` means one entry there, and nothing
in interview.py, router.py, or the prompts has to change shape.
"""
from typing import Any

from pydantic import BaseModel, Field

# The slot catalogue: domain -> the questions an interview for that domain is
# trying to answer, in the order they're normally asked.
#
# `eye` is the only clinically-specified domain so far (it's the one worked
# example in the task spec). `general` is the fallback the router falls back
# to when nothing matches -- it has no slots, so an interview there stays a
# free-form triage conversation rather than pretending to fill a form.
#
# CLINICAL REVIEW NEEDED before adding domains: which slots matter for a
# domain is a clinical decision, not an engineering one.
DOMAIN_SLOTS: dict[str, tuple[str, ...]] = {
    "eye": ("onset", "redness", "vision_change"),
    "general": (),
}

DEFAULT_DOMAIN = "general"


def slots_for(domain: str | None) -> tuple[str, ...]:
    """The slot names for `domain`, or () for an unknown/None domain."""
    return DOMAIN_SLOTS.get(domain or "", ())


class ConversationState(BaseModel):
    """What the interview knows so far, carried across turns.

    Example (the eye domain, mid-interview):
        {"domain": "eye",
         "slots": {"onset": "۲ روز پیش", "redness": None, "vision_change": None},
         "turn_count": 1}

    A slot present with value `None` means "asked-about but not yet answered";
    a slot absent from the dict means "not part of this domain". `filled()`
    and `missing()` treat both the same way, so callers don't have to care.
    """

    domain: str | None = None
    slots: dict[str, Any] = Field(default_factory=dict)
    turn_count: int = 0

    def missing(self) -> list[str]:
        """Slots for this domain that still have no value, in catalogue order."""
        return [s for s in slots_for(self.domain) if self.slots.get(s) in (None, "")]

    def filled(self) -> dict[str, Any]:
        """Only the slots that actually have a value."""
        return {k: v for k, v in self.slots.items() if v not in (None, "")}

    def is_complete(self) -> bool:
        """True once every slot the domain declares has a value.

        A domain with no slots (`general`) is never 'complete' in this sense --
        there's no form to finish -- so it reports False and the caller decides
        when to stop.
        """
        return bool(slots_for(self.domain)) and not self.missing()
