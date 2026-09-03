"""Checks run on the raw user input BEFORE it reaches the model.

Two kinds of rule live here, and the difference matters:

  1. STRUCTURAL rules (empty input, oversized input). These are settled --
     they protect the service, not the patient, and need no clinical sign-off.

  2. CONTENT rules. The three below are PLACEHOLDERS, marked as such, and are
     not a clinical safety policy. Writing one is not an engineering decision:
     what an interview assistant must refuse, redirect, or escalate is for
     whoever owns clinical safety on this project to define, and it needs to
     be reviewed against real Persian phrasing rather than invented here.
     What this module provides is the HOOK (`register_rule`) and the shape a
     real rule set would plug into.

CLINICAL SIGN-OFF REQUIRED before this file's content rules are treated as
anything more than a smoke test.
"""
import re
from typing import Callable

from pydantic import BaseModel

import config


class PrecheckResult(BaseModel):
    """Example (rejection):
        {"ok": false, "code": "empty_input",
         "reason": "پیام خالی است.", "replacement_question": null}
    """

    ok: bool
    code: str | None = None
    reason: str | None = None
    replacement_question: str | None = None


PASS = PrecheckResult(ok=True)

# --- Content rules: PLACEHOLDERS, pending clinical review -------------------
# (pattern, code, message shown to the user). Each is a clearly-labelled
# stand-in chosen to be uncontroversial in its direction -- "send this person
# to emergency services" and "don't answer a prescription question in an
# intake interview" -- not to be complete.
PLACEHOLDER_CONTENT_RULES: tuple[tuple[str, str, str], ...] = (
    # PLACEHOLDER: self-harm. Direction is not in doubt; the wording, the
    # trigger list and the escalation path all need clinical ownership.
    (r"خودکشی|خودم را بکشم|به زندگی ?ام پایان|suicide|kill myself",
     "crisis_content",
     "این گفتگو برای گرفتن شرح‌حال چشم‌پزشکی است. اگر در خطر فوری هستید، همین حالا "
     "با اورژانس ۱۱۵ تماس بگیرید یا به نزدیک‌ترین مرکز درمانی مراجعه کنید."),
    # PLACEHOLDER: prescription/dosage requests. An intake interview does not
    # prescribe; where the line sits (is "چند قطره بریزم؟" a dosage question?)
    # is a clinical call.
    (r"چه دارویی (بخورم|مصرف کنم)|دوز (دارو|مصرف)|نسخه بنویس|prescribe|what dose",
     "medication_request",
     "در این مرحله فقط شرح‌حال شما ثبت می‌شود و دارو تجویز نمی‌شود. "
     "لطفاً درباره‌ی علائم خودتان توضیح دهید."),
    # PLACEHOLDER: prompt-injection style attempts to rewrite the assistant's
    # instructions. Structural in spirit, listed here because the trigger set
    # is as open-ended as the clinical ones.
    (r"ignore (all )?(previous|above) instructions|دستورات قبلی را نادیده",
     "instruction_override",
     "لطفاً فقط علائم و شرح‌حال خودتان را بنویسید."),
)

# Extra rules registered at runtime, e.g. by a future clinical-policy module.
# Each takes the raw message and returns a PrecheckResult (ok=True to pass).
_EXTRA_RULES: list[Callable[[str], PrecheckResult]] = []


def register_rule(rule: Callable[[str], PrecheckResult]) -> None:
    """Add a content rule. This is the intended extension point -- a real
    clinical rule set should arrive through here rather than by editing the
    placeholder table above."""
    _EXTRA_RULES.append(rule)


def clear_rules() -> None:
    """Drop runtime-registered rules (tests use this to stay isolated)."""
    _EXTRA_RULES.clear()


def precheck(user_message: str, max_chars: int | None = None) -> PrecheckResult:
    """Structural checks, then placeholder content rules, then registered ones.

    Example:
        precheck("   ") -> PrecheckResult(ok=False, code="empty_input", ...)
        precheck("چشمم قرمز شده") -> PrecheckResult(ok=True)
    """
    max_chars = max_chars if max_chars is not None else config.MAX_INPUT_CHARS

    if user_message is None or not str(user_message).strip():
        return PrecheckResult(ok=False, code="empty_input",
                              reason="پیام خالی است؛ لطفاً علائم خود را بنویسید.")
    if len(user_message) > max_chars:
        return PrecheckResult(
            ok=False, code="input_too_long",
            reason=f"پیام بیش از حد طولانی است ({len(user_message)} نویسه، "
                   f"حداکثر {max_chars}). لطفاً کوتاه‌تر بنویسید.")

    text = user_message.casefold()
    for pattern, code, message in PLACEHOLDER_CONTENT_RULES:
        if re.search(pattern, text):
            return PrecheckResult(ok=False, code=code, reason=message,
                                  replacement_question=message)

    for rule in _EXTRA_RULES:
        result = rule(user_message)
        if not result.ok:
            return result

    return PASS
