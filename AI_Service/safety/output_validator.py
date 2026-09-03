"""Structural validation of the raw model output. No clinical judgement here.

This answers exactly one question: did the model return the JSON object we
asked for? Whether the *content* is clinically sensible is postcheck.py's
problem, and whether the input was acceptable is precheck.py's.

Returns a typed `ValidationResult` rather than raising, because "the model
produced junk" is an expected outcome that the caller has to handle and that
evaluation/metrics.py has to count -- not an exception to guess at.
"""
import json
import re
from typing import Any

from pydantic import BaseModel, Field

import config
from schemas.response import Urgency

# `model` is deliberately NOT required. The router already knows which model
# was called, so the model echoing its own name back carries no information --
# and rejecting an otherwise-perfect answer over a field we can fill in
# ourselves would be throwing away good turns.
REQUIRED_FIELDS = ("domain", "question", "urgency")

ALLOWED_URGENCY = {u.value for u in Urgency}

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ValidationResult(BaseModel):
    """Example (a good turn):
        {"valid": true, "repaired": false, "reason": null,
         "data": {"domain": "eye", "question": "...", "urgency": "routine"}}
    """

    valid: bool
    data: dict[str, Any] | None = None
    reason: str | None = None
    repaired: bool = Field(
        default=False,
        description="True when the JSON only parsed after stripping code fences "
                    "or surrounding prose. Counts as a format failure for "
                    "metrics, but is still usable by the service.",
    )


def _parse(raw: str) -> tuple[dict | None, bool, str | None]:
    """(parsed, repaired, error). Tries strict JSON first, then recovery."""
    try:
        return json.loads(raw), False, None
    except (json.JSONDecodeError, TypeError):
        pass

    # Recovery, in the same spirit as Orchestrator's _extract_json(): local
    # models have no enforced JSON mode, so a stray ```json fence or a
    # sentence of preamble is the common failure -- usable, but recorded as
    # `repaired` so evaluation can see how often it happens per prompt version.
    candidate = _FENCE.sub("", raw or "").strip()
    try:
        return json.loads(candidate), True, None
    except (json.JSONDecodeError, TypeError):
        pass

    match = _OBJECT.search(candidate)
    if match:
        try:
            return json.loads(match.group(0)), True, None
        except json.JSONDecodeError as exc:
            return None, True, f"not valid JSON even after extraction: {exc}"
    return None, False, "output is not JSON and contains no JSON object"


def validate_output(raw: str, allowed_domains: list[str] | None = None,
                    max_chars: int | None = None) -> ValidationResult:
    """Check one raw model reply against InterviewResponse's shape.

    Example:
        validate_output('{"domain":"eye","question":"...","urgency":"routine"}')
        -> ValidationResult(valid=True, data={...})

    `allowed_domains` defaults to the router's currently-routable domains --
    a model naming a domain nothing can route to is a failure, not a new domain.
    """
    if allowed_domains is None:
        # Imported lazily: safety must not drag the router in at import time,
        # so a test can exercise this file on its own.
        from router.policy import routable_domains
        allowed_domains = routable_domains()
    max_chars = max_chars if max_chars is not None else config.MAX_OUTPUT_CHARS

    if raw is None or not str(raw).strip():
        return ValidationResult(valid=False, reason="empty model output")
    if len(raw) > max_chars:
        return ValidationResult(
            valid=False, reason=f"output is {len(raw)} chars, over the {max_chars} limit")

    parsed, repaired, error = _parse(raw)
    if parsed is None:
        return ValidationResult(valid=False, reason=error, repaired=repaired)
    if not isinstance(parsed, dict):
        return ValidationResult(valid=False, repaired=repaired,
                                reason=f"expected a JSON object, got {type(parsed).__name__}")

    missing = [f for f in REQUIRED_FIELDS if not str(parsed.get(f) or "").strip()]
    if missing:
        return ValidationResult(valid=False, data=parsed, repaired=repaired,
                                reason=f"missing required field(s): {', '.join(missing)}")

    domain = str(parsed["domain"]).strip()
    if domain not in allowed_domains:
        return ValidationResult(
            valid=False, data=parsed, repaired=repaired,
            reason=f"domain '{domain}' is not routable -- allowed: {allowed_domains}")

    urgency = str(parsed["urgency"]).strip().lower()
    if urgency not in ALLOWED_URGENCY:
        return ValidationResult(
            valid=False, data=parsed, repaired=repaired,
            reason=f"urgency '{urgency}' is not one of {sorted(ALLOWED_URGENCY)}")

    question = str(parsed["question"]).strip()
    if len(question) > config.MAX_QUESTION_CHARS:
        return ValidationResult(
            valid=False, data=parsed, repaired=repaired,
            reason=f"question is {len(question)} chars, over the "
                   f"{config.MAX_QUESTION_CHARS} limit")

    parsed["domain"], parsed["urgency"], parsed["question"] = domain, urgency, question
    return ValidationResult(valid=True, data=parsed, repaired=repaired)
