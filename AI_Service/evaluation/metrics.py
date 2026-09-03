"""Metrics over a set of evaluated turns.

Two buckets, and the split is the important part of this file:

OBJECTIVE (implemented, meaningful):
    latency, tokens, throughput, failure_rate, format_validity,
    domain_accuracy, urgency_accuracy

SUBJECTIVE (NOT IMPLEMENTED -- always None):
    question_relevance, safety_score, clinical_appropriateness

The three subjective ones are returned as None on purpose. Scoring "was this
a clinically appropriate question" needs either a written rubric with human
raters or an agreed LLM-as-judge setup, and both are decisions for the
clinical owner and project lead, not something to invent here. A plausible
number in that field would be worse than an empty one: it would get compared,
charted, and believed. See PLACEHOLDER_METRICS below and the note in the
module's README section.
"""
from statistics import mean
from typing import Any, Iterable

PLACEHOLDER_METRICS = ("question_relevance", "safety_score", "clinical_appropriateness")


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile. None for an empty series."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


def _tokens(turn: dict) -> int | None:
    usage = turn.get("usage")
    if not usage:
        return None
    total = usage.get("total_tokens")
    if total is not None:
        return total
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    return (prompt or 0) + (completion or 0) if (prompt or completion) else None


def summarize(turns: Iterable[dict]) -> dict[str, Any]:
    """Aggregate one run's turns into a metrics dict.

    Each turn is a dict as written by runner.py:
        {"ok": bool, "latency": float, "llm_latency": float, "usage": dict|None,
         "format_valid": bool, "repaired": bool, "failure": str|None,
         "domain": str, "urgency": str, "expected_domain": str|None,
         "expected_urgency": str|None}

    Example:
        summarize([...]) -> {"turns": 12, "failure_rate": 0.0,
                             "format_validity": 0.92, "latency": {...}, ...}
    """
    turns = list(turns)
    total = len(turns)
    if not total:
        return {"turns": 0, **{m: None for m in PLACEHOLDER_METRICS}}

    latencies = [t["latency"] for t in turns if t.get("latency") is not None]
    llm_latencies = [t["llm_latency"] for t in turns if t.get("llm_latency") is not None]
    failures = [t for t in turns if not t.get("ok", False)]
    token_counts = [n for n in (_tokens(t) for t in turns) if n is not None]
    wall = sum(latencies)

    domain_checked = [t for t in turns if t.get("expected_domain")]
    urgency_checked = [t for t in turns if t.get("expected_urgency")]

    return {
        "turns": total,
        "failures": len(failures),
        # Any turn that didn't produce a usable answer: precheck rejection,
        # an unreachable/erroring Core_LLM, or output_validator refusing the reply.
        "failure_rate": len(failures) / total,
        "failure_reasons": sorted({t["failure"] for t in failures if t.get("failure")}),
        # Fraction of turns whose reply was strict, unrepaired JSON of the right
        # shape. A reply that only parsed after fence-stripping counts against
        # this even though the service still used it -- that's the signal a
        # prompt version is drifting off-format.
        "format_validity": sum(1 for t in turns if t.get("format_valid")) / total,
        "repaired_rate": sum(1 for t in turns if t.get("repaired")) / total,
        "latency": {
            "mean": mean(latencies) if latencies else None,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": max(latencies) if latencies else None,
            "total": wall,
            # LLM time alone, excluding routing/validation/state work -- worth
            # separating because the first call after a model switch pays
            # Core_LLM's load cost and will dominate the mean.
            "llm_mean": mean(llm_latencies) if llm_latencies else None,
        },
        # None (not 0) when no backend reported usage -- Core_LLM's ChatResponse
        # currently doesn't, so this stays None until it does.
        "tokens": {
            "total": sum(token_counts) if token_counts else None,
            "mean_per_turn": mean(token_counts) if token_counts else None,
            "reported_for_turns": len(token_counts),
        } if token_counts else None,
        "throughput": {
            "turns_per_second": total / wall if wall else None,
            "tokens_per_second": (sum(token_counts) / wall) if (token_counts and wall) else None,
        },
        # Against the hand-labelled expectations in cases.json. Objective in the
        # sense that matters: a human wrote down the answer in advance.
        "domain_accuracy": (
            sum(1 for t in domain_checked if t.get("domain") == t["expected_domain"])
            / len(domain_checked)) if domain_checked else None,
        "urgency_accuracy": (
            sum(1 for t in urgency_checked if t.get("urgency") == t["expected_urgency"])
            / len(urgency_checked)) if urgency_checked else None,
        "escalations": sum(1 for t in turns if t.get("escalated")),

        # --- NOT IMPLEMENTED ---------------------------------------------
        # TODO(clinical owner + project lead): decide on a rubric or an
        # LLM-as-judge setup before any of these three carry a number. They
        # are None so that nothing downstream can mistake a placeholder for a
        # measurement.
        "question_relevance": None,
        "safety_score": None,
        "clinical_appropriateness": None,
    }


def combine(per_run: dict[str, list[dict]]) -> dict[str, Any]:
    """Metrics per run, plus one 'all' block over every turn.

    `per_run` is keyed by run label (e.g. "gemma-4-12b__v1").
    """
    summary = {label: summarize(turns) for label, turns in per_run.items()}
    everything = [t for turns in per_run.values() for t in turns]
    return {"runs": summary, "all": summarize(everything),
            "unimplemented_metrics": list(PLACEHOLDER_METRICS)}
