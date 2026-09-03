"""Routing policies: message + state -> (domain, model_key).

Versioned and pluggable. `POLICIES` maps a version string to a function with
this signature:

    route(user_message: str, conversation_state: ConversationState,
          version: str) -> RoutingDecision

v1 is keyword/rule-based, which is enough while `eye` is the only specified
domain. The signature is what matters: an LLM-based classifier registered as
"v2" would take the same arguments and return the same RoutingDecision, so
router.py -- and everything above it -- would not change at all. That is why
`user_message` and the full state are both passed even though v1 barely looks
at the state.
"""
import re

import config
from schemas.routing import RoutingDecision
from schemas.state import DEFAULT_DOMAIN, DOMAIN_SLOTS, ConversationState


class UnknownPolicyVersion(ValueError):
    """Asked for a routing policy version that isn't registered."""


# --- v1: keyword rules ------------------------------------------------------

# Persian and English triggers per domain. Matched case-insensitively as whole
# words where the script allows it. Deliberately small: a keyword list that
# tries to cover everything ends up matching things it shouldn't, and the
# fallback (`general`) is a safe place to land.
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "eye": (
        "چشم", "چشمم", "چشمان", "بینایی", "دید", "قرمزی چشم", "تاری دید",
        "پلک", "اشک", "قرنیه", "عنبیه", "شبکیه", "لنز", "عینک",
        "eye", "eyes", "vision", "blurry", "eyelid", "retina", "cornea",
    ),
}

# Which model serves which domain, best first. Every key here must exist in
# Core_LLM's MODEL_REGISTRY (GET /models). The fallback chain matters because
# a preferred model can be filtered out by licensing (see below).
DOMAIN_MODELS: dict[str, tuple[str, ...]] = {
    # gemma-4-12b: the largest audio-capable Gemma 4, so the same loaded model
    # can serve a future voice-driven interview without a reload.
    "eye": ("gemma-4-12b", "gemma-4-31b", "aya-expanse-8b"),
    "general": ("gemma-4-12b", "gemma-4-31b", "aya-expanse-8b"),
}


def _licensed(model_key: str) -> bool:
    """False for CC-BY-NC models unless config explicitly allows them.

    Aya Expanse is CC-BY-NC (non-commercial). Whether this deployment may use
    it is a licensing question the team must answer before go-live, so it is a
    config flag (ALLOW_NON_COMMERCIAL_MODELS, default false) rather than a
    hard-coded choice in either direction. With the default, the router simply
    steps past Aya to the next model in the chain.
    """
    if model_key not in config.NON_COMMERCIAL_MODEL_KEYS:
        return True
    return config.ALLOW_NON_COMMERCIAL_MODELS


def pick_model(domain: str) -> str:
    """First licensed model preferred for `domain`."""
    chain = DOMAIN_MODELS.get(domain) or DOMAIN_MODELS[DEFAULT_DOMAIN]
    for key in chain:
        if _licensed(key):
            return key
    # Every preferred model is licence-blocked. Falling back to the configured
    # default (rather than raising) keeps the service answering; if that one is
    # blocked too, the operator has misconfigured this deliberately.
    return config.DEFAULT_MODEL_KEY


def detect_domain(user_message: str) -> tuple[str | None, str | None]:
    """(domain, reason) from keywords alone, or (None, None) if nothing matched."""
    text = (user_message or "").casefold()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for word in keywords:
            # \b works for the ASCII triggers; the Persian ones are matched as
            # plain substrings, which is right for an agglutinative script
            # where "چشمم" is "چشم" + a possessive suffix.
            pattern = rf"\b{re.escape(word)}\b" if word.isascii() else re.escape(word)
            if re.search(pattern, text):
                return domain, f"keyword match: {word}"
    return None, None


def route_v1(user_message: str, conversation_state: ConversationState,
             version: str = "v1") -> RoutingDecision:
    """Keyword rules, with the conversation's existing domain winning.

    Once a conversation has a domain, it keeps it: a mid-interview message
    like "بله" carries no keywords, and re-detecting per turn would drop the
    conversation back to `general` on every short answer. Switching domains
    mid-conversation is a real scenario, but it needs a deliberate rule (and
    probably a classifier) rather than falling out of an accident.
    """
    if conversation_state.domain and conversation_state.domain in DOMAIN_SLOTS:
        domain = conversation_state.domain
        reason = "state: domain already set"
    else:
        detected, reason = detect_domain(user_message)
        domain = detected or DEFAULT_DOMAIN
        reason = reason or "no keyword matched -- fell back to the default domain"

    return RoutingDecision(domain=domain, model_key=pick_model(domain),
                           policy_version=version, reason=reason)


POLICIES = {"v1": route_v1}


def route(user_message: str, conversation_state: ConversationState,
          version: str | None = None) -> RoutingDecision:
    """Run the named policy version (config.ROUTING_POLICY_VERSION by default)."""
    version = version or config.ROUTING_POLICY_VERSION
    policy = POLICIES.get(version)
    if policy is None:
        raise UnknownPolicyVersion(
            f"unknown routing policy '{version}' -- available: {sorted(POLICIES)}")
    return policy(user_message, conversation_state, version)


def routable_domains() -> list[str]:
    """Domains this service can currently route to -- used by output_validator
    to reject a domain the model invented."""
    return sorted(DOMAIN_SLOTS)
