"""Router: sample inputs map to the expected domain and model under policy v1."""
import pytest

import config
from router import policy, router
from schemas.routing import RouterInput
from schemas.state import ConversationState


@pytest.mark.parametrize("message,expected", [
    ("چشمم از دیروز قرمز شده", "eye"),
    ("دید چشم راستم تار شده است", "eye"),
    ("پلک بالایی‌ام ورم کرده", "eye"),
    ("My eye is red and painful", "eye"),
    ("از دیروز سرم درد می‌کند", "general"),
    ("سلام", "general"),
])
def test_v1_domain_detection(message, expected):
    assert router.route_message(message).domain == expected


def test_decision_carries_policy_version_and_reason():
    decision = router.route_message("چشمم درد می‌کند")
    assert decision.policy_version == config.ROUTING_POLICY_VERSION
    assert "keyword" in (decision.reason or "")


def test_existing_domain_wins_over_keywordless_message():
    """Mid-interview answers like 'بله' carry no keywords -- re-detecting per
    turn would drop the conversation back to `general` on every short reply."""
    state = ConversationState(domain="eye", slots={"onset": "sudden"})
    decision = router.route_message("بله", state)
    assert decision.domain == "eye"
    assert "already set" in (decision.reason or "")


def test_model_is_licensed_by_default():
    """ALLOW_NON_COMMERCIAL_MODELS is false by default, so a CC-BY-NC model
    must never be selected."""
    for message in ("چشمم قرمز شده", "سرم درد می‌کند"):
        assert router.route_message(message).model_key not in config.NON_COMMERCIAL_MODEL_KEYS


def test_non_commercial_model_only_when_explicitly_allowed(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_NON_COMMERCIAL_MODELS", False)
    monkeypatch.setitem(policy.DOMAIN_MODELS, "eye", ("aya-expanse-8b", "gemma-4-12b"))
    assert policy.pick_model("eye") == "gemma-4-12b"

    monkeypatch.setattr(config, "ALLOW_NON_COMMERCIAL_MODELS", True)
    assert policy.pick_model("eye") == "aya-expanse-8b"


def test_forced_model_key_keeps_routed_domain():
    """The evaluation harness pins a model while leaving routing alone."""
    decision = router.route_message("چشمم قرمز شده", model_key="gemma-4-31b")
    assert (decision.domain, decision.model_key) == ("eye", "gemma-4-31b")
    assert "forced" in (decision.reason or "")


def test_router_disabled_pins_domain_and_model(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_ENABLED", False)
    decision = router.route(RouterInput(user_message="چشمم قرمز شده"))
    assert decision.domain == "general"
    assert decision.model_key == config.DEFAULT_MODEL_KEY
    assert "disabled" in decision.policy_version


def test_unknown_policy_version_raises():
    with pytest.raises(policy.UnknownPolicyVersion, match="v99"):
        router.route_message("چشمم قرمز شده", version="v99")


def test_every_routed_model_exists_in_core_llm_registry():
    """Guards against a policy naming a model key Core_LLM doesn't serve.

    Core_LLM's registry is read from its README table rather than imported --
    CONTRIBUTING.md forbids cross-module imports, and this is the same list
    demo_app hard-codes.
    """
    core_llm_keys = {"aya-expanse-8b", "aya-expanse-32b", "gemma-4-31b",
                     "gemma-4-e4b", "gemma-4-12b", "qwen3-omni-30b"}
    for chain in policy.DOMAIN_MODELS.values():
        assert set(chain) <= core_llm_keys
    assert config.DEFAULT_MODEL_KEY in core_llm_keys


def test_every_routable_domain_has_a_model_chain():
    for domain in policy.routable_domains():
        assert policy.pick_model(domain)
