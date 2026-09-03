"""The Router: a thin wrapper over whichever policy version is configured.

Kept separate from policy.py so that callers depend on a stable contract
(`route(RouterInput) -> RouterOutput`) while policies come and go. It also
owns the two things that aren't a policy's business: the ROUTER_ENABLED kill
switch, and stamping the policy version onto every decision.
"""
import config
from schemas.routing import RouterInput, RouterOutput
from schemas.state import DEFAULT_DOMAIN, ConversationState

from . import policy


def route(router_input: RouterInput, version: str | None = None,
          model_key: str | None = None) -> RouterOutput:
    """Decide domain + model for one turn.

    Example:
        route(RouterInput(user_message="چشمم قرمز شده"))
        -> RoutingDecision(domain="eye", model_key="gemma-4-12b",
                           policy_version="v1", reason="keyword match: چشم")

    `model_key` forces a model while leaving domain detection alone -- the
    evaluation harness needs exactly that, to compare two models on identical
    routing. With `config.ROUTER_ENABLED` false, domain detection is skipped
    too and everything lands on the default domain/model.
    """
    version = version or config.ROUTING_POLICY_VERSION

    if not config.ROUTER_ENABLED:
        state_domain = router_input.conversation_state.domain
        decision = RouterOutput(
            domain=state_domain or DEFAULT_DOMAIN,
            model_key=model_key or config.DEFAULT_MODEL_KEY,
            policy_version=f"{version} (router disabled)",
            reason="ROUTER_ENABLED=false -- routing bypassed",
        )
        return decision

    decision = policy.route(router_input.user_message,
                            router_input.conversation_state, version)
    if model_key:
        decision = decision.model_copy(update={
            "model_key": model_key,
            "reason": f"{decision.reason}; model forced by caller",
        })
    # policy.route() already stamps this; re-stated here because the stamp is
    # the Router's promise, not the policy's.
    decision.policy_version = decision.policy_version or version
    return decision


def route_message(user_message: str, conversation_state: ConversationState | None = None,
                  version: str | None = None, model_key: str | None = None) -> RouterOutput:
    """Convenience form for callers holding a message and a state, not a RouterInput."""
    return route(
        RouterInput(user_message=user_message,
                    conversation_state=conversation_state or ConversationState()),
        version=version, model_key=model_key,
    )
