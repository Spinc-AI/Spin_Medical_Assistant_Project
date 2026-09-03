"""The per-turn interview loop -- the piece everything else exists to serve.

One turn, in order:

    precheck(user input)
      -> router.route()            domain + model_key
      -> state.ensure_domain()     state shaped for that domain
      -> state.advance()           extract + merge the patient's answer
      -> prompt_loader             system + fewshot + history + state + message
      -> llm.generate()            HTTP to Core_LLM
      -> output_validator          is the reply the JSON we asked for?
      -> postcheck                 raise urgency if the slots demand it
      -> (InterviewResponse, ConversationState)

Nothing here is domain-specific: domains come from `schemas.state.DOMAIN_SLOTS`
and the rules in `orchestrator/state.py`, so a second domain needs no change
in this file.

The LLM client is injected (`llm_client=`), which is what makes the whole loop
testable without a GPU -- the tests and the evaluation harness pass a fake that
returns canned JSON.
"""
import time
from dataclasses import dataclass, field

import config
from llm.core_llm_client import CoreLLMClient, CoreLLMError
from llm.interface import LLMClient, LLMResponse
from prompts import prompt_loader
from router import router
from safety import postcheck as postcheck_mod
from safety import precheck as precheck_mod
from safety.output_validator import validate_output
from schemas.response import InterviewRequest, InterviewResponse, Message, Urgency
from schemas.state import ConversationState

from . import state as state_mod

_DEFAULT_CLIENT: LLMClient | None = None


def default_client() -> LLMClient:
    """The process-wide CoreLLMClient, built on first use.

    Lazy so that importing this module never opens a connection or requires
    Core_LLM to be up -- tests and `python -c` imports must stay free.
    """
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = CoreLLMClient()
    return _DEFAULT_CLIENT


@dataclass
class TurnTrace:
    """Everything about one turn that isn't part of the patient-facing answer.

    The evaluation harness reads this (latency, validity, failures); nothing in
    the HTTP response depends on it, so it can grow without touching the API.
    """

    ok: bool = True
    latency: float = 0.0
    llm_latency: float = 0.0
    usage: dict | None = None
    model_key: str = ""
    prompt_version: str = ""
    policy_version: str = ""
    format_valid: bool = False
    repaired: bool = False
    failure: str | None = None
    notes: list[str] = field(default_factory=list)


def _refusal(request: InterviewRequest, state: ConversationState, reason: str,
             code: str, decision_domain: str | None = None,
             model_key: str = "") -> InterviewResponse:
    """The response for a turn that never reached the model (precheck, or an
    unusable reply). Still a well-formed InterviewResponse -- a caller should
    not have to branch on error shapes to keep a conversation going."""
    return InterviewResponse(
        domain=decision_domain or state.domain or "general",
        model=model_key,
        question=reason,
        urgency=Urgency.ROUTINE,
        session_id=request.session_id,
        conversation_state=state,
        complete=state.is_complete(),
        policy_version=request.policy_version or config.ROUTING_POLICY_VERSION,
        prompt_version=request.prompt_version or config.PROMPT_VERSION,
        safety_version=config.SAFETY_VERSION,
        notes=[f"turn rejected: {code}"],
    )


def run_turn(request: InterviewRequest, state: ConversationState | None = None,
             llm_client: LLMClient | None = None,
             history: list[Message] | None = None,
             trace: TurnTrace | None = None,
             ) -> tuple[InterviewResponse, ConversationState]:
    """Run one interview turn. Returns the response and the updated state.

    Example:
        resp, state = run_turn(InterviewRequest(user_message="چشمم قرمز شده"))
        resp.question   # the next question to ask
        state.slots     # what's been recorded so far -- pass this back next turn

    `state` is the state from the previous turn (None starts a conversation).
    `history` is the prior user/assistant turns for prompt context; the caller
    owns it (main.py keeps it per session).

    Pass a `TurnTrace` as `trace` to have the per-turn measurements written
    into it (latency, format validity, failures). evaluation/runner.py does;
    main.py doesn't need to. It's an out-parameter rather than a third return
    value so the (response, state) contract stays as specified.
    """
    started = time.perf_counter()
    client = llm_client or default_client()
    prompt_version = request.prompt_version or config.PROMPT_VERSION
    fewshot_version = request.prompt_version or config.FEWSHOT_VERSION
    policy_version = request.policy_version or config.ROUTING_POLICY_VERSION
    state = request.conversation_state or state or ConversationState()
    trace = trace if trace is not None else TurnTrace()
    trace.prompt_version = prompt_version
    trace.policy_version = policy_version

    # 1. Safety: the raw input, before it costs a model call.
    gate = precheck_mod.precheck(request.user_message)
    if not gate.ok:
        trace.ok = False
        trace.failure = f"precheck:{gate.code}"
        trace.latency = time.perf_counter() - started
        response = _refusal(request, state, gate.reason or "ورودی پذیرفته نشد.",
                            gate.code or "precheck")
        return response, state

    # 2. Route: which domain are we in, and which model answers.
    decision = router.route_message(request.user_message, state,
                                    version=policy_version, model_key=request.model_key)
    trace.model_key = decision.model_key
    trace.policy_version = decision.policy_version
    state = state_mod.ensure_domain(state, decision.domain)

    # 3. Record what the patient just told us -- BEFORE building the prompt,
    # so the state block the model sees already reflects this message. Doing it
    # after the call (the obvious order) means the model is told a slot is
    # still missing when the message in front of it just answered that slot,
    # and it dutifully asks the question again.
    state = state_mod.advance(state, request.user_message)

    # 4. Build the prompt for this turn.
    messages = prompt_loader.load_messages(
        prompt_version, state, history, request.user_message,
        fewshot_version=fewshot_version, model_key=decision.model_key,
    )

    # 5. Ask the model.
    try:
        llm_response: LLMResponse = client.generate(
            prompt_loader.as_dicts(messages),
            temperature=request.temperature,
            max_tokens=config.MAX_TOKENS,
            model=decision.model_key,
        )
    except CoreLLMError as exc:
        trace.ok = False
        trace.failure = f"llm_error:{exc}"
        trace.latency = time.perf_counter() - started
        response = _refusal(request, state,
                            "سرویس مدل در دسترس نیست؛ لطفاً دوباره تلاش کنید.",
                            "llm_unavailable", decision.domain, decision.model_key)
        return response, state
    trace.llm_latency = llm_response.latency
    trace.usage = llm_response.usage

    # 6. Safety: is the reply structurally what we asked for?
    validation = validate_output(llm_response.text)
    trace.format_valid = validation.valid and not validation.repaired
    trace.repaired = validation.repaired
    if not validation.valid:
        trace.ok = False
        trace.failure = f"output_invalid:{validation.reason}"
        trace.latency = time.perf_counter() - started
        # The patient still gets a usable turn: re-ask the slot the interview
        # was already on, rather than surfacing a parse error to them.
        response = _refusal(request, state, _fallback_question(state),
                            "output_invalid", decision.domain, decision.model_key)
        response.notes.append(f"output_validator: {validation.reason}")
        return response, state

    data = validation.data or {}

    response = InterviewResponse(
        domain=decision.domain,
        # The router's choice, not the model's self-report -- the model can
        # echo the wrong name and it changes nothing about what actually ran.
        model=decision.model_key,
        question=str(data["question"]),
        urgency=Urgency(data["urgency"]),
        session_id=request.session_id,
        conversation_state=state,
        complete=state.is_complete(),
        policy_version=decision.policy_version,
        prompt_version=prompt_version,
        safety_version=config.SAFETY_VERSION,
        latency=llm_response.latency,
    )

    # 7. Safety: does the state demand a higher urgency than the model gave?
    checked = postcheck_mod.postcheck(response, state)
    response = checked.response
    response.notes.extend(checked.notes)
    if validation.repaired:
        response.notes.append("output needed repair before parsing (not strict JSON)")
    trace.notes = list(response.notes)
    trace.latency = time.perf_counter() - started
    return response, state


def _fallback_question(state: ConversationState) -> str:
    """What to ask when the model's reply was unusable.

    Re-asks the slot the interview is already on, in plain wording, so a bad
    turn costs a repeat rather than ending the conversation.
    """
    missing = state.missing()
    prompts = {
        "onset": "این مشکل از چه زمانی شروع شده و ناگهانی بود یا تدریجی؟",
        "redness": "آیا چشم شما قرمز شده است؟",
        "vision_change": "آیا در دید شما تغییری ایجاد شده است (تاری، دوبینی یا کاهش دید)؟",
    }
    if missing and missing[0] in prompts:
        return prompts[missing[0]]
    return "لطفاً کمی بیشتر درباره‌ی علائم خود توضیح دهید."
