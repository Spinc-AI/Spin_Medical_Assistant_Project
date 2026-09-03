"""End-to-end interview turns against a fake LLM client -- no Core_LLM, no GPU."""
import pytest
from conftest import FakeLLMClient, reply

import config
from llm.core_llm_client import CoreLLMUnavailable
from orchestrator import interview
from schemas.response import InterviewRequest, Message, Urgency


def run(message, state=None, client=None, history=None, **request_kwargs):
    trace = interview.TurnTrace()
    response, state = interview.run_turn(
        InterviewRequest(user_message=message, **request_kwargs),
        state, llm_client=client or FakeLLMClient(), history=history, trace=trace)
    return response, state, trace


# --- a single turn ----------------------------------------------------------

def test_first_turn_routes_fills_and_answers():
    response, state, trace = run("چشمم از دیروز قرمز شده")
    assert response.domain == "eye"
    assert response.model == "gemma-4-12b"
    assert response.question
    assert response.urgency is Urgency.ROUTINE
    assert state.slots["redness"] == "yes"
    assert trace.ok and trace.format_valid


def test_response_stamps_every_version():
    response, _, _ = run("چشمم قرمز شده")
    assert response.policy_version == config.ROUTING_POLICY_VERSION
    assert response.prompt_version == config.PROMPT_VERSION
    assert response.safety_version == config.SAFETY_VERSION


def test_the_router_not_the_model_decides_which_model_ran():
    """A model echoing the wrong name back changes nothing about what ran."""
    client = FakeLLMClient([reply(model="some-other-model")])
    response, _, _ = run("چشمم قرمز شده", client=client)
    assert response.model == "gemma-4-12b"
    assert client.calls[0]["model"] == "gemma-4-12b"


def test_prompt_and_model_overrides_reach_the_client():
    client = FakeLLMClient()
    run("چشمم قرمز شده", client=client, model_key="gemma-4-31b", prompt_version="v2")
    assert client.calls[0]["model"] == "gemma-4-31b"
    from prompts.prompt_loader import load_system
    assert client.calls[0]["messages"][0]["content"] == load_system("v2")


# --- a whole conversation ---------------------------------------------------

def test_multi_turn_conversation_fills_state_and_completes():
    client = FakeLLMClient([reply("چه زمانی شروع شد؟"),
                            reply("آیا قرمزی دارید؟"),
                            reply("تغییری در دید دارید؟")])
    state, history, responses = None, [], []
    for message in ["چشمم اذیت می‌کند", "از سه روز پیش کم‌کم شروع شد",
                    "بله قرمز است", "نه، دیدم خوب است"]:
        response, state, _ = run(message, state, client, history)
        history += [Message(role="user", content=message),
                    Message(role="assistant", content=response.question)]
        responses.append(response)

    assert [r.domain for r in responses] == ["eye"] * 4
    assert state.slots["onset"] == "gradual"
    assert state.slots["redness"] == "yes"
    assert state.turn_count == 4
    assert state.is_complete()
    assert responses[-1].complete is True


def test_domain_sticks_once_set():
    """Later turns carry no eye keyword at all."""
    client = FakeLLMClient()
    _, state, _ = run("چشمم درد می‌کند", client=client)
    for message in ["از دیروز", "بله", "نه"]:
        response, state, _ = run(message, state, client)
        assert response.domain == "eye"


def test_history_is_passed_to_the_model():
    client = FakeLLMClient()
    history = [Message(role="user", content="قبلی"),
               Message(role="assistant", content="پاسخ قبلی")]
    run("پیام تازه", client=client, history=history)
    contents = [m["content"] for m in client.calls[0]["messages"]]
    assert "قبلی" in contents and "پاسخ قبلی" in contents


# --- safety is actually wired in -------------------------------------------

def test_precheck_rejection_never_reaches_the_model():
    client = FakeLLMClient()
    response, state, trace = run("   ", client=client)
    assert client.calls == []
    assert trace.ok is False and trace.failure == "precheck:empty_input"
    assert response.notes == ["turn rejected: empty_input"]


def test_rejected_turn_preserves_collected_state():
    client = FakeLLMClient()
    _, state, _ = run("چشمم قرمز شده", client=client)
    _, state_after, _ = run("   ", state, client)
    assert state_after.slots["redness"] == "yes"


def test_malformed_output_falls_back_to_a_usable_question():
    client = FakeLLMClient(["این JSON نیست"])
    response, _, trace = run("چشمم قرمز شده", client=client)
    assert trace.ok is False
    assert trace.failure.startswith("output_invalid")
    assert response.question  # the patient still gets asked something
    assert any("output_validator" in n for n in response.notes)


def test_repaired_output_is_used_but_flagged():
    client = FakeLLMClient([f"```json\n{reply()}\n```"])
    response, _, trace = run("چشمم قرمز شده", client=client)
    assert trace.repaired is True
    assert trace.format_valid is False
    assert any("repair" in n for n in response.notes)


def test_postcheck_escalates_when_the_model_under_calls_urgency():
    client = FakeLLMClient([reply(urgency="routine")])
    state, history = None, []
    for message in ["چشمم قرمز شده", "از دیشب ناگهانی",
                    "دیدم را از دست داده‌ام"]:
        response, state, _ = run(message, state, client, history)
    assert state.slots["vision_change"] == "sudden_loss"
    assert response.urgency is Urgency.URGENT
    assert any("escalated" in n for n in response.notes)


def test_llm_failure_becomes_a_clean_turn_not_an_exception():
    class Broken(FakeLLMClient):
        def generate(self, *a, **kw):
            raise CoreLLMUnavailable("Core_LLM is down")

    response, _, trace = run("چشمم قرمز شده", client=Broken())
    assert trace.ok is False and trace.failure.startswith("llm_error")
    assert response.notes == ["turn rejected: llm_unavailable"]


def test_caller_supplied_state_wins_over_the_server_copy():
    from schemas.state import ConversationState
    supplied = ConversationState(domain="eye", slots={"onset": "sudden"})
    response, state, _ = run("بله قرمز است", conversation_state=supplied)
    assert state.slots["onset"] == "sudden"
    assert state.slots["redness"] == "yes"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_both_prompt_versions_drive_a_full_conversation(version):
    client = FakeLLMClient()
    state = None
    for message in ["چشمم قرمز شده", "از دیروز ناگهانی", "دیدم تار است"]:
        response, state, trace = run(message, state, client, prompt_version=version)
        assert trace.ok
    assert state.is_complete()
