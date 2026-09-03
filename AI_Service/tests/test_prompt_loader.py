"""prompt_loader: message construction and ordering, for both v1 and v2."""
import json

import pytest

from prompts.prompt_loader import (
    STATE_HEADER,
    UnknownPromptVersion,
    available_versions,
    load_fewshot,
    load_messages,
    load_system,
)
from schemas.response import Message
from schemas.state import ConversationState


def test_both_versions_exist():
    assert available_versions() == ["v1", "v2"]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_order_is_system_fewshot_history_state_user(version):
    history = [Message(role="user", content="قبلی"),
               Message(role="assistant", content="جواب قبلی")]
    messages = load_messages(version, ConversationState(domain="eye"), history, "پیام تازه")

    assert messages[0].role == "system"
    assert messages[0].content == load_system(version)

    fewshot_count = 2 * len(load_fewshot(version))
    fewshot = messages[1:1 + fewshot_count]
    assert [m.role for m in fewshot] == ["user", "assistant"] * len(load_fewshot(version))

    after_fewshot = messages[1 + fewshot_count:]
    assert after_fewshot[0].content == "قبلی"
    assert after_fewshot[1].content == "جواب قبلی"
    assert after_fewshot[2].role == "system"
    assert after_fewshot[2].content.startswith(STATE_HEADER)
    assert after_fewshot[3] == Message(role="user", content="پیام تازه")
    assert len(after_fewshot) == 4


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_fewshot_assistant_turns_are_json_objects(version):
    """The examples must model the exact output format, or they teach nothing."""
    for example in load_fewshot(version):
        parsed = example["assistant"]
        assert set(parsed) == {"domain", "model", "question", "urgency"}
        assert parsed["urgency"] in {"routine", "urgent"}


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_each_version_demonstrates_urgent(version):
    """A few-shot set that only ever shows 'routine' never exercises urgency."""
    urgencies = {e["assistant"]["urgency"] for e in load_fewshot(version)}
    assert "urgent" in urgencies


def test_v2_is_not_a_copy_of_v1():
    """v2 exists to be A/B'd against v1 -- an identical variant measures nothing."""
    assert load_system("v1") != load_system("v2")
    assert load_fewshot("v1") != load_fewshot("v2")


def test_state_message_carries_filled_missing_and_model():
    state = ConversationState(domain="eye",
                              slots={"onset": "sudden", "redness": None,
                                     "vision_change": None},
                              turn_count=2)
    messages = load_messages("v1", state, [], "x", model_key="gemma-4-12b")
    payload = json.loads(messages[-2].content[len(STATE_HEADER):])
    assert payload["filled"] == {"onset": "sudden"}
    assert payload["missing"] == ["redness", "vision_change"]
    assert payload["all_slots"] == ["onset", "redness", "vision_change"]
    assert payload["model"] == "gemma-4-12b"
    assert payload["turn_count"] == 2


def test_fewshot_version_can_differ_from_system_version():
    mixed = load_messages("v1", ConversationState(domain="eye"), [], "x",
                          fewshot_version="v2")
    assert mixed[0].content == load_system("v1")
    assert len(mixed) == 1 + 2 * len(load_fewshot("v2")) + 2


def test_unknown_version_raises_instead_of_falling_back():
    """A silent fallback to v1 would make an evaluation report a version it
    never actually ran."""
    with pytest.raises(UnknownPromptVersion, match="v99"):
        load_messages("v99", ConversationState(), [], "x")
    with pytest.raises(UnknownPromptVersion):
        load_messages("v1", ConversationState(), [], "x", fewshot_version="v99")


def test_no_history_is_fine():
    assert load_messages("v1", ConversationState(), None, "x")[-1].content == "x"
