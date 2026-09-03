"""Conversation state: slot extraction, merging, and domain handling."""
import pytest

from orchestrator import state as state_mod
from schemas.state import ConversationState


def eye_state(turn_count=1, **slots):
    state = state_mod.new_state("eye")
    state.slots.update(slots)
    state.turn_count = turn_count
    return state


# --- creation ---------------------------------------------------------------

def test_new_state_declares_every_slot_unanswered():
    state = state_mod.new_state("eye")
    assert state.slots == {"onset": None, "redness": None, "vision_change": None}
    assert state.missing() == ["onset", "redness", "vision_change"]
    assert state.is_complete() is False


def test_new_state_for_unknown_domain_has_no_slots():
    assert state_mod.new_state("dermatology").slots == {}


def test_general_domain_is_never_complete():
    """It has no form to finish, so 'complete' would be meaningless."""
    assert state_mod.new_state("general").is_complete() is False


# --- extraction -------------------------------------------------------------

@pytest.mark.parametrize("message,expected", [
    ("از دیروز یک‌دفعه شروع شد", {"onset": "sudden"}),          # ZWNJ inside یک‌دفعه
    ("از سه روز پیش کم‌کم بدتر شد", {"onset": "gradual"}),
    ("از دو هفته پیش", {"onset": "reported"}),
    ("چشمم قرمز است", {"redness": "yes"}),
    ("دید چشمم تار شده", {"vision_change": "blurred"}),
    ("دوبینی دارم", {"vision_change": "double"}),
    ("دیدم کم شده", {"vision_change": "decreased"}),
    ("دیدم را از دست دادم", {"vision_change": "sudden_loss"}),
])
def test_extraction_normalises_values(message, expected):
    assert state_mod.extract_slots(message, eye_state()) == expected


def test_negation_within_the_clause_is_believed():
    assert state_mod.extract_slots("چشمم قرمز نیست", eye_state()) == {"redness": "no"}
    assert state_mod.extract_slots("بدون قرمزی", eye_state()) == {"redness": "no"}


def test_negation_in_a_later_clause_is_not_applied_to_an_earlier_symptom():
    """'...از دست داده‌ام و چیزی نمی‌بینم' confirms the loss; a fixed-width
    negation window would read the نمی as denying it."""
    found = state_mod.extract_slots("دیدم را از دست داده‌ام و چیزی نمی‌بینم", eye_state())
    assert found == {"vision_change": "sudden_loss"}


def test_arabic_characters_and_digits_are_normalised():
    """Some keyboards produce ي/ك and Arabic-Indic digits."""
    assert state_mod.extract_slots("چشمم قرمز شده يعني ٢ روز", eye_state()) == {
        "redness": "yes", "onset": "reported"}


def test_positional_fallback_answers_the_pending_slot():
    state = eye_state(turn_count=1)
    assert state_mod.extract_slots("از سه‌شنبه", state) == {"onset": "از سه‌شنبه"}


def test_no_positional_fallback_on_the_first_turn():
    """The opening message is the chief complaint, not an answer -- filing it
    under `onset` would lose it AND block the real answer, since merge_slots
    never overwrites."""
    state = state_mod.new_state("eye")  # turn_count == 0
    assert state_mod.extract_slots("توپ محکم به چشمم خورد", state) == {}


def test_empty_message_extracts_nothing():
    assert state_mod.extract_slots("   ", eye_state()) == {}


# --- merging ----------------------------------------------------------------

def test_merge_never_clobbers_an_answered_slot():
    state = eye_state(redness="no")
    state_mod.merge_slots(state, {"redness": "yes"})
    assert state.slots["redness"] == "no"


def test_merge_fills_only_empty_slots():
    state = eye_state(onset="sudden")
    state_mod.merge_slots(state, {"onset": "gradual", "redness": "yes"})
    assert state.slots == {"onset": "sudden", "redness": "yes", "vision_change": None}


def test_merge_ignores_slots_the_domain_does_not_declare():
    state = eye_state()
    state_mod.merge_slots(state, {"invented_slot": "x", "redness": "yes"})
    assert "invented_slot" not in state.slots


def test_merge_ignores_empty_values():
    state = eye_state()
    state_mod.merge_slots(state, {"redness": None, "onset": ""})
    assert state.missing() == ["onset", "redness", "vision_change"]


def test_advance_counts_the_turn():
    state = state_mod.new_state("eye")
    state_mod.advance(state, "چشمم قرمز شده")
    assert state.turn_count == 1
    assert state.slots["redness"] == "yes"


# --- domain handling --------------------------------------------------------

def test_ensure_domain_creates_state_for_a_new_conversation():
    assert state_mod.ensure_domain(None, "eye").domain == "eye"


def test_ensure_domain_keeps_answers_when_the_domain_is_unchanged():
    state = eye_state(onset="sudden")
    assert state_mod.ensure_domain(state, "eye").slots["onset"] == "sudden"


def test_ensure_domain_drops_slots_the_new_domain_does_not_share():
    state = eye_state(onset="sudden", redness="yes")
    switched = state_mod.ensure_domain(state, "general")
    assert switched.domain == "general"
    assert switched.slots == {}


def test_ensure_domain_backfills_a_newly_added_slot():
    """A state stored before the catalogue grew must not report complete."""
    state = ConversationState(domain="eye", slots={"onset": "sudden"})
    backfilled = state_mod.ensure_domain(state, "eye")
    assert backfilled.missing() == ["redness", "vision_change"]


def test_unknown_domain_is_handled_without_raising():
    state = state_mod.ensure_domain(None, "cardiology")
    assert state_mod.extract_slots("هر چیزی", state) == {}
    assert state_mod.merge_slots(state, {"x": "y"}).slots == {}
