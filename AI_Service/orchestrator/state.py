"""Creating conversation state, extracting slot values, and merging them.

Split out from interview.py because these are the only rules that are
domain-specific: interview.py runs the same loop whatever the domain is, and
everything it needs to know about a domain's shape comes from here plus
`schemas.state.DOMAIN_SLOTS`.

CLINICAL REVIEW NEEDED on `EXTRACTION_RULES`. The patterns below are a
deliberately small, readable first pass so the interview can fill slots
without a second LLM call. They are engineering placeholders: whoever owns
clinical safety decides what actually counts as "sudden vision loss" in a
patient's own words, and in which dialects. Treat every value here as
provisional.
"""
import re
from typing import Any

from schemas.state import DOMAIN_SLOTS, ConversationState, slots_for

# Persian text arrives with variation that is invisible on screen but breaks a
# plain regex: ZWNJ (U+200C) inside "یک‌دفعه", Arabic ي/ك instead of Persian
# ی/ک from some keyboards, Arabic-Indic digits. Normalising once here is much
# less error-prone than writing every pattern to tolerate all of it.
ZWNJ = "‌"
CHAR_MAP = {ZWNJ: " ", "ي": "ی", "ك": "ک", "ﻙ": "ک", "ة": "ه", "أ": "ا", "إ": "ا", "ؤ": "و"}
DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")})

# Clause boundaries. Negation is only believed inside the clause the symptom
# was mentioned in -- see _is_negated.
CLAUSE_BOUNDARY = re.compile(r"\s+و\s+|[،,.!?؟؛;\n]")

# Negating words, checked AFTER the keyword (Persian puts the negated verb at
# the end of its clause: "قرمز نیست").
NEGATIONS_AFTER = ("نیست", "ندارم", "نداره", "ندارد", "نشده", "نبود", "نمی", " no ", " not ")
# ...and BEFORE it ("بدون قرمزی", "هیچ قرمزی").
NEGATIONS_BEFORE = ("بدون", "هیچ")

# domain -> slot -> ((pattern, normalised value), ...). The first pattern that
# matches wins, so order these most-specific-first. Patterns are matched
# against the NORMALISED text (see _normalize).
EXTRACTION_RULES: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "eye": {
        "onset": (
            (r"ناگهان|یک\s*دفعه|یکهو|یهو|sudden", "sudden"),
            (r"تدریج|کم\s*کم|به\s*مرور|gradual", "gradual"),
            # A bare time expression with no sudden/gradual marker still
            # answers "when did it start" -- recorded as answered, unspecified.
            (r"\d+\s*(روز|هفته|ماه|ساعت|سال)|دیروز|امروز|دیشب|امشب|هفته", "reported"),
        ),
        "redness": (
            (r"قرمز|سرخ|red\b", "yes"),
        ),
        "vision_change": (
            (r"از دست داد|نمی\s*بینم|کور شد|قطع شد|blind|lost (my )?vision", "sudden_loss"),
            (r"کاهش (شدید )?دید|کم شد(ن)? دید|دید(م|ت|ش)?\s*کم شد|افت دید"
             r"|vision loss|decreased vision", "decreased"),
            (r"دوبینی|دو\s*تا می\s*بینم|double", "double"),
            (r"تار|مه\s*آلود|blurr?", "blurred"),
        ),
    },
}


def _normalize(text: str) -> str:
    """Fold the script variations above away, and casefold."""
    text = (text or "").translate(DIGIT_MAP)
    for source, target in CHAR_MAP.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip().casefold()


def new_state(domain: str | None = None) -> ConversationState:
    """A fresh state for `domain`, with every slot present and unanswered.

    Example:
        new_state("eye")
        -> ConversationState(domain="eye",
                             slots={"onset": None, "redness": None,
                                    "vision_change": None},
                             turn_count=0)
    """
    return ConversationState(domain=domain,
                             slots={slot: None for slot in slots_for(domain)})


def ensure_domain(state: ConversationState | None, domain: str) -> ConversationState:
    """Return a state that is in `domain`, keeping what's still relevant.

    A new conversation gets `new_state`. A conversation that changes domain
    keeps only the slots the new domain also declares -- dropping the rest
    rather than carrying an eye answer into a skin interview.
    """
    if state is None:
        return new_state(domain)
    if state.domain == domain:
        # Backfill any slot the catalogue has gained since this state was made.
        for slot in slots_for(domain):
            state.slots.setdefault(slot, None)
        return state
    carried = {slot: state.slots.get(slot) for slot in slots_for(domain)}
    return ConversationState(domain=domain, slots=carried, turn_count=state.turn_count)


def _clause_around(text: str, start: int, end: int) -> tuple[str, str]:
    """(text before the match, text after it) within the same clause.

    Clause-scoped instead of a fixed character window, because a window is
    wrong in both directions: "از دست داده‌ام و چیزی نمی‌بینم" would look
    negated (the نمی belongs to the next clause and actually *confirms* the
    symptom), while a long "قرمزی ... در چشم راست وجود ندارد" would not.
    """
    before = text[:start]
    after = text[end:]
    boundaries = list(CLAUSE_BOUNDARY.finditer(before))
    if boundaries:
        before = before[boundaries[-1].end():]
    boundary = CLAUSE_BOUNDARY.search(after)
    if boundary:
        after = after[:boundary.start()]
    return before, after


def _is_negated(text: str, start: int, end: int) -> bool:
    before, after = _clause_around(text, start, end)
    return (any(neg.strip() in after for neg in NEGATIONS_AFTER)
            or any(neg in before for neg in NEGATIONS_BEFORE))


def extract_slots(user_message: str, state: ConversationState) -> dict[str, Any]:
    """Slot values found in one user message. Only returns what it found.

    Two mechanisms, in order:
      1. The pattern table above, which yields normalised values
         ("sudden_loss", "yes", ...) that postcheck.py can reason about.
      2. A positional fallback: if the message matched nothing AND a question
         has already been asked, the message is taken as the free-text answer
         to the slot the interview was on. Without it, a perfectly good answer
         the patterns don't recognise ("از سه‌شنبه") would leave the interview
         asking the same question forever.

         It deliberately does NOT apply on the first turn: that message is the
         presenting complaint ("توپ به چشمم خورد"), not an answer to anything,
         and filing it under `onset` would both lose it and -- since
         merge_slots never overwrites -- block the real answer that follows.

    Example:
        extract_slots("از دیشب یک‌دفعه دیدم را از دست دادم", new_state("eye"))
        -> {"onset": "sudden", "vision_change": "sudden_loss"}
    """
    text = _normalize(user_message)
    if not text:
        return {}

    rules = EXTRACTION_RULES.get(state.domain or "", {})
    found: dict[str, Any] = {}
    for slot, patterns in rules.items():
        for pattern, value in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            if _is_negated(text, match.start(), match.end()):
                found[slot] = "no" if value == "yes" else "none"
            else:
                found[slot] = value
            break

    if not found and state.turn_count > 0:
        pending = state.missing()
        if pending:
            found[pending[0]] = user_message.strip()
    return found


def merge_slots(state: ConversationState, values: dict[str, Any]) -> ConversationState:
    """Write `values` into `state`, never overwriting an answered slot.

    Once a patient has answered something, a later turn's incidental keyword
    match must not silently rewrite it -- a follow-up mentioning "قرمز" again
    shouldn't reset a redness answer that was recorded as "no". Correcting an
    earlier answer is a real need, but it has to be an explicit decision
    (an "actually, ..." intent), not a side effect of pattern matching.

    Slots the domain doesn't declare are ignored, so a model or extractor
    inventing a field can't grow the state.
    """
    known = set(slots_for(state.domain))
    for slot, value in values.items():
        if slot not in known or value in (None, ""):
            continue
        if state.slots.get(slot) in (None, ""):
            state.slots[slot] = value
    return state


def advance(state: ConversationState, user_message: str) -> ConversationState:
    """One turn's worth of state update: extract, merge, count the turn."""
    merged = merge_slots(state, extract_slots(user_message, state))
    merged.turn_count += 1
    return merged


def known_domains() -> list[str]:
    return sorted(DOMAIN_SLOTS)
