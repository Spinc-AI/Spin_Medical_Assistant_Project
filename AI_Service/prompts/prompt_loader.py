"""Assembles the message list sent to the model, from a versioned prompt folder.

The order is fixed and is the whole point of versioning:

    system_v{n}  +  fewshot_v{n}  +  history  +  current user message

A prompt version is a folder (`v1/`, `v2/`, ...) holding `system.txt` and
`fewshot.json`. Adding `v3/` is enough to make it selectable -- nothing here
enumerates the known versions, so there is no list to forget to update.

The few-shot version can be pinned separately from the system version
(config.FEWSHOT_VERSION), because "same instructions, different examples" is a
legitimate A/B arm on its own.
"""
import json
from pathlib import Path
from typing import Any

from schemas.response import Message
from schemas.state import ConversationState, slots_for

PROMPTS_DIR = Path(__file__).resolve().parent

# Injected as a system turn between the few-shot examples and the live
# conversation, so the model can see what's already answered without us
# re-sending the whole transcript.
STATE_HEADER = "وضعیت مکالمه (به‌صورت JSON):"


class UnknownPromptVersion(ValueError):
    """Asked for a prompt version that has no folder.

    Raised rather than falling back to v1: a silent fallback would make an
    evaluation run report results for a version it never actually used.
    """


def available_versions() -> list[str]:
    """Every `v*/` folder that has a system.txt, sorted."""
    return sorted(p.name for p in PROMPTS_DIR.iterdir()
                  if p.is_dir() and (p / "system.txt").is_file())


def _version_dir(version: str) -> Path:
    path = PROMPTS_DIR / version
    if not (path / "system.txt").is_file():
        raise UnknownPromptVersion(
            f"unknown prompt version '{version}' -- available: {available_versions()}")
    return path


def load_system(version: str) -> str:
    return _version_dir(version).joinpath("system.txt").read_text(encoding="utf-8").strip()


def load_fewshot(version: str) -> list[dict[str, Any]]:
    """The `examples` list from a version's fewshot.json ([] if there's no file)."""
    path = _version_dir(version) / "fewshot.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("examples", [])


def _fewshot_messages(version: str) -> list[Message]:
    """Each example becomes a user turn + an assistant turn whose content is
    the JSON object serialised exactly as we want the model to answer."""
    messages: list[Message] = []
    for example in load_fewshot(version):
        messages.append(Message(role="user", content=example["user"]))
        assistant = example["assistant"]
        content = (assistant if isinstance(assistant, str)
                   else json.dumps(assistant, ensure_ascii=False))
        messages.append(Message(role="assistant", content=content))
    return messages


def _state_message(state: ConversationState, model_key: str | None) -> Message:
    """Tells the model what's already filled, what's still missing, and which
    model name to echo back in its `model` field."""
    payload = {
        "domain": state.domain,
        "filled": state.filled(),
        "missing": state.missing(),
        "all_slots": list(slots_for(state.domain)),
        "turn_count": state.turn_count,
    }
    if model_key:
        payload["model"] = model_key
    return Message(role="system",
                   content=f"{STATE_HEADER}\n{json.dumps(payload, ensure_ascii=False)}")


def load_messages(version: str, conversation_state: ConversationState,
                  history: list[Message] | None, user_message: str,
                  fewshot_version: str | None = None,
                  model_key: str | None = None) -> list[Message]:
    """Build the full message list for one interview turn.

    Example:
        load_messages("v1", ConversationState(domain="eye"), [], "چشمم درد می‌کند")
        -> [system, fewshot user, fewshot assistant, ..., state system, user]

    `history` is the prior turns of THIS conversation (already Message-shaped);
    pass None or [] on the first turn. `fewshot_version` defaults to `version`.
    Raises UnknownPromptVersion for a version with no folder -- including for
    the few-shot version, which is checked even when it differs.
    """
    messages: list[Message] = [Message(role="system", content=load_system(version))]
    messages += _fewshot_messages(fewshot_version or version)
    messages += list(history or [])
    messages.append(_state_message(conversation_state, model_key))
    messages.append(Message(role="user", content=user_message))
    return messages


def as_dicts(messages: list[Message]) -> list[dict]:
    """Wire format for Core_LLM's POST /chat."""
    return [m.model_dump() for m in messages]
