"""Shared fixtures. Nothing here touches Core_LLM, a GPU, or the network."""
import json
import sys
from pathlib import Path

import pytest

# AI_Service/ on the path, so tests import the module the same way main.py does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.interface import LLMClient, LLMResponse  # noqa: E402


class FakeLLMClient(LLMClient):
    """Returns canned replies. Records every call for assertions.

    Pass `replies` (a list of raw strings) to script a conversation; the last
    one repeats once the list runs out. With no `replies`, it answers a
    well-formed routine JSON object every time.
    """

    def __init__(self, replies: list[str] | None = None, latency: float = 0.01):
        self.replies = list(replies or [])
        self.latency = latency
        self.calls: list[dict] = []

    def generate(self, messages, temperature=0.3, max_tokens=None, model=None):
        self.calls.append({"messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens, "model": model})
        if self.replies:
            text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        else:
            text = json.dumps({"domain": "eye", "model": model or "fake",
                               "question": "سؤال بعدی؟", "urgency": "routine"},
                              ensure_ascii=False)
        return LLMResponse(text=text, model=model or "fake", latency=self.latency)


def reply(question="سؤال بعدی؟", domain="eye", urgency="routine", model="gemma-4-12b") -> str:
    """A well-formed model reply, as a raw JSON string."""
    return json.dumps({"domain": domain, "model": model, "question": question,
                       "urgency": urgency}, ensure_ascii=False)


@pytest.fixture
def fake_client():
    return FakeLLMClient()


@pytest.fixture(autouse=True)
def _isolate_precheck_rules():
    """Drop any runtime-registered precheck rule between tests."""
    from safety import precheck
    yield
    precheck.clear_rules()
