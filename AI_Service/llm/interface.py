"""The contract every AI_Service LLM client implements.

This is intentionally a MIRROR of Core_LLM/deployment/llm/interface.py, not a
shared import. CONTRIBUTING.md is explicit that modules never reach into each
other's Python -- they talk over HTTP -- so AI_Service and Core_LLM stay
independently deployable, each carrying its own copy of the shape. The cost is
a duplicated 30 lines; the benefit is that either module can be deployed, or
have its interface evolved, without the other.
"""
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """One completion plus the metadata evaluation/metrics.py needs.

    Example:
        LLMResponse(text='{"domain": "eye", ...}', model="gemma-4-12b",
                    latency=1.94, usage=None)
    """

    text: str
    model: str
    latency: float = Field(description="Wall-clock seconds for the call.")
    usage: dict | None = Field(
        default=None,
        description="Token accounting when the backend reports it; None when it doesn't.",
    )


class LLMClient(ABC):
    """Turns chat messages into an `LLMResponse`.

    `messages` is the standard OpenAI shape:
    `[{"role": "system"|"user"|"assistant", "content": str}, ...]`.
    """

    @abstractmethod
    def generate(self, messages: list[dict], temperature: float = 0.3,
                 max_tokens: int | None = None, model: str | None = None) -> LLMResponse:
        """`model` picks a specific backend model key for this one call (the
        router chooses a different one per turn); None means the client's own
        default."""


# Core_LLM names the same ABC `BaseLLM`. Both names work here so code moved
# between the two modules doesn't break on the import line alone.
BaseLLM = LLMClient
