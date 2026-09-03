"""AI_Service's LLM clients.

`core_llm_client.CoreLLMClient` is the real one -- HTTP to Core_LLM, the only
way this module reaches a model. `vllm_client.VLLMClient` is unused
scaffolding for a possible future direct-vLLM deployment.
"""
from .core_llm_client import CoreLLMClient, CoreLLMError, CoreLLMUnavailable
from .interface import BaseLLM, LLMClient, LLMResponse

__all__ = ["LLMClient", "BaseLLM", "LLMResponse", "CoreLLMClient",
           "CoreLLMError", "CoreLLMUnavailable"]
