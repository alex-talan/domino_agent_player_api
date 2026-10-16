from importlib import import_module

from llm_providers.domain.entities.chat_message import ChatMessage
from llm_providers.domain.ports.llm_provider import LlmProvider
from llm_providers.infrastructure.adapters._retrying_llm_provider import (
    RetryingLlmProvider,
)

# Adapters load on first access: each pulls its own provider integration
# (bedrock, google, ollama, …), which need not be installed to use another one.
_ADAPTER_MODULES = {
    "AzureOpenAiLlmProvider": "llm_providers.infrastructure.adapters.azure_openai_llm_provider",
    "BedrockLlmProvider": "llm_providers.infrastructure.adapters.bedrock_llm_provider",
    "GoogleLlmProvider": "llm_providers.infrastructure.adapters.google_llm_provider",
    "OllamaLlmProvider": "llm_providers.infrastructure.adapters.ollama_llm_provider",
    "OpenRouterLlmProvider": "llm_providers.infrastructure.adapters.open_router_llm_provider",
    "OpenAiLlmProvider": "llm_providers.infrastructure.adapters.openai_llm_provider",
    "VertexAiLlmProvider": "llm_providers.infrastructure.adapters.vertex_ai_llm_provider",
}

__all__ = [
    "AzureOpenAiLlmProvider",
    "BedrockLlmProvider",
    "ChatMessage",
    "GoogleLlmProvider",
    "LlmProvider",
    "OllamaLlmProvider",
    "OpenAiLlmProvider",
    "OpenRouterLlmProvider",
    "RetryingLlmProvider",
    "VertexAiLlmProvider",
]


def __getattr__(name: str):
    module_path = _ADAPTER_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
