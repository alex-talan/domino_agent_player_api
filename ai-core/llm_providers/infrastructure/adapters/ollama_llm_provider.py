from __future__ import annotations

from langchain_ollama import ChatOllama

from llm_providers.infrastructure.adapters._retrying_llm_provider import (
    RetryingLlmProvider,
)


class OllamaLlmProvider(RetryingLlmProvider):
    """Local Ollama chat adapter."""

    def __init__(
        self,
        model_name: str = "gemma3",
        temperature: float = 0.1,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._llm = ChatOllama(model=model_name, temperature=temperature, base_url=base_url)

    def get_llm(self) -> ChatOllama:
        return self._llm
