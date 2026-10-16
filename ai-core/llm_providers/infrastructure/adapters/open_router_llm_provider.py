from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from llm_providers.infrastructure.adapters._retrying_llm_provider import (
    RetryingLlmProvider,
)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLlmProvider(RetryingLlmProvider):
    """OpenRouter chat adapter (OpenAI-compatible API)."""

    def __init__(self, model_name: str, temperature: float = 0.0) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY must be set.")
        self._llm = ChatOpenAI(
            api_key=SecretStr(api_key),
            model=model_name,
            base_url=_OPENROUTER_BASE_URL,
            temperature=temperature,
        )

    def get_llm(self) -> ChatOpenAI:
        return self._llm
