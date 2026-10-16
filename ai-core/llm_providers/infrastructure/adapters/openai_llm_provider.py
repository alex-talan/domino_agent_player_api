from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from llm_providers.infrastructure.adapters._retrying_llm_provider import (
    RetryingLlmProvider,
)


class OpenAiLlmProvider(RetryingLlmProvider):
    """OpenAI chat adapter."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        if not resolved_key:
            raise ValueError("api_key must be provided or OPENAI_API_KEY must be set.")
        self._llm = ChatOpenAI(
            api_key=SecretStr(resolved_key),
            model=model_name,
            temperature=temperature,
        )

    def get_llm(self) -> ChatOpenAI:
        return self._llm
