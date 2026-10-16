from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from llm_providers.infrastructure.adapters._retrying_llm_provider import (
    RetryingLlmProvider,
)


class GoogleLlmProvider(RetryingLlmProvider):
    """Google Gemini Developer API chat adapter."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        resolved_key = api_key or os.getenv("GOOGLE_API_KEY", "").strip()
        if not resolved_key:
            raise ValueError("api_key must be provided or GOOGLE_API_KEY must be set.")
        self._llm = ChatGoogleGenerativeAI(
            api_key=SecretStr(resolved_key),
            model=model_name,
            temperature=temperature,
        )

    def get_llm(self) -> ChatGoogleGenerativeAI:
        return self._llm
