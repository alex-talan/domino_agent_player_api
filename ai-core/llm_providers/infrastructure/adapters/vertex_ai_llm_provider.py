"""Vertex AI chat generation via LangChain's unified Google GenAI integration.

``ChatGoogleGenerativeAI`` supports both the Gemini Developer API and the
Vertex AI Platform. Passing ``project``/``location`` with ``vertexai=True``
selects Vertex AI, authenticated through Application Default Credentials
(``gcloud auth application-default login``). Vertex AI has no API-key mode,
so this adapter always requires ADC.
"""
from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from llm_providers.infrastructure.adapters._retrying_llm_provider import (
    RetryingLlmProvider,
)


class VertexAiLlmProvider(RetryingLlmProvider):
    """Vertex AI adapter for chat generation, backed by Gemini models."""

    def __init__(
        self,
        project: str,
        location: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
    ) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            project=project,
            location=location,
            vertexai=True,
            temperature=temperature,
        )

    def get_llm(self) -> ChatGoogleGenerativeAI:
        return self._llm
