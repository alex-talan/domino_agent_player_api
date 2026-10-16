from __future__ import annotations

from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from llm_providers.infrastructure.adapters._retrying_llm_provider import (
    RetryingLlmProvider,
)


class AzureOpenAiLlmProvider(RetryingLlmProvider):
    """Azure OpenAI chat adapter."""

    def __init__(
        self,
        azure_openai_endpoint: str,
        azure_openai_api_key: str,
        azure_openai_api_version: str,
        azure_openai_deployment: str,
        azure_openai_temperature: float = 0.0,
    ) -> None:
        self._llm = AzureChatOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=SecretStr(azure_openai_api_key),
            api_version=azure_openai_api_version,
            azure_deployment=azure_openai_deployment,
            temperature=azure_openai_temperature,
        )

    def get_llm(self) -> AzureChatOpenAI:
        return self._llm
