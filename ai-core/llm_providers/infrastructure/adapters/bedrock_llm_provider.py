from __future__ import annotations

from langchain_aws import ChatBedrockConverse

from llm_providers.infrastructure.adapters._retrying_llm_provider import (
    RetryingLlmProvider,
)


class BedrockLlmProvider(RetryingLlmProvider):
    """Amazon Bedrock chat adapter."""

    def __init__(
        self,
        model_id: str,
        region_name: str,
        temperature: float = 0.0,
    ) -> None:
        self._llm = ChatBedrockConverse(
            model=model_id,
            region_name=region_name,
            temperature=temperature,
        )

    def get_llm(self) -> ChatBedrockConverse:
        return self._llm
