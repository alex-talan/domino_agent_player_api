from collections.abc import Mapping
from typing import Protocol

from langchain_core.prompts import ChatPromptTemplate
from langsmith.async_client import AsyncClient

from domain.ports.prompt_provider import PromptProvider

LANGSMITH_EU_API_URL = "https://eu.api.smith.langchain.com"


class _PromptClient(Protocol):
    async def pull_prompt(
        self,
        prompt_identifier: str,
        *,
        include_model: bool = False,
        secrets_from_env: bool = False,
        dangerously_pull_public_prompt: bool = False,
    ) -> object: ...


def build_langsmith_client(
    langsmith_api_key: str,
    api_url: str = LANGSMITH_EU_API_URL,
) -> AsyncClient:
    """Build an authenticated asynchronous LangSmith client."""
    if not langsmith_api_key.strip():
        raise ValueError("langsmith_api_key must not be empty")
    if not api_url.strip():
        raise ValueError("api_url must not be empty")
    return AsyncClient(api_key=langsmith_api_key, api_url=api_url)


class LangSmithPromptProvider(
    PromptProvider[Mapping[str, object], ChatPromptTemplate]
):
    """Pull a trusted chat prompt from LangSmith and apply its context."""

    def __init__(
        self,
        prompt_name: str,
        langsmith_api_key: str | None = None,
        *,
        api_url: str = LANGSMITH_EU_API_URL,
        client: _PromptClient | None = None,
    ) -> None:
        if not prompt_name.strip():
            raise ValueError("prompt_name must not be empty")
        if client is None and langsmith_api_key is None:
            raise ValueError("langsmith_api_key is required when client is not provided")

        if client is None:
            owned_client = build_langsmith_client(langsmith_api_key or "", api_url)
            self._client: _PromptClient = owned_client
            self._owned_client: AsyncClient | None = owned_client
        else:
            self._client = client
            self._owned_client = None
        self._prompt_name = prompt_name.strip()

    async def get_prompt(
        self, context: Mapping[str, object]
    ) -> ChatPromptTemplate:
        prompt = await self._client.pull_prompt(
            self._prompt_name,
            include_model=False,
            secrets_from_env=False,
            dangerously_pull_public_prompt=False,
        )
        if not isinstance(prompt, ChatPromptTemplate):
            raise TypeError(
                "LangSmith prompt must deserialize to ChatPromptTemplate, "
                f"got {type(prompt).__name__}"
            )
        return prompt.partial(**context)

    async def close(self) -> None:
        """Close the internally created client, if any."""
        if self._owned_client is not None:
            await self._owned_client.aclose()
            self._owned_client = None