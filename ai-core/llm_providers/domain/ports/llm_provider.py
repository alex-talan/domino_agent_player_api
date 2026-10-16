from abc import ABC, abstractmethod
from typing import Any, TypeVar

from llm_providers.domain.entities.chat_message import ChatMessage

T = TypeVar("T")


class LlmProvider(ABC):
    """Port for chat-completion providers."""

    @property
    def retry_count(self) -> int:
        """Return retries used by the most recent call in this execution context."""
        return 0

    @abstractmethod
    def get_llm(self) -> Any:
        """Return the underlying LangChain chat model."""
        ...

    @abstractmethod
    async def invoke(self, messages: list[ChatMessage]) -> ChatMessage:
        """Generate a reply for the given conversation."""
        ...

    @abstractmethod
    async def invoke_structured(self, messages: list[ChatMessage], output_type: type[T]) -> T:
        """Generate a reply parsed into *output_type* (a Pydantic model)."""
        ...
