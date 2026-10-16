from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from llm_providers.domain.entities.chat_message import ChatMessage

ResultT_co = TypeVar("ResultT_co", covariant=True)


class Agent(ABC, Generic[ResultT_co]):
    """Run one conversational turn and return a result."""

    @abstractmethod
    async def invoke(
        self,
        query: str,
        *,
        history: list[ChatMessage] | None = None,
        variables: dict[str, object] | None = None,
    ) -> ResultT_co:
        """Run a query with optional history and prompt variables."""
        ...
