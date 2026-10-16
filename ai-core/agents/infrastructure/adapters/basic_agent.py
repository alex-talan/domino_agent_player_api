from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_core.messages import AIMessage

from agents.infrastructure.adapters._agent_base import AgentBase

if TYPE_CHECKING:
    from llm_providers.domain.entities.chat_message import ChatMessage


class BasicAgent(AgentBase[str]):
    """Return the model's final answer as text."""

    @property
    def output_type(self) -> None:
        return None

    async def invoke(
        self,
        query: str,
        *,
        history: list[ChatMessage] | None = None,
        variables: dict[str, object] | None = None,
    ) -> str:
        messages = await self.prepare_messages(query, history, variables)
        message = await self.llm_provider.invoke(messages)
        return message.text or ""

    def result_from_agent_state(self, state: dict[str, object]) -> str:
        messages = cast(list[object], state.get("messages", []))
        if not messages or not isinstance(messages[-1], AIMessage):
            raise RuntimeError("LangChain agent did not return an AIMessage")
        content = messages[-1].content
        return content if isinstance(content, str) else str(content)
