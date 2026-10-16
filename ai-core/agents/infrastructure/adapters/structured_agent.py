from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar, cast

from pydantic import BaseModel

from agents.infrastructure.adapters._agent_base import AgentBase

if TYPE_CHECKING:
    from langchain_core.prompts import ChatPromptTemplate

    from llm_providers.domain.entities.chat_message import ChatMessage
    from llm_providers.domain.ports.llm_provider import LlmProvider
    from prompt_providers.domain.ports.prompt_provider import PromptProvider

OutputT = TypeVar("OutputT", bound=BaseModel)


class StructuredAgent(AgentBase[OutputT], Generic[OutputT]):
    """Return provider-validated Pydantic objects."""

    def __init__(
        self,
        prompt_provider: PromptProvider[dict[str, object], ChatPromptTemplate],
        llm_provider: LlmProvider,
        output_type: type[OutputT],
    ) -> None:
        super().__init__(prompt_provider, llm_provider)
        self._output_type = output_type

    @property
    def output_type(self) -> type[OutputT]:
        return self._output_type

    async def invoke(
        self,
        query: str,
        *,
        history: list[ChatMessage] | None = None,
        variables: dict[str, object] | None = None,
    ) -> OutputT:
        messages = await self.prepare_messages(query, history, variables)
        return await self.llm_provider.invoke_structured(messages, self._output_type)

    def result_from_agent_state(self, state: dict[str, object]) -> OutputT:
        result = state.get("structured_response")
        if not isinstance(result, self._output_type):
            raise TypeError("LangChain agent did not return the structured response")
        return cast(OutputT, result)
