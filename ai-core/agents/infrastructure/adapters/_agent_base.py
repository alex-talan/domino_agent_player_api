from abc import abstractmethod
from typing import Generic, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from agents.domain.ports.agent import Agent
from llm_providers.domain.entities.chat_message import ChatMessage
from llm_providers.domain.ports.llm_provider import LlmProvider
from prompt_providers.domain.ports.prompt_provider import PromptProvider

ResultT = TypeVar("ResultT")


def to_domain_message(message: BaseMessage) -> ChatMessage:
    content = message.content if isinstance(message.content, str) else str(message.content)
    if isinstance(message, SystemMessage):
        return ChatMessage.from_system(content)
    if isinstance(message, HumanMessage):
        return ChatMessage.from_user(content)
    if isinstance(message, AIMessage):
        return ChatMessage.from_assistant(content)
    raise TypeError(f"Unsupported prompt message type: {type(message).__name__}")


def to_langchain_message(message: ChatMessage) -> BaseMessage:
    if message.role == "system":
        return SystemMessage(content=message.text)
    if message.role == "assistant":
        return AIMessage(content=message.text)
    return HumanMessage(content=message.text)


class AgentBase(Agent[ResultT], Generic[ResultT]):
    """Shared prompt and message preparation for LangChain agents."""

    def __init__(
        self,
        prompt_provider: PromptProvider[dict[str, object], ChatPromptTemplate],
        llm_provider: LlmProvider,
    ) -> None:
        self._prompt_provider = prompt_provider
        self._llm_provider = llm_provider

    @property
    def llm_provider(self) -> LlmProvider:
        return self._llm_provider

    @property
    @abstractmethod
    def output_type(self) -> type[BaseModel] | None:
        """Return the structured result type, if configured."""
        ...

    async def prepare_messages(
        self,
        query: str,
        history: list[ChatMessage] | None,
        variables: dict[str, object] | None,
    ) -> list[ChatMessage]:
        prompt = await self._prompt_provider.get_prompt(variables or {})
        messages = [to_domain_message(message) for message in prompt.format_messages()]
        messages.extend(history or [])
        messages.append(ChatMessage.from_user(query))
        return messages

    @abstractmethod
    def result_from_agent_state(self, state: dict[str, object]) -> ResultT:
        """Extract this agent's result from a LangChain agent state."""
        ...