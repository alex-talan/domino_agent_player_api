from typing import Generic, TypeVar, cast

from langchain.agents import create_agent

from agents.domain.ports.agent import Agent
from agents.infrastructure.adapters._agent_base import (
    AgentBase,
    to_langchain_message,
)
from agents.infrastructure.adapters._tool_bridge import to_langchain_tool
from llm_providers.domain.entities.chat_message import ChatMessage
from tools.domain.ports.tool import Tool

ResultT = TypeVar("ResultT")


class TooledAgent(Agent[ResultT], Generic[ResultT]):
    """Decorate an agent with a LangChain-managed tool loop."""

    def __init__(
        self,
        agent: AgentBase[ResultT],
        tools: list[Tool],
        *,
        max_steps: int = 25,
    ) -> None:
        if not tools:
            raise ValueError("tools must not be empty")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        tool_names = [tool.spec.name for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tool names must be unique")

        self._agent = agent
        self._max_steps = max_steps
        self._runtime = create_agent(
            model=agent.llm_provider.get_llm(),
            tools=[to_langchain_tool(tool) for tool in tools],
            response_format=agent.output_type,
        )

    async def invoke(
        self,
        query: str,
        *,
        history: list[ChatMessage] | None = None,
        variables: dict[str, object] | None = None,
    ) -> ResultT:
        messages = await self._agent.prepare_messages(query, history, variables)
        state = await self._runtime.ainvoke(
            {"messages": [to_langchain_message(message) for message in messages]},
            config={"recursion_limit": self._max_steps},
        )
        return self._agent.result_from_agent_state(cast(dict[str, object], state))