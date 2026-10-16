from __future__ import annotations

from copy import deepcopy

from agents.domain.ports.agent import Agent
from tools.domain.entities.tool_spec import ToolSpec
from tools.domain.ports.tool import Tool

_DEFAULT_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The request to delegate to this sub-agent.",
        }
    },
    "required": ["query"],
}


class AgentTool(Tool):
    """Expose an agent as a tool for delegation."""

    def __init__(
        self,
        agent: Agent[object],
        *,
        name: str,
        description: str,
        parameters: dict[str, object] | None = None,
    ) -> None:
        self._agent = agent
        self._spec = ToolSpec(
            name=name,
            description=description,
            parameters=parameters or deepcopy(_DEFAULT_PARAMETERS),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, arguments: dict[str, object]) -> object:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("AgentTool requires a non-empty query")
        variables = arguments.get("variables")
        if variables is not None and not isinstance(variables, dict):
            raise TypeError("AgentTool variables must be a dictionary")
        return await self._agent.invoke(query, variables=variables)
