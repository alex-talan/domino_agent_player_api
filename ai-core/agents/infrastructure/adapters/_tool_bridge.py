from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

if TYPE_CHECKING:
    from tools.domain.ports.tool import Tool


def to_langchain_tool(tool: Tool) -> StructuredTool:
    """Adapt an async domain tool to LangChain."""
    spec = tool.spec

    async def invoke_tool(**kwargs: object) -> object:
        return await tool.invoke(kwargs)

    return StructuredTool.from_function(
        name=spec.name,
        description=spec.description,
        args_schema=spec.parameters,
        coroutine=invoke_tool,
    )
