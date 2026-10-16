import dataclasses

import pytest

from tools.domain.entities.tool_spec import ToolSpec
from tools.domain.ports.tool import Tool


class _EchoTool(Tool):
    """Minimal in-memory Tool used to exercise the port contract."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description="Echo the provided message back to the caller.",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        )

    async def invoke(self, arguments: dict[str, object]) -> object:
        return arguments["message"]


class TestToolSpec:
    def test_defaults_to_empty_object_schema(self) -> None:
        spec = ToolSpec(name="noop", description="does nothing")
        assert spec.parameters == {"type": "object", "properties": {}}

    def test_is_frozen(self) -> None:
        spec = ToolSpec(name="noop", description="does nothing")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "other"  # type: ignore[misc]

    def test_default_parameters_are_independent_per_instance(self) -> None:
        a = ToolSpec(name="a", description="a")
        b = ToolSpec(name="b", description="b")
        assert a.parameters is not b.parameters


class TestToolPort:
    def test_cannot_instantiate_abstract_port(self) -> None:
        with pytest.raises(TypeError):
            Tool()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_concrete_tool_exposes_spec_and_invokes(self) -> None:
        tool = _EchoTool()

        assert tool.spec.name == "echo"
        assert tool.spec.parameters["required"] == ["message"]

        result = await tool.invoke({"message": "hello"})
        assert result == "hello"
