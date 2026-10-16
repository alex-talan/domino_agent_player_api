from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from agents.domain.ports.agent import Agent
from agents.infrastructure.adapters._tool_bridge import to_langchain_tool
from agents.infrastructure.adapters.agent_tool import AgentTool
from agents.infrastructure.adapters.basic_agent import BasicAgent
from agents.infrastructure.adapters.structured_agent import StructuredAgent
from agents.infrastructure.adapters.tooled_agent import TooledAgent
from llm_providers.domain.entities.chat_message import ChatMessage
from tools.domain.entities.tool_spec import ToolSpec
from tools.domain.ports.tool import Tool


class Answer(BaseModel):
    value: int = Field(description="The computed value.")


class RecordingPromptProvider:
    def __init__(self, template: str = "Assistant for {company}.") -> None:
        self._template = template
        self.context: dict[str, object] | None = None

    async def get_prompt(
        self, context: dict[str, object]
    ) -> ChatPromptTemplate:
        self.context = context
        return ChatPromptTemplate.from_messages(
            [("system", self._template)]
        ).partial(**context)


class FakeLlmProvider:
    def __init__(
        self,
        *,
        text: str = "done",
        structured: BaseModel | None = None,
    ) -> None:
        self.text = text
        self.structured = structured
        self.messages: list[ChatMessage] = []
        self.output_type: type[BaseModel] | None = None
        self.model = object()

    def get_llm(self) -> object:
        return self.model

    async def invoke(self, messages: list[ChatMessage]) -> ChatMessage:
        self.messages = messages
        return ChatMessage.from_assistant(self.text)

    async def invoke_structured(
        self,
        messages: list[ChatMessage],
        output_type: type[BaseModel],
    ) -> BaseModel:
        self.messages = messages
        self.output_type = output_type
        if self.structured is None:
            raise RuntimeError("No structured response configured")
        return self.structured


class ScriptedToolCallingModel(BaseChatModel):
    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=self.responses.pop(0))]
        )

    def bind_tools(self, tools, *, tool_choice=None, **kwargs) -> Runnable:
        return self


class AdderTool(Tool):
    def __init__(self) -> None:
        self.arguments: dict[str, object] | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="adder",
            description="Add two integers.",
            parameters={
                "type": "object",
                "properties": {
                    "left": {"type": "integer"},
                    "right": {"type": "integer"},
                },
                "required": ["left", "right"],
            },
        )

    async def invoke(self, arguments: dict[str, object]) -> object:
        self.arguments = arguments
        return int(arguments["left"]) + int(arguments["right"])


class EchoAgent(Agent[str]):
    async def invoke(
        self,
        query: str,
        *,
        history: list[ChatMessage] | None = None,
        variables: dict[str, object] | None = None,
    ) -> str:
        return query


@pytest.mark.asyncio
async def test_basic_agent_prepares_prompt_history_and_query() -> None:
    prompt = RecordingPromptProvider()
    llm = FakeLlmProvider(text="Hello")
    agent = BasicAgent(prompt, llm)

    result = await agent.invoke(
        "latest",
        history=[ChatMessage.from_user("earlier")],
        variables={"company": "Talan"},
    )

    assert result == "Hello"
    assert prompt.context == {"company": "Talan"}
    assert llm.messages == [
        ChatMessage.from_system("Assistant for Talan."),
        ChatMessage.from_user("earlier"),
        ChatMessage.from_user("latest"),
    ]


@pytest.mark.asyncio
async def test_structured_agent_uses_native_typed_output() -> None:
    expected = Answer(value=5)
    llm = FakeLlmProvider(structured=expected)
    agent = StructuredAgent(RecordingPromptProvider(), llm, Answer)

    result = await agent.invoke("add", variables={"company": "Talan"})

    assert result is expected
    assert llm.output_type is Answer


@pytest.mark.asyncio
async def test_langchain_tool_bridge_preserves_async_contract() -> None:
    tool = AdderTool()

    result = await to_langchain_tool(tool).ainvoke({"left": 2, "right": 3})

    assert result == 5
    assert tool.arguments == {"left": 2, "right": 3}


@pytest.mark.asyncio
async def test_tooled_basic_agent_uses_langchain_runtime(monkeypatch) -> None:
    runtime = MagicMock()
    runtime.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="tool-assisted answer")]}
    )
    create_agent = MagicMock(return_value=runtime)
    monkeypatch.setattr(
        "agents.infrastructure.adapters.tooled_agent.create_agent", create_agent
    )
    llm = FakeLlmProvider()
    base = BasicAgent(RecordingPromptProvider(), llm)
    agent = TooledAgent(base, [AdderTool()], max_steps=7)

    result = await agent.invoke("calculate", variables={"company": "Talan"})

    assert result == "tool-assisted answer"
    create_agent.assert_called_once()
    assert create_agent.call_args.kwargs["model"] is llm.model
    assert create_agent.call_args.kwargs["response_format"] is None
    [langchain_tool] = create_agent.call_args.kwargs["tools"]
    assert langchain_tool.name == "adder"
    state = runtime.ainvoke.await_args.args[0]
    assert isinstance(state["messages"][0], SystemMessage)
    assert isinstance(state["messages"][-1], HumanMessage)
    assert runtime.ainvoke.await_args.kwargs["config"] == {"recursion_limit": 7}


@pytest.mark.asyncio
async def test_tooled_agent_runs_real_langchain_tool_loop() -> None:
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "adder",
                        "args": {"left": 2, "right": 3},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The answer is 5."),
        ]
    )
    llm = FakeLlmProvider()
    llm.model = model
    tool = AdderTool()
    agent = TooledAgent(
        BasicAgent(RecordingPromptProvider(), llm),
        [tool],
    )

    result = await agent.invoke("calculate", variables={"company": "Talan"})

    assert result == "The answer is 5."
    assert tool.arguments == {"left": 2, "right": 3}


@pytest.mark.asyncio
async def test_tooled_structured_agent_returns_model(monkeypatch) -> None:
    expected = Answer(value=5)
    runtime = MagicMock()
    runtime.ainvoke = AsyncMock(return_value={"structured_response": expected})
    create_agent = MagicMock(return_value=runtime)
    monkeypatch.setattr(
        "agents.infrastructure.adapters.tooled_agent.create_agent", create_agent
    )
    base = StructuredAgent(
        RecordingPromptProvider(), FakeLlmProvider(), Answer
    )

    result = await TooledAgent(base, [AdderTool()]).invoke(
        "calculate", variables={"company": "Talan"}
    )

    assert result is expected
    assert create_agent.call_args.kwargs["response_format"] is Answer


@pytest.mark.asyncio
async def test_tooled_structured_agent_runs_real_langchain_loop() -> None:
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "adder",
                        "args": {"left": 2, "right": 3},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Answer",
                        "args": {"value": 5},
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    llm = FakeLlmProvider()
    llm.model = model
    tool = AdderTool()
    base = StructuredAgent(RecordingPromptProvider(), llm, Answer)

    result = await TooledAgent(base, [tool]).invoke(
        "calculate", variables={"company": "Talan"}
    )

    assert result == Answer(value=5)
    assert tool.arguments == {"left": 2, "right": 3}


def test_tooled_agent_validates_configuration() -> None:
    base = BasicAgent(RecordingPromptProvider(), FakeLlmProvider())

    with pytest.raises(ValueError, match="must not be empty"):
        TooledAgent(base, [])
    with pytest.raises(ValueError, match="at least 1"):
        TooledAgent(base, [AdderTool()], max_steps=0)
    with pytest.raises(ValueError, match="must be unique"):
        TooledAgent(base, [AdderTool(), AdderTool()])


@pytest.mark.asyncio
async def test_agent_tool_delegates_query() -> None:
    tool = AgentTool(EchoAgent(), name="echo", description="Echo a query.")

    result = await tool.invoke({"query": "hello"})

    assert result == "hello"
    assert tool.spec.parameters["required"] == ["query"]


@pytest.mark.asyncio
async def test_agent_tool_validates_arguments() -> None:
    tool = AgentTool(EchoAgent(), name="echo", description="Echo a query.")

    with pytest.raises(ValueError, match="non-empty query"):
        await tool.invoke({})
    with pytest.raises(TypeError, match="dictionary"):
        await tool.invoke({"query": "hello", "variables": "invalid"})


def test_agent_port_is_abstract_and_adapters_implement_it() -> None:
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]

    assert issubclass(BasicAgent, Agent)
    assert issubclass(StructuredAgent, Agent)
    assert issubclass(TooledAgent, Agent)