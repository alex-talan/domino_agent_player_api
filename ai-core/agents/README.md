# Agents

Composable LangChain agents with async prompt, model, and tool interfaces.

## Agent Types

- `BasicAgent` returns text.
- `StructuredAgent[T]` returns a validated Pydantic model.
- `TooledAgent[T]` adds a LangChain tool loop to either agent type.
- `AgentTool` exposes an agent as a tool for delegation.

Every agent accepts a query, prior `ChatMessage` history, and prompt variables.

```python
answer = await BasicAgent(prompt_provider, llm_provider).invoke(
    "What changed?",
    history=history,
    variables={"language": "English"},
)
```

## Structured Output

Field descriptions become part of the provider-native structured-output schema.

```python
from pydantic import BaseModel, Field


class Classification(BaseModel):
    category: str = Field(description="The request category.")
    requires_search: bool = Field(description="Whether retrieval is required.")


agent = StructuredAgent(prompt_provider, llm_provider, Classification)
result = await agent.invoke("Classify this request")
```

## Tools

Implement the `tools.Tool` port, then decorate a basic or structured agent.
LangChain validates arguments and manages the model/tool loop.

```python
agent = TooledAgent(
    BasicAgent(prompt_provider, llm_provider),
    tools=[search_tool, calculator_tool],
)
```

`AgentTool` can wrap a specialist agent so another `TooledAgent` can delegate
to it.
