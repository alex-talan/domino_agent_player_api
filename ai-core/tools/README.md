# tools

Domain contracts for **tools** that ai-core agents can call.

This module is intentionally *ports + entities only* — it has **no runtime
dependencies** and **no adapters**. It defines the contract every tool must
satisfy so that the `agents` module (and any future tool implementation) can
depend on a stable, framework-free interface.

## Public surface

| Symbol     | Kind          | Purpose                                                          |
| ---------- | ------------- | ---------------------------------------------------------------- |
| `ToolSpec` | entity        | Frozen dataclass: `name`, `description`, `parameters` (JSON Schema). Describes the tool contract advertised to an LLM. |
| `Tool`     | port (ABC)    | `spec` property + `async invoke(arguments: dict) -> object`.     |

## Writing a tool

Implement the `Tool` port in your own module (adapters do **not** live here):

```python
from tools.domain.entities.tool_spec import ToolSpec
from tools.domain.ports.tool import Tool


class WeatherTool(Tool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_weather",
            description="Return the current weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

    async def invoke(self, arguments: dict[str, object]) -> object:
        city = arguments["city"]
        ...  # call a weather API
        return {"city": city, "temperature_c": 21}
```

The `agents` module bridges any `Tool` into the underlying LLM framework, so a
tool never needs to know which agent or provider will call it.
