from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolSpec:
    """Describes the contract a :class:`~domain.ports.tool.Tool` exposes to an LLM.

    A tool advertises *what* it can do and *how* it is called; the actual work
    is performed by :meth:`~domain.ports.tool.Tool.invoke`.

    Parameters
    ----------
    name:
        Unique identifier the model uses to select the tool. Must match the
        function name registered with the LLM's tool-calling API.
    description:
        Natural-language explanation of what the tool does. The model relies on
        this text to decide *when* to call the tool, so it should be precise.
    parameters:
        JSON-Schema object describing the tool's inputs. Agent adapters pass
        this schema to the model for argument validation. Defaults to an empty
        object schema.
    """

    name: str
    description: str
    parameters: dict[str, object] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
