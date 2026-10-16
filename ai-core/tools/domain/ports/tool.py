from abc import ABC, abstractmethod

from tools.domain.entities.tool_spec import ToolSpec


class Tool(ABC):
    """A capability an agent can invoke during its reasoning loop.

    A tool advertises its contract through :attr:`spec` (name, description and
    a JSON-Schema for its arguments) and performs work through :meth:`invoke`.

    Concrete implementations live *outside* this package — an adapter may wrap
    an HTTP call, a database query, a plain Python function, or even another
    agent (see ``agents.AgentTool``, which turns an ``Agent`` into a ``Tool`` so
    agents can be composed).
    """

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool's contract: name, description, and argument schema."""
        ...

    @abstractmethod
    async def invoke(self, arguments: dict[str, object]) -> object:
        """Execute the tool with LLM-supplied arguments.

        Parameters
        ----------
        arguments:
            Keyword arguments produced by the model, matching the shape of
            :attr:`ToolSpec.parameters`. The calling adapter is responsible for
            validating them against the schema before invocation.

        Returns
        -------
        object
            The tool result. The calling adapter serialises this into a
            tool-result message before feeding it back to the model, so it
            should be JSON-serialisable (``str``, ``dict``, ``list``, numbers…).
        """
        ...
