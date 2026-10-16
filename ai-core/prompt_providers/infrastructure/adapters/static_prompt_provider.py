from langchain_core.prompts import ChatPromptTemplate

from domain.ports.prompt_provider import PromptProvider


class StaticPromptProvider(PromptProvider[object, ChatPromptTemplate]):
    """Provide a fixed system prompt."""

    def __init__(self, content: str) -> None:
        self._prompt = ChatPromptTemplate.from_messages([("system", content)])

    async def get_prompt(self, context: object) -> ChatPromptTemplate:
        return self._prompt
