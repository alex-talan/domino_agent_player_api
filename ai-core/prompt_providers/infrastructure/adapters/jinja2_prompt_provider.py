from collections.abc import Mapping

from langchain_core.prompts import ChatPromptTemplate

from prompt_providers.domain.ports.prompt_provider import PromptProvider


class Jinja2PromptProvider(
    PromptProvider[Mapping[str, object], ChatPromptTemplate]
):
    """Provide a system prompt backed by a trusted Jinja2 template."""

    def __init__(self, template: str) -> None:
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", template)],
            template_format="jinja2",
        )

    async def get_prompt(
        self, context: Mapping[str, object]
    ) -> ChatPromptTemplate:
        return self._prompt.partial(**context)
