from collections.abc import Mapping
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from domain.ports.prompt_provider import PromptProvider

_DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts"


class MarkdownFilePromptProvider(
    PromptProvider[Mapping[str, object], ChatPromptTemplate]
):
    """Provide a trusted Jinja2 system prompt loaded from Markdown."""

    def __init__(
        self,
        template_name: str,
        prompts_dir: Path | str | None = None,
    ) -> None:
        directory = (
            Path(prompts_dir) if prompts_dir is not None else _DEFAULT_PROMPTS_DIR
        ).resolve()
        path = (directory / f"{template_name}.md").resolve()

        if not path.is_relative_to(directory):
            raise ValueError("Prompt template must be inside prompts_dir")

        if not path.is_file():
            raise FileNotFoundError(
                f"Prompt template '{template_name}' not found at: {path}"
            )

        template_text = path.read_text(encoding="utf-8")
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", template_text)],
            template_format="jinja2",
        )

    async def get_prompt(
        self, context: Mapping[str, object]
    ) -> ChatPromptTemplate:
        return self._prompt.partial(**context)
