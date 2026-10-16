from dataclasses import dataclass

import pytest
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.ports.prompt_provider import PromptProvider
from infrastructure.adapters.jinja2_prompt_provider import Jinja2PromptProvider
from infrastructure.adapters.langsmith_prompt_provider import (
    LANGSMITH_EU_API_URL,
    LangSmithPromptProvider,
    build_langsmith_client,
)
from infrastructure.adapters.markdown_file_prompt_provider import (
    _DEFAULT_PROMPTS_DIR,
    MarkdownFilePromptProvider,
)
from infrastructure.adapters.static_prompt_provider import StaticPromptProvider


def render_system(prompt: ChatPromptTemplate) -> str:
    message = prompt.invoke({}).messages[0]
    assert isinstance(message, SystemMessage)
    return str(message.content)


@dataclass(frozen=True)
class RequestContext:
    language: str


class ContextPromptProvider(PromptProvider[RequestContext, ChatPromptTemplate]):
    async def get_prompt(self, context: RequestContext) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [("system", f"Reply in {context.language}.")]
        )


class FakeLangSmithClient:
    def __init__(self, prompt: object) -> None:
        self.prompt = prompt
        self.calls: list[tuple[str, dict[str, bool]]] = []

    async def pull_prompt(
        self,
        prompt_identifier: str,
        *,
        include_model: bool = False,
        secrets_from_env: bool = False,
        dangerously_pull_public_prompt: bool = False,
    ) -> object:
        self.calls.append(
            (
                prompt_identifier,
                {
                    "include_model": include_model,
                    "secrets_from_env": secrets_from_env,
                    "dangerously_pull_public_prompt": dangerously_pull_public_prompt,
                },
            )
        )
        return self.prompt


@pytest.mark.asyncio
async def test_provider_supports_client_defined_context() -> None:
    provider = ContextPromptProvider()

    prompt = await provider.get_prompt(RequestContext(language="French"))

    assert render_system(prompt) == "Reply in French."


@pytest.mark.asyncio
async def test_static_provider_returns_reusable_prompt() -> None:
    provider = StaticPromptProvider("Be concise.")

    first = await provider.get_prompt(None)
    second = await provider.get_prompt({"ignored": True})

    assert first is second
    assert render_system(first) == "Be concise."


@pytest.mark.asyncio
async def test_jinja2_provider_applies_context() -> None:
    provider = Jinja2PromptProvider(
        "{% if formal %}Good day{% else %}Hello{% endif %}, {{ name }}."
    )

    prompt = await provider.get_prompt({"formal": True, "name": "Ada"})

    assert render_system(prompt) == "Good day, Ada."


@pytest.mark.asyncio
async def test_jinja2_provider_leaves_missing_variables_required() -> None:
    provider = Jinja2PromptProvider("Hello, {{ name }}.")

    prompt = await provider.get_prompt({})

    with pytest.raises(KeyError, match="name"):
        prompt.invoke({})


def test_langsmith_client_factory_uses_eu_endpoint(monkeypatch) -> None:
    captured: dict[str, str] = {}
    client = object()

    def create_client(**kwargs: str) -> object:
        captured.update(kwargs)
        return client

    monkeypatch.setattr(
        "infrastructure.adapters.langsmith_prompt_provider.AsyncClient",
        create_client,
    )

    assert build_langsmith_client("test-key") is client
    assert captured == {
        "api_key": "test-key",
        "api_url": LANGSMITH_EU_API_URL,
    }


def test_langsmith_client_factory_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_langsmith_client("  ")


@pytest.mark.asyncio
async def test_langsmith_provider_pulls_secure_chat_prompt() -> None:
    stored_prompt = ChatPromptTemplate.from_messages(
        [("system", "Reply in {language}.")]
    )
    client = FakeLangSmithClient(stored_prompt)
    provider = LangSmithPromptProvider("support:production", client=client)

    prompt = await provider.get_prompt({"language": "French"})

    assert render_system(prompt) == "Reply in French."
    assert client.calls == [
        (
            "support:production",
            {
                "include_model": False,
                "secrets_from_env": False,
                "dangerously_pull_public_prompt": False,
            },
        )
    ]


@pytest.mark.asyncio
async def test_langsmith_provider_rejects_non_chat_prompt() -> None:
    provider = LangSmithPromptProvider(
        "invalid", client=FakeLangSmithClient(object())
    )

    with pytest.raises(TypeError, match="ChatPromptTemplate"):
        await provider.get_prompt({})


def test_langsmith_provider_requires_credentials_or_client() -> None:
    with pytest.raises(ValueError, match="api_key is required"):
        LangSmithPromptProvider("support")


@pytest.mark.asyncio
async def test_markdown_provider_loads_and_applies_context(tmp_path) -> None:
    (tmp_path / "greeting.md").write_text(
        "Hello, {{ name }}!\n", encoding="utf-8"
    )
    provider = MarkdownFilePromptProvider("greeting", prompts_dir=tmp_path)

    prompt = await provider.get_prompt({"name": "Ada"})

    assert render_system(prompt) == "Hello, Ada!"


def test_markdown_provider_rejects_path_traversal(tmp_path) -> None:
    with pytest.raises(ValueError, match="inside prompts_dir"):
        MarkdownFilePromptProvider("../outside", prompts_dir=tmp_path)


def test_markdown_provider_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        MarkdownFilePromptProvider("missing", prompts_dir=tmp_path)


@pytest.mark.asyncio
async def test_bundled_prompt_is_available() -> None:
    assert _DEFAULT_PROMPTS_DIR.is_dir()
    provider = MarkdownFilePromptProvider("rag_assistant")

    prompt = await provider.get_prompt(
        {"company": "Test Co", "language": "English", "tone": None}
    )

    content = render_system(prompt)
    assert "Test Co" in content
    assert "English" in content