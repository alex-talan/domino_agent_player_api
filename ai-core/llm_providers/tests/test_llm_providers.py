from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from openai import RateLimitError

from llm_providers.domain.entities.chat_message import ChatMessage
from llm_providers.infrastructure.adapters._retrying_llm_provider import (
    MAX_RETRIES,
    RETRY_BASE_WAIT_SECONDS,
    RetryingLlmProvider,
)
from llm_providers.infrastructure.adapters.azure_openai_llm_provider import (
    AzureOpenAiLlmProvider,
)
from llm_providers.infrastructure.adapters.bedrock_llm_provider import (
    BedrockLlmProvider,
)
from llm_providers.infrastructure.adapters.google_llm_provider import GoogleLlmProvider
from llm_providers.infrastructure.adapters.ollama_llm_provider import OllamaLlmProvider
from llm_providers.infrastructure.adapters.open_router_llm_provider import (
    OpenRouterLlmProvider,
)
from llm_providers.infrastructure.adapters.openai_llm_provider import OpenAiLlmProvider
from llm_providers.infrastructure.adapters.vertex_ai_llm_provider import (
    VertexAiLlmProvider,
)

# -- helpers ------------------------------------------------------------------

def _make_rate_limit_error() -> RateLimitError:
    return RateLimitError("rate limited", response=MagicMock(), body=None)


def _make_llm(reply_text: str = "world") -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=reply_text))
    return llm


_MESSAGES = [ChatMessage.from_user("hello")]
_SLEEP_PATH = "llm_providers.infrastructure.adapters._retrying_llm_provider.asyncio.sleep"


class _FakeRetryingProvider(RetryingLlmProvider):
    """Minimal concrete subclass used to test RetryingLlmProvider in isolation."""

    def __init__(self, llm: MagicMock) -> None:
        self._llm = llm

    def get_llm(self) -> MagicMock:
        return self._llm


# -- TestChatMessage -----------------------------------------------------------

class TestChatMessage:
    def test_from_user_sets_role_and_text(self):
        message = ChatMessage.from_user("hi")
        assert message.role == "user"
        assert message.text == "hi"

    def test_rejects_unknown_role(self):
        with pytest.raises(ValueError):
            ChatMessage(role="system2", text="hi")  # type: ignore[arg-type]


# -- TestRetryingLlmProvider ----------------------------------------------------

class TestRetryingLlmProvider:
    @pytest.mark.asyncio
    async def test_returns_response_on_first_successful_call(self):
        llm = _make_llm()
        result = await _FakeRetryingProvider(llm).invoke(_MESSAGES)
        assert result == ChatMessage.from_assistant("world")

    @pytest.mark.asyncio
    async def test_forwards_converted_messages_to_ainvoke(self):
        llm = _make_llm()
        await _FakeRetryingProvider(llm).invoke(_MESSAGES)
        [lc_messages], _ = llm.ainvoke.call_args
        assert lc_messages[0].content == "hello"

    @pytest.mark.asyncio
    async def test_does_not_sleep_when_no_rate_limit_error(self):
        llm = _make_llm()
        sleep_mock = AsyncMock()
        with patch(_SLEEP_PATH, new=sleep_mock):
            await _FakeRetryingProvider(llm).invoke(_MESSAGES)
        sleep_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_and_eventually_succeeds(self):
        llm = _make_llm()
        llm.ainvoke = AsyncMock(
            side_effect=[_make_rate_limit_error(), AIMessage(content="world")]
        )
        with patch(_SLEEP_PATH, new=AsyncMock()):
            result = await _FakeRetryingProvider(llm).invoke(_MESSAGES)
        assert result == ChatMessage.from_assistant("world")

    @pytest.mark.asyncio
    async def test_raises_rate_limit_error_after_all_retries_exhausted(self):
        llm = _make_llm()
        llm.ainvoke = AsyncMock(side_effect=[_make_rate_limit_error()] * MAX_RETRIES)
        with patch(_SLEEP_PATH, new=AsyncMock()), pytest.raises(RateLimitError):
            await _FakeRetryingProvider(llm).invoke(_MESSAGES)

    @pytest.mark.asyncio
    async def test_sleep_duration_increases_linearly_between_retries(self):
        llm = _make_llm()
        llm.ainvoke = AsyncMock(
            side_effect=[_make_rate_limit_error(), _make_rate_limit_error(), AIMessage(content="world")]
        )
        sleep_mock = AsyncMock()
        with patch(_SLEEP_PATH, new=sleep_mock):
            await _FakeRetryingProvider(llm).invoke(_MESSAGES)
        sleep_calls = [call.args[0] for call in sleep_mock.call_args_list]
        assert sleep_calls == [RETRY_BASE_WAIT_SECONDS, RETRY_BASE_WAIT_SECONDS * 2]

    @pytest.mark.asyncio
    async def test_does_not_retry_non_rate_limit_errors(self):
        llm = _make_llm()
        llm.ainvoke = AsyncMock(side_effect=ValueError("boom"))
        with pytest.raises(ValueError):
            await _FakeRetryingProvider(llm).invoke(_MESSAGES)
        llm.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invoke_structured_uses_with_structured_output(self):
        class _Schema:
            pass

        llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value=_Schema())
        llm.with_structured_output = MagicMock(return_value=structured)

        result = await _FakeRetryingProvider(llm).invoke_structured(_MESSAGES, _Schema)

        llm.with_structured_output.assert_called_once_with(_Schema)
        assert isinstance(result, _Schema)


# -- TestOllamaLlmProvider ------------------------------------------------------

class TestOllamaLlmProvider:
    def test_get_llm_returns_chat_ollama_instance(self):
        with patch("llm_providers.infrastructure.adapters.ollama_llm_provider.ChatOllama") as MockLlm:
            provider = OllamaLlmProvider(model_name="gemma3", temperature=0.1)
            assert provider.get_llm() is MockLlm.return_value

    def test_constructor_passes_model_temperature_and_base_url(self):
        with patch("llm_providers.infrastructure.adapters.ollama_llm_provider.ChatOllama") as MockLlm:
            OllamaLlmProvider(model_name="llama3", temperature=0.5)
            MockLlm.assert_called_once_with(
                model="llama3", temperature=0.5, base_url="http://localhost:11434"
            )

    @pytest.mark.asyncio
    async def test_invoke_returns_chat_message(self):
        with patch("llm_providers.infrastructure.adapters.ollama_llm_provider.ChatOllama") as MockLlm:
            MockLlm.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="world"))
            result = await OllamaLlmProvider().invoke(_MESSAGES)
            assert result == ChatMessage.from_assistant("world")


# -- TestAzureOpenAiLlmProvider --------------------------------------------------

class TestAzureOpenAiLlmProvider:
    _INIT_KWARGS = dict(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="key-abc",
        azure_openai_api_version="2024-02-01",
        azure_openai_deployment="gpt-4o",
        azure_openai_temperature=0.0,
    )

    def test_get_llm_returns_azure_chat_openai_instance(self):
        with patch("llm_providers.infrastructure.adapters.azure_openai_llm_provider.AzureChatOpenAI") as MockLlm:
            provider = AzureOpenAiLlmProvider(**self._INIT_KWARGS)
            assert provider.get_llm() is MockLlm.return_value

    def test_constructor_passes_all_params(self):
        from pydantic import SecretStr

        with patch("llm_providers.infrastructure.adapters.azure_openai_llm_provider.AzureChatOpenAI") as MockLlm:
            AzureOpenAiLlmProvider(**self._INIT_KWARGS)
            MockLlm.assert_called_once_with(
                azure_endpoint="https://example.openai.azure.com",
                api_key=SecretStr("key-abc"),
                api_version="2024-02-01",
                azure_deployment="gpt-4o",
                temperature=0.0,
            )

    @pytest.mark.asyncio
    async def test_invoke_returns_chat_message(self):
        with patch("llm_providers.infrastructure.adapters.azure_openai_llm_provider.AzureChatOpenAI") as MockLlm:
            MockLlm.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="world"))
            result = await AzureOpenAiLlmProvider(**self._INIT_KWARGS).invoke(_MESSAGES)
            assert result == ChatMessage.from_assistant("world")


# -- TestOpenRouterLlmProvider ---------------------------------------------------

class TestOpenRouterLlmProvider:
    def test_get_llm_returns_chat_openai_instance(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
        with patch("llm_providers.infrastructure.adapters.open_router_llm_provider.ChatOpenAI") as MockLlm:
            provider = OpenRouterLlmProvider(model_name="openai/gpt-4o")
            assert provider.get_llm() is MockLlm.return_value

    def test_constructor_passes_model_base_url_and_temperature(self, monkeypatch):
        from pydantic import SecretStr

        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
        with patch("llm_providers.infrastructure.adapters.open_router_llm_provider.ChatOpenAI") as MockLlm:
            from llm_providers.infrastructure.adapters.open_router_llm_provider import (
                _OPENROUTER_BASE_URL,
            )

            OpenRouterLlmProvider(model_name="mistral/mistral-7b", temperature=0.2)
            MockLlm.assert_called_once_with(
                api_key=SecretStr("or-test"),
                model="mistral/mistral-7b",
                base_url=_OPENROUTER_BASE_URL,
                temperature=0.2,
            )

    def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterLlmProvider(model_name="openai/gpt-4o")

    @pytest.mark.asyncio
    async def test_invoke_returns_chat_message(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
        with patch("llm_providers.infrastructure.adapters.open_router_llm_provider.ChatOpenAI") as MockLlm:
            MockLlm.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="world"))
            result = await OpenRouterLlmProvider(model_name="openai/gpt-4o").invoke(_MESSAGES)
            assert result == ChatMessage.from_assistant("world")


# -- TestOpenAiLlmProvider --------------------------------------------------------

class TestOpenAiLlmProvider:
    def test_get_llm_returns_chat_openai_instance(self):
        with patch("llm_providers.infrastructure.adapters.openai_llm_provider.ChatOpenAI") as MockLlm:
            provider = OpenAiLlmProvider(model_name="gpt-4o", api_key="sk-test")
            assert provider.get_llm() is MockLlm.return_value

    def test_constructor_passes_model_api_key_and_temperature(self):
        from pydantic import SecretStr

        with patch("llm_providers.infrastructure.adapters.openai_llm_provider.ChatOpenAI") as MockLlm:
            OpenAiLlmProvider(model_name="gpt-4o-mini", api_key="sk-test", temperature=0.7)
            MockLlm.assert_called_once_with(
                api_key=SecretStr("sk-test"),
                model="gpt-4o-mini",
                temperature=0.7,
            )

    def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAiLlmProvider(model_name="gpt-4o")

    @pytest.mark.asyncio
    async def test_invoke_returns_chat_message(self):
        with patch("llm_providers.infrastructure.adapters.openai_llm_provider.ChatOpenAI") as MockLlm:
            MockLlm.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="world"))
            result = await OpenAiLlmProvider(model_name="gpt-4o", api_key="sk-test").invoke(_MESSAGES)
            assert result == ChatMessage.from_assistant("world")


# -- TestBedrockLlmProvider -------------------------------------------------------

class TestBedrockLlmProvider:
    _INIT_KWARGS = dict(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-east-1",
        temperature=0.0,
    )

    def test_get_llm_returns_chat_bedrock_instance(self):
        with patch("llm_providers.infrastructure.adapters.bedrock_llm_provider.ChatBedrockConverse") as MockLlm:
            provider = BedrockLlmProvider(**self._INIT_KWARGS)
            assert provider.get_llm() is MockLlm.return_value

    def test_constructor_passes_model_region_and_temperature(self):
        with patch("llm_providers.infrastructure.adapters.bedrock_llm_provider.ChatBedrockConverse") as MockLlm:
            BedrockLlmProvider(**self._INIT_KWARGS)
            MockLlm.assert_called_once_with(
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                region_name="us-east-1",
                temperature=0.0,
            )

    @pytest.mark.asyncio
    async def test_invoke_returns_chat_message(self):
        with patch("llm_providers.infrastructure.adapters.bedrock_llm_provider.ChatBedrockConverse") as MockLlm:
            MockLlm.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="world"))
            result = await BedrockLlmProvider(**self._INIT_KWARGS).invoke(_MESSAGES)
            assert result == ChatMessage.from_assistant("world")


# -- TestGoogleLlmProvider --------------------------------------------------------

class TestGoogleLlmProvider:
    def test_get_llm_returns_chat_google_generative_ai_instance(self):
        with patch("llm_providers.infrastructure.adapters.google_llm_provider.ChatGoogleGenerativeAI") as MockLlm:
            provider = GoogleLlmProvider(model_name="gemini-2.0-flash", api_key="test-key")
            assert provider.get_llm() is MockLlm.return_value

    def test_constructor_passes_model_api_key_and_temperature(self):
        from pydantic import SecretStr

        with patch("llm_providers.infrastructure.adapters.google_llm_provider.ChatGoogleGenerativeAI") as MockLlm:
            GoogleLlmProvider(model_name="gemini-2.0-flash", api_key="test-key", temperature=0.3)
            MockLlm.assert_called_once_with(
                api_key=SecretStr("test-key"),
                model="gemini-2.0-flash",
                temperature=0.3,
            )

    def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GoogleLlmProvider(model_name="gemini-2.0-flash")

    @pytest.mark.asyncio
    async def test_invoke_returns_chat_message(self):
        with patch("llm_providers.infrastructure.adapters.google_llm_provider.ChatGoogleGenerativeAI") as MockLlm:
            MockLlm.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="world"))
            result = await GoogleLlmProvider(model_name="gemini-2.0-flash", api_key="test-key").invoke(_MESSAGES)
            assert result == ChatMessage.from_assistant("world")


# -- TestVertexAiLlmProvider ------------------------------------------------------

class TestVertexAiLlmProvider:
    def test_get_llm_returns_chat_google_generative_ai_instance(self):
        with patch("llm_providers.infrastructure.adapters.vertex_ai_llm_provider.ChatGoogleGenerativeAI") as MockLlm:
            provider = VertexAiLlmProvider(project="prj", location="us-central1")
            assert provider.get_llm() is MockLlm.return_value

    def test_constructor_selects_vertex_backend(self):
        with patch("llm_providers.infrastructure.adapters.vertex_ai_llm_provider.ChatGoogleGenerativeAI") as MockLlm:
            VertexAiLlmProvider(project="prj", location="us-central1", model="gemini-2.5-flash")
            MockLlm.assert_called_once_with(
                model="gemini-2.5-flash",
                project="prj",
                location="us-central1",
                vertexai=True,
                temperature=0.0,
            )

    @pytest.mark.asyncio
    async def test_invoke_returns_chat_message(self):
        with patch("llm_providers.infrastructure.adapters.vertex_ai_llm_provider.ChatGoogleGenerativeAI") as MockLlm:
            MockLlm.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="world"))
            result = await VertexAiLlmProvider(project="prj", location="us-central1").invoke(_MESSAGES)
            assert result == ChatMessage.from_assistant("world")


