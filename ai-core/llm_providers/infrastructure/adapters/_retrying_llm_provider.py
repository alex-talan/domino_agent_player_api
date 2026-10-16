from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import RateLimitError

from llm_providers.domain.entities.chat_message import ChatMessage
from llm_providers.domain.ports.llm_provider import LlmProvider

MAX_RETRIES = 3
RETRY_BASE_WAIT_SECONDS = 5
_logger = logging.getLogger(__name__)

# Providers surface rate limits through different exception types; matching on
# these markers lets one retry policy cover OpenAI, Bedrock, and Google alike.
_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "ThrottlingException", "Too Many Requests")

T = TypeVar("T")


def _to_langchain_message(message: ChatMessage) -> BaseMessage:
    if message.role == "system":
        return SystemMessage(content=message.text)
    if message.role == "assistant":
        return AIMessage(content=message.text)
    return HumanMessage(content=message.text)


def _from_langchain_message(message: AIMessage) -> ChatMessage:
    content = message.content if isinstance(message.content, str) else str(message.content)
    return ChatMessage.from_assistant(content)


def _is_rate_limited(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    text = str(exc)
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


class RetryingLlmProvider(LlmProvider, ABC):
    """LlmProvider that retries on rate limits with linear backoff.

    Concrete subclasses only need to implement ``get_llm()``, returning a
    LangChain chat model.
    """

    @property
    def retry_count(self) -> int:
        """Return retries used by the most recent call in this async context."""
        counter = self.__dict__.get("_retry_count_context")
        return counter.get() if counter is not None else 0

    def _retry_counter(self) -> ContextVar[int]:
        counter = self.__dict__.get("_retry_count_context")
        if counter is None:
            counter = ContextVar(f"llm_retry_count_{id(self)}", default=0)
            self.__dict__["_retry_count_context"] = counter
        return counter

    @abstractmethod
    def get_llm(self) -> Any: ...

    async def invoke(self, messages: list[ChatMessage]) -> ChatMessage:
        lc_messages = [_to_langchain_message(m) for m in messages]
        response = await self._with_retry(lambda: self.get_llm().ainvoke(lc_messages))
        return _from_langchain_message(response)

    async def invoke_structured(self, messages: list[ChatMessage], output_type: type[T]) -> T:
        """Uses the provider's native structured output via LangChain."""
        lc_messages = [_to_langchain_message(m) for m in messages]
        structured_llm = self.get_llm().with_structured_output(output_type)
        return await self._with_retry(lambda: structured_llm.ainvoke(lc_messages))

    async def _with_retry(self, call: Callable[[], Awaitable[T]]) -> T:
        retry_counter = self._retry_counter()
        retry_counter.set(0)
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return await call()
            except Exception as exc:
                if not _is_rate_limited(exc):
                    raise
                last_error = exc
                retry_counter.set(min(attempt + 1, MAX_RETRIES - 1))
                if attempt == MAX_RETRIES - 1:
                    break
                wait_time = RETRY_BASE_WAIT_SECONDS * (attempt + 1)
                _logger.warning(
                    "LLM rate limit reached; retrying after %s seconds",
                    wait_time,
                )
                await asyncio.sleep(wait_time)
        if last_error is None:
            raise RuntimeError("Retry loop completed without an error")
        raise last_error
