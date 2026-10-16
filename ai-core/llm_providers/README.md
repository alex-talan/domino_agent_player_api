# llm_providers

Unified async interface for chat-completion providers: OpenAI, Azure OpenAI,
Bedrock, Google (Gemini Developer API and Vertex AI), Ollama, and OpenRouter.

## Layout

- `domain/entities/chat_message.py` — plain `ChatMessage` (role + text), no framework deps.
- `domain/ports/llm_provider.py` — `LlmProvider` port: `get_llm()`, `invoke()`, `invoke_structured()`.
- `infrastructure/adapters/` — one adapter per provider, each backed by a LangChain chat model.
- `infrastructure/adapters/_retrying_llm_provider.py` — shared retry-on-rate-limit base class.

## Design

- Adapters build a LangChain `BaseChatModel` in `__init__` and expose it via `get_llm()`.
- `invoke()` converts domain `ChatMessage` objects to LangChain messages, calls `ainvoke`,
  and converts the reply back.
- `invoke_structured()` uses `get_llm().with_structured_output(output_type)`, so each
  provider's native structured-output mechanism (function calling, JSON mode, etc.)
  is used instead of manual prompt/schema injection.
- `RetryingLlmProvider` retries both methods on rate limits (`openai.RateLimitError`,
  or a `429`/`RESOURCE_EXHAUSTED`/`ThrottlingException` marker in the error message),
  with linear backoff.
- Vertex AI and the Gemini Developer API share `ChatGoogleGenerativeAI`; passing
  `project`/`location` with `vertexai=True` selects Vertex (Application Default
  Credentials only — no API-key mode).

## Adding a provider

1. Add a class in `infrastructure/adapters/` that builds the LangChain chat model
   in `__init__` and extends `RetryingLlmProvider`.
2. Implement only `get_llm()`. `invoke()`/`invoke_structured()` are inherited.
3. Register it in `__init__.py`'s lazy adapter map.

## Tests

```bash
cd llm_providers
pip install -e ".[dev]"
pytest
```
