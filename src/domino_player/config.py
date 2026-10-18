from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from llm_providers.domain.ports.llm_provider import LlmProvider

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    turn_timeout_seconds: float
    temperature: float
    prompts_dir: Path
    host: str
    port: int

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(dotenv_path=ENV_FILE, override=False, encoding="utf-8-sig")
        provider = os.getenv("AI_CORE_PROVIDER", "").strip().lower()
        model = os.getenv("AI_CORE_MODEL", "").strip()
        if not provider:
            raise ValueError("AI_CORE_PROVIDER must be configured")
        if not model:
            raise ValueError("AI_CORE_MODEL must be configured")
        if provider not in {
            "openai",
            "azure-openai",
            "google",
            "bedrock",
            "ollama",
            "openrouter",
            "vertex-ai",
        }:
            raise ValueError(f"Unsupported AI_CORE_PROVIDER: {provider}")

        timeout = _positive_float("TURN_TIMEOUT_SECONDS", required=True)
        temperature = _positive_float("AI_CORE_TEMPERATURE", default=0.0)
        if temperature > 2:
            raise ValueError("AI_CORE_TEMPERATURE must be between 0 and 2")

        configured_prompts = os.getenv("DOMINO_PROMPTS_DIR", "").strip()
        prompts_dir = Path(configured_prompts) if configured_prompts else PROJECT_ROOT / "prompts"
        if not prompts_dir.is_absolute():
            prompts_dir = PROJECT_ROOT / prompts_dir

        port_text = os.getenv("APP_PORT", "8000")
        try:
            port = int(port_text)
        except ValueError as exc:
            raise ValueError("APP_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("APP_PORT must be between 1 and 65535")

        return cls(
            provider=provider,
            model=model,
            turn_timeout_seconds=timeout,
            temperature=temperature,
            prompts_dir=prompts_dir.resolve(),
            host=os.getenv("APP_HOST", "0.0.0.0"),
            port=port,
        )


def create_provider(settings: Settings) -> LlmProvider:
    provider_name = settings.provider
    model = settings.model
    temperature = settings.temperature

    if provider_name == "openai":
        api_key = _required_env("OPENAI_API_KEY")
        from llm_providers.infrastructure.adapters.openai_llm_provider import (
            OpenAiLlmProvider,
        )

        return OpenAiLlmProvider(model_name=model, api_key=api_key, temperature=temperature)

    if provider_name == "azure-openai":
        from llm_providers.infrastructure.adapters.azure_openai_llm_provider import (
            AzureOpenAiLlmProvider,
        )

        return AzureOpenAiLlmProvider(
            azure_openai_endpoint=_required_env("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=_required_env("AZURE_OPENAI_API_KEY"),
            azure_openai_api_version=_required_env("AZURE_OPENAI_API_VERSION"),
            azure_openai_deployment=model,
            azure_openai_temperature=temperature,
        )

    if provider_name == "google":
        from llm_providers.infrastructure.adapters.google_llm_provider import (
            GoogleLlmProvider,
        )

        return GoogleLlmProvider(
            model_name=model,
            api_key=_required_env("GOOGLE_API_KEY"),
            temperature=temperature,
        )

    if provider_name == "bedrock":
        from llm_providers.infrastructure.adapters.bedrock_llm_provider import (
            BedrockLlmProvider,
        )

        region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "")).strip()
        if not region:
            raise ValueError("AWS_REGION or AWS_DEFAULT_REGION must be configured for Bedrock")
        return BedrockLlmProvider(model_id=model, region_name=region, temperature=temperature)

    if provider_name == "ollama":
        from llm_providers.infrastructure.adapters.ollama_llm_provider import (
            OllamaLlmProvider,
        )

        return OllamaLlmProvider(
            model_name=model,
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        )

    if provider_name == "openrouter":
        _required_env("OPENROUTER_API_KEY")
        from llm_providers.infrastructure.adapters.open_router_llm_provider import (
            OpenRouterLlmProvider,
        )

        return OpenRouterLlmProvider(model_name=model, temperature=temperature)

    if provider_name == "vertex-ai":
        from llm_providers.infrastructure.adapters.vertex_ai_llm_provider import (
            VertexAiLlmProvider,
        )

        return VertexAiLlmProvider(
            project=_required_env("GOOGLE_CLOUD_PROJECT"),
            location=_required_env("GOOGLE_CLOUD_LOCATION"),
            model=model,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported AI_CORE_PROVIDER: {provider_name}")


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be configured for the selected AI-Core provider")
    return value


def _positive_float(
    name: str,
    *,
    default: float | None = None,
    required: bool = False,
) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        if required:
            raise ValueError(f"{name} must be configured")
        assert default is not None
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if parsed <= 0 and required:
        raise ValueError(f"{name} must be greater than zero")
    if parsed < 0:
        raise ValueError(f"{name} must not be negative")
    return parsed
