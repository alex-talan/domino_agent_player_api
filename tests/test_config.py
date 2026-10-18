import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from domino_player import config
from domino_player.app import create_app
from domino_player.config import PROJECT_ROOT, Settings


@pytest.fixture(autouse=True)
def isolated_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / ".env")


def test_project_dotenv_is_loaded_from_another_working_directory(
    monkeypatch, tmp_path
) -> None:
    config.ENV_FILE.write_text(
        "AI_CORE_PROVIDER=ollama\nAI_CORE_MODEL=dotenv-model\n"
        "TURN_TIMEOUT_SECONDS=5\nOPENAI_API_KEY=test-only-key\n",
        encoding="utf-8-sig",
    )
    for key in (
        "AI_CORE_PROVIDER",
        "AI_CORE_MODEL",
        "TURN_TIMEOUT_SECONDS",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    working_directory = tmp_path / "working"
    working_directory.mkdir()
    (working_directory / ".env").write_text(
        "AI_CORE_MODEL=wrong-directory-model\n", encoding="utf-8"
    )
    monkeypatch.chdir(working_directory)

    settings = Settings.from_env()

    assert settings.provider == "ollama"
    assert settings.model == "dotenv-model"
    assert settings.turn_timeout_seconds == 5
    assert os.environ["OPENAI_API_KEY"] == "test-only-key"


def test_dotenv_does_not_override_process_environment(monkeypatch) -> None:
    config.ENV_FILE.write_text(
        "AI_CORE_PROVIDER=openai\nAI_CORE_MODEL=dotenv-model\n"
        "TURN_TIMEOUT_SECONDS=5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_CORE_PROVIDER", "ollama")
    monkeypatch.setenv("AI_CORE_MODEL", "environment-model")
    monkeypatch.setenv("TURN_TIMEOUT_SECONDS", "7")

    settings = Settings.from_env()

    assert settings.provider == "ollama"
    assert settings.model == "environment-model"
    assert settings.turn_timeout_seconds == 7


def test_asgi_startup_loads_dotenv(monkeypatch) -> None:
    config.ENV_FILE.write_text(
        "AI_CORE_PROVIDER=ollama\nAI_CORE_MODEL=dotenv-model\n"
        "TURN_TIMEOUT_SECONDS=5\n",
        encoding="utf-8",
    )
    for key in ("AI_CORE_PROVIDER", "AI_CORE_MODEL", "TURN_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)

    with TestClient(create_app()) as client:
        response = client.post("/end", json={"win": False})

    assert response.status_code == 200
    assert response.content == b""


def test_required_runtime_configuration_is_enforced(monkeypatch) -> None:
    for key in ("AI_CORE_PROVIDER", "AI_CORE_MODEL", "TURN_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="AI_CORE_PROVIDER"):
        Settings.from_env()


def test_default_prompt_directory_is_project_relative(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_CORE_PROVIDER", "ollama")
    monkeypatch.setenv("AI_CORE_MODEL", "local-model")
    monkeypatch.setenv("TURN_TIMEOUT_SECONDS", "5")
    monkeypatch.delenv("DOMINO_PROMPTS_DIR", raising=False)

    settings = Settings.from_env()

    assert settings.prompts_dir == (Path(PROJECT_ROOT) / "prompts").resolve()


def test_startup_loads_prompts_and_provider_from_another_directory(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_CORE_PROVIDER", "ollama")
    monkeypatch.setenv("AI_CORE_MODEL", "local-model")
    monkeypatch.setenv("TURN_TIMEOUT_SECONDS", "5")

    with TestClient(create_app()) as client:
        response = client.post("/end", json={"win": False})

    assert response.status_code == 200
    assert response.content == b""


def test_application_startup_fails_without_required_configuration(monkeypatch) -> None:
    for key in ("AI_CORE_PROVIDER", "AI_CORE_MODEL", "TURN_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="AI_CORE_PROVIDER"):
        with TestClient(create_app()):
            pass
