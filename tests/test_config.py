from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from domino_player.app import create_app
from domino_player.config import PROJECT_ROOT, Settings


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
