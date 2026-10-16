from pathlib import Path

import pytest

from domino_player.config import Settings


@pytest.fixture
def settings() -> Settings:
    project_root = Path(__file__).resolve().parents[1]
    return Settings(
        provider="openai",
        model="test-model",
        turn_timeout_seconds=0.5,
        temperature=0.0,
        prompts_dir=project_root / "prompts",
        host="127.0.0.1",
        port=8000,
    )


@pytest.fixture
def game_request() -> dict[str, object]:
    return {
        "table": [1],
        "head": 0,
        "tail": 1,
        "p0": [],
        "p1": [],
        "p2": [],
        "p3": [],
        "to_play": "p0",
        "your_tiles": [2, 11],
    }
