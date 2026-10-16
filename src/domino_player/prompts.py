from __future__ import annotations

import json
from pathlib import Path

from domino_player.models import PlayRequest
from domino_player.rules import LegalMove, TILES

GAME_STATE_TOKEN = "{{GAME_STATE_JSON}}"
DECODED_CONTEXT_TOKEN = "{{DECODED_CONTEXT_JSON}}"
LEGAL_MOVES_TOKEN = "{{LEGAL_MOVES_JSON}}"
TOKENS = (GAME_STATE_TOKEN, DECODED_CONTEXT_TOKEN, LEGAL_MOVES_TOKEN)


def load_prompt_files(prompts_dir: Path) -> tuple[str, str]:
    system_path = prompts_dir / "domino-system.md"
    player_path = prompts_dir / "domino-player.md"
    if not system_path.is_file():
        raise FileNotFoundError(f"Required system prompt not found: {system_path}")
    if not player_path.is_file():
        raise FileNotFoundError(f"Required player prompt not found: {player_path}")

    system_prompt = system_path.read_text(encoding="utf-8")
    player_template = player_path.read_text(encoding="utf-8")
    if not system_prompt.strip() or not player_template.strip():
        raise ValueError("Domino prompt files must not be empty")
    for token in TOKENS:
        if player_template.count(token) != 1:
            raise ValueError(f"Player prompt must contain {token} exactly once")
    return system_prompt, player_template


def render_player_prompt(
    template: str,
    request: PlayRequest,
    legal_moves: list[LegalMove],
) -> str:
    hand = [
        {
            "tile": tile_id,
            "values": list(TILES[tile_id]),
            "points": sum(TILES[tile_id]),
        }
        for tile_id in request.your_tiles
    ]
    table = [
        {"tile": tile_id, "values": list(TILES[tile_id])}
        for tile_id in request.table
    ]
    replacements = {
        GAME_STATE_TOKEN: _to_json(request.model_dump(mode="json")),
        DECODED_CONTEXT_TOKEN: _to_json({"hand": hand, "table": table}),
        LEGAL_MOVES_TOKEN: _to_json([move.prompt_item() for move in legal_moves]),
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if any(token in rendered for token in TOKENS):
        raise ValueError("Rendered player prompt contains an unresolved token")
    return rendered


def _to_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
