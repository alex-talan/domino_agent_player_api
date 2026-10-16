import asyncio

import pytest

from domino_player.decision import DominoPlayer
from domino_player.models import PlayRequest
from domino_player.prompts import TOKENS
from domino_player.rules import enumerate_legal_moves
from llm_providers.infrastructure.adapters._retrying_llm_provider import (
    RetryingLlmProvider,
)


class ScriptedProvider:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[list[object]] = []
        self.retry_count = 0

    async def invoke_structured(self, messages, output_type):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def parse_request(payload: dict[str, object]) -> PlayRequest:
    return PlayRequest.model_validate(payload)


@pytest.mark.asyncio
async def test_player_prompt_contains_context_and_only_literal_placeholders(
    settings, game_request
) -> None:
    provider = ScriptedProvider({"tile": 2, "position": "head"})
    player = DominoPlayer.create(settings, provider)

    await player.play(parse_request(game_request))

    assert len(provider.calls) == 1
    system_message, user_message = provider.calls[0]
    assert system_message.role == "system"
    assert user_message.role == "user"
    assert '"to_play":"p0"' in user_message.text
    assert '"tile":2,"position":"head"' in user_message.text
    assert '"tile":11,"position":"tail"' in user_message.text
    assert not any(token in user_message.text for token in TOKENS)


@pytest.mark.asyncio
async def test_legal_lower_points_choice_is_honored(settings, game_request) -> None:
    provider = ScriptedProvider({"tile": 2, "position": "head"})
    player = DominoPlayer.create(settings, provider)

    move = await player.play(parse_request(game_request))

    assert move.model_dump() == {"tile": 2, "position": "head"}


@pytest.mark.asyncio
async def test_invalid_move_gets_one_repair_and_uses_legal_response(
    settings, game_request
) -> None:
    provider = ScriptedProvider(
        {"tile": 27, "position": "tail"},
        {"tile": 2, "position": "head"},
    )
    player = DominoPlayer.create(settings, provider)

    move = await player.play(parse_request(game_request))

    assert move.model_dump() == {"tile": 2, "position": "head"}
    assert len(provider.calls) == 2
    assert "supplied legal-move list" in provider.calls[1][-1].text
    assert len(provider.calls[1]) == len(provider.calls[0]) + 1
    assert "27" not in provider.calls[1][-1].text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_response",
    [
        {"tile": 2, "position": "head", "reasoning": "extra"},
        "The move should be tile 2 at head.",
        {"tile": True, "position": "head"},
    ],
)
async def test_invalid_output_shape_gets_repaired(
    settings, game_request, bad_response
) -> None:
    provider = ScriptedProvider(
        bad_response,
        {"tile": 2, "position": "head"},
    )
    player = DominoPlayer.create(settings, provider)

    move = await player.play(parse_request(game_request))

    assert move.model_dump() == {"tile": 2, "position": "head"}
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_failed_repair_uses_highest_points_fallback(settings, game_request) -> None:
    provider = ScriptedProvider(
        {"tile": 27, "position": "tail"},
        {"tile": 27, "position": "tail"},
    )
    player = DominoPlayer.create(settings, provider)

    move = await player.play(parse_request(game_request))

    assert move.model_dump() == {"tile": 11, "position": "tail"}
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_provider_failure_falls_back_without_repair(settings, game_request) -> None:
    provider = ScriptedProvider(RuntimeError("private provider details"))
    player = DominoPlayer.create(settings, provider)

    move = await player.play(parse_request(game_request))

    assert move.model_dump() == {"tile": 11, "position": "tail"}
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_timeout_falls_back_and_cancels_provider_work(settings, game_request) -> None:
    class SlowProvider:
        retry_count = 0

        async def invoke_structured(self, messages, output_type):
            await asyncio.sleep(1)

    settings = settings.__class__(
        provider=settings.provider,
        model=settings.model,
        turn_timeout_seconds=0.01,
        temperature=settings.temperature,
        prompts_dir=settings.prompts_dir,
        host=settings.host,
        port=settings.port,
    )
    player = DominoPlayer.create(settings, SlowProvider())

    move = await player.play(parse_request(game_request))

    assert move.model_dump() == {"tile": 11, "position": "tail"}


@pytest.mark.asyncio
async def test_valid_pass_is_sent_to_llm_for_no_move_turn(settings, game_request) -> None:
    game_request.update(
        table=[1],
        head=0,
        tail=1,
        your_tiles=[27],
    )
    provider = ScriptedProvider({"tile": -1, "position": ""})
    player = DominoPlayer.create(settings, provider)

    move = await player.play(parse_request(game_request))

    assert move.model_dump() == {"tile": -1, "position": ""}
    assert len(provider.calls) == 1


def test_prompt_root_is_independent_of_working_directory(
    settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)

    player = DominoPlayer.create(settings, ScriptedProvider({"tile": 0, "position": "head"}))

    assert player.system_prompt
    assert player.player_template


@pytest.mark.asyncio
async def test_ai_core_retry_metadata_counts_actual_retries(monkeypatch) -> None:
    class RetryProvider(RetryingLlmProvider):
        def get_llm(self):
            return None

    provider = RetryProvider()
    call_count = 0

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        "llm_providers.infrastructure.adapters._retrying_llm_provider.asyncio.sleep",
        no_wait,
    )

    async def fail_twice_then_succeed() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("429 rate limit")
        return "ok"

    assert await provider._with_retry(fail_twice_then_succeed) == "ok"
    assert provider.retry_count == 2
