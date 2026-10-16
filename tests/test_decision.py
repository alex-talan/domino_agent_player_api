import asyncio
import logging
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import TypeVar, cast
from unittest.mock import AsyncMock

import pytest

from agents import StructuredAgent
from prompt_providers import StaticPromptProvider

from domino_player.decision import DominoPlayer, TurnInput
from domino_player.models import MoveResponse, PlayRequest
from domino_player.prompts import TOKENS
from llm_providers.domain.entities.chat_message import ChatMessage
from llm_providers.domain.ports.llm_provider import LlmProvider
from llm_providers.infrastructure.adapters._retrying_llm_provider import (
    RetryingLlmProvider,
)


OutputT = TypeVar("OutputT")


class ScriptedProvider(LlmProvider):
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    def get_llm(self) -> None:
        return None

    async def invoke(self, messages: list[ChatMessage]) -> ChatMessage:
        raise AssertionError("The domino agent must request structured output")

    async def invoke_structured(
        self, messages: list[ChatMessage], output_type: type[OutputT]
    ) -> OutputT:
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        # Deliberately allow malformed provider output to test local validation.
        return cast(OutputT, response)


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
async def test_real_ai_core_agent_and_prompt_provider_are_used(
    settings, game_request, monkeypatch
) -> None:
    provider = ScriptedProvider({"tile": 2, "position": "head"})
    player = DominoPlayer.create(settings, provider)
    assert isinstance(player.agent, StructuredAgent)
    assert isinstance(player.prompt_provider, StaticPromptProvider)
    invoke = AsyncMock(wraps=player.agent.invoke)
    get_prompt = AsyncMock(wraps=player.prompt_provider.get_prompt)
    monkeypatch.setattr(player.agent, "invoke", invoke)
    monkeypatch.setattr(player.prompt_provider, "get_prompt", get_prompt)

    move = await player.play(parse_request(game_request))

    assert move.tile == 2
    invoke.assert_awaited_once()
    get_prompt.assert_awaited_once_with({})
    assert provider.calls[0][0].text == player.system_prompt
    assert player.agent.output_type is MoveResponse


def test_readme_mermaid_matches_compiled_graph(settings) -> None:
    player = DominoPlayer.create(settings, ScriptedProvider())
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    diagram = readme.split("```mermaid\n", 1)[1].split("```", 1)[0]
    documented_edges = set(
        re.findall(r"^\s*(\w+).*?-->(?:\|[^|]*\|)?\s*(\w+)", diagram, re.MULTILINE)
    )
    aliases = {"__start__": "START", "__end__": "END"}
    actual_edges = {
        (aliases.get(edge.source, edge.source), aliases.get(edge.target, edge.target))
        for edge in player.graph.get_graph().edges
    }

    assert diagram.startswith("flowchart TD\n")
    assert documented_edges == actual_edges


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("responses", "expected_path", "expected_source"),
    [
        (
            [{"tile": 2, "position": "head"}],
            ["prepare", "decide", "validate", "record"],
            "llm",
        ),
        (
            [{"tile": 27, "position": "head"}, {"tile": 2, "position": "head"}],
            ["prepare", "decide", "validate", "repair", "validate", "record"],
            "llm",
        ),
        (
            [{"tile": -1, "position": ""}, {"tile": -1, "position": ""}],
            ["prepare", "decide", "validate", "repair", "validate", "fallback", "record"],
            "fallback",
        ),
        (
            [RuntimeError("provider unavailable")],
            ["prepare", "decide", "validate", "fallback", "record"],
            "fallback",
        ),
    ],
)
async def test_graph_executes_expected_nodes(
    settings, game_request, responses, expected_path, expected_source, caplog
) -> None:
    provider = ScriptedProvider(*responses)
    player = DominoPlayer.create(settings, provider)
    initial: TurnInput = {
        "request": parse_request(game_request),
        "request_id": uuid.uuid4().hex,
        "started": time.monotonic(),
        "deadline": asyncio.get_running_loop().time() + settings.turn_timeout_seconds,
    }
    path: list[str] = []
    with caplog.at_level(logging.INFO, logger="domino_player.decision"):
        async for update in player.graph.astream(initial, stream_mode="updates"):
            path.extend(update)

    assert path == expected_path
    assert f"source={expected_source}" in caplog.text
    assert player.graph.checkpointer is None


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
    class SlowProvider(ScriptedProvider):
        def __init__(self) -> None:
            super().__init__()
            self.cancelled = False

        async def invoke_structured(
            self, messages: list[ChatMessage], output_type: type[OutputT]
        ) -> OutputT:
            try:
                await asyncio.sleep(1)
            finally:
                self.cancelled = True
            raise AssertionError("Provider work must be cancelled before completion")

    settings = settings.__class__(
        provider=settings.provider,
        model=settings.model,
        turn_timeout_seconds=0.01,
        temperature=settings.temperature,
        prompts_dir=settings.prompts_dir,
        host=settings.host,
        port=settings.port,
    )
    provider = SlowProvider()
    player = DominoPlayer.create(settings, provider)

    started = time.monotonic()
    move = await player.play(parse_request(game_request))

    assert move.model_dump() == {"tile": 11, "position": "tail"}
    assert time.monotonic() - started < 0.2
    assert provider.cancelled


@pytest.mark.asyncio
async def test_repair_shares_the_original_deadline(
    settings, game_request, caplog, monkeypatch
) -> None:
    class SlowRepairProvider(ScriptedProvider):
        async def invoke_structured(
            self, messages: list[ChatMessage], output_type: type[OutputT]
        ) -> OutputT:
            if not self.calls:
                await asyncio.sleep(0.03)
                return await super().invoke_structured(messages, output_type)
            self.calls.append(messages)
            await asyncio.sleep(1)
            raise AssertionError("Repair must be cancelled before completion")

    provider = SlowRepairProvider({"tile": -1, "position": ""})
    player = DominoPlayer.create(
        replace(settings, turn_timeout_seconds=0.08), provider
    )
    deadlines: list[float | None] = []
    timeout_at = asyncio.timeout_at

    def track_deadline(when: float | None) -> asyncio.Timeout:
        deadlines.append(when)
        return timeout_at(when)

    monkeypatch.setattr("domino_player.decision.asyncio.timeout_at", track_deadline)

    started = time.monotonic()
    with caplog.at_level(logging.INFO, logger="domino_player.decision"):
        move = await player.play(parse_request(game_request))
    elapsed = time.monotonic() - started

    assert move.model_dump() == {"tile": 11, "position": "tail"}
    assert len(provider.calls) == 2
    assert elapsed < 0.2
    assert len(deadlines) == 2
    assert deadlines[0] == deadlines[1]
    assert "failure_category=turn_deadline" in caplog.text


@pytest.mark.asyncio
async def test_concurrent_graph_runs_do_not_share_hands_or_messages(
    settings, game_request
) -> None:
    class ConcurrentProvider(ScriptedProvider):
        def __init__(self) -> None:
            super().__init__()
            self.both_started = asyncio.Event()

        async def invoke_structured(
            self, messages: list[ChatMessage], output_type: type[OutputT]
        ) -> OutputT:
            self.calls.append(messages)
            if len(self.calls) == 2:
                self.both_started.set()
            await self.both_started.wait()
            if '"your_tiles":[11]' in messages[-1].text:
                return cast(OutputT, MoveResponse(tile=11, position="tail"))
            return cast(OutputT, MoveResponse(tile=2, position="head"))

    provider = ConcurrentProvider()
    player = DominoPlayer.create(settings, provider)
    second_request = {**game_request, "to_play": "p1", "your_tiles": [11]}

    first, second = await asyncio.gather(
        player.play(parse_request(game_request)),
        player.play(parse_request(second_request)),
    )

    assert first.model_dump() == {"tile": 2, "position": "head"}
    assert second.model_dump() == {"tile": 11, "position": "tail"}
    assert len(provider.calls) == 2
    assert all(len(messages) == 2 for messages in provider.calls)


@pytest.mark.asyncio
async def test_retry_counts_survive_graph_node_contexts(
    settings, game_request, monkeypatch, caplog
) -> None:
    class RetryThenRepairProvider(RetryingLlmProvider):
        def get_llm(self):
            return None

        async def invoke_structured(
            self, messages: list[ChatMessage], output_type: type[OutputT]
        ) -> OutputT:
            is_repair = len(messages) == 3
            retries_before_success = 2 if is_repair else 1
            attempts = 0

            async def generate():
                nonlocal attempts
                attempts += 1
                if attempts <= retries_before_success:
                    raise RuntimeError("429")
                if is_repair:
                    return {"tile": 2, "position": "head"}
                return {"tile": -1, "position": ""}

            return cast(OutputT, await self._with_retry(generate))

    monkeypatch.setattr(
        "llm_providers.infrastructure.adapters._retrying_llm_provider.RETRY_BASE_WAIT_SECONDS",
        0,
    )
    player = DominoPlayer.create(settings, RetryThenRepairProvider())

    with caplog.at_level(logging.INFO, logger="domino_player.decision"):
        move = await player.play(parse_request(game_request))

    assert move.model_dump() == {"tile": 2, "position": "head"}
    assert "retry_count=3" in caplog.text
    assert "source=llm" in caplog.text


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
