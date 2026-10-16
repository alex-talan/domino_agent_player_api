from __future__ import annotations

import asyncio
import logging
import time
import uuid

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from domino_player.config import Settings
from domino_player.models import MoveResponse, PlayRequest
from domino_player.prompts import load_prompt_files, render_player_prompt
from domino_player.rules import LegalMove, choose_fallback, enumerate_legal_moves
from llm_providers.domain.entities.chat_message import ChatMessage
from llm_providers.domain.ports.llm_provider import LlmProvider

logger = logging.getLogger(__name__)
REPAIR_PROMPTS = {
    "invalid_response_schema": (
        "The previous response did not match the required two-field JSON schema. "
        "Return exactly the required JSON object with no extra keys or prose."
    ),
    "move_not_in_legal_moves": (
        "The previous move was not in the supplied legal-move list. Choose exactly "
        "one listed move and return only its tile and position as the required JSON."
    ),
}


class InvalidModelResponse(Exception):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class DominoPlayer:
    def __init__(
        self,
        settings: Settings,
        llm_provider: LlmProvider,
        system_prompt: str,
        player_template: str,
    ) -> None:
        self.settings = settings
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt
        self.player_template = player_template

    @classmethod
    def create(
        cls,
        settings: Settings,
        llm_provider: LlmProvider,
    ) -> DominoPlayer:
        system_prompt, player_template = load_prompt_files(settings.prompts_dir)
        return cls(settings, llm_provider, system_prompt, player_template)

    async def play(self, request: PlayRequest) -> MoveResponse:
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.turn_timeout_seconds
        legal_moves = enumerate_legal_moves(request)
        player_prompt = render_player_prompt(
            self.player_template, request, legal_moves
        )
        messages = [
            ChatMessage.from_system(self.system_prompt),
            ChatMessage.from_user(player_prompt),
        ]
        retries = 0
        failure_category: str | None = None

        async def invoke(messages_to_use: list[ChatMessage]) -> MoveResponse:
            nonlocal retries
            try:
                return await self._invoke_and_validate(messages_to_use, legal_moves)
            finally:
                retries += self._retry_count()

        try:
            async with asyncio.timeout_at(deadline):
                if loop.time() >= deadline:
                    raise TimeoutError
                try:
                    move = await invoke(messages)
                except InvalidModelResponse as first_error:
                    failure_category = first_error.category
                    if loop.time() >= deadline:
                        raise TimeoutError
                    repair_messages = [
                        *messages,
                        ChatMessage.from_user(REPAIR_PROMPTS[first_error.category]),
                    ]
                    move = await invoke(repair_messages)
                return self._record_decision(
                    request,
                    move,
                    request_id,
                    "llm",
                    started,
                    retries,
                    None,
                )
        except TimeoutError:
            failure_category = "turn_deadline"
        except Exception as model_error:
            failure_category = _failure_category(model_error)

        fallback = choose_fallback(legal_moves)
        return self._record_decision(
            request,
            fallback,
            request_id,
            "fallback",
            started,
            retries,
            failure_category or "invalid_model_response",
        )

    async def _invoke_and_validate(
        self,
        messages: list[ChatMessage],
        legal_moves: list[LegalMove],
    ) -> MoveResponse:
        try:
            response = await self.llm_provider.invoke_structured(
                messages, MoveResponse
            )
        except (ValidationError, OutputParserException) as exc:
            raise InvalidModelResponse("invalid_response_schema") from exc
        try:
            move = MoveResponse.model_validate(response, strict=True)
        except ValidationError as exc:
            raise InvalidModelResponse("invalid_response_schema") from exc

        legal = {(item.tile, item.position) for item in legal_moves}
        if (move.tile, move.position) not in legal:
            raise InvalidModelResponse("move_not_in_legal_moves")
        return move

    def _retry_count(self) -> int:
        return getattr(self.llm_provider, "retry_count", 0)

    @staticmethod
    def _record_decision(
        request: PlayRequest,
        move: MoveResponse,
        request_id: str,
        source: str,
        started: float,
        retries: int,
        failure_category: str | None,
    ) -> MoveResponse:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "domino_decision request_id=%s player=%s tile=%d position=%s "
            "source=%s elapsed_ms=%d retry_count=%d failure_category=%s "
            "token_usage=unavailable",
            request_id,
            request.to_play,
            move.tile,
            move.position,
            source,
            elapsed_ms,
            retries,
            failure_category or "none",
        )
        return move


def _failure_category(error: Exception) -> str:
    if isinstance(error, TimeoutError):
        return "provider_timeout"
    if isinstance(error, InvalidModelResponse):
        return error.category
    return type(error).__name__
