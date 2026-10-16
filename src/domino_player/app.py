from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from domino_player.config import Settings, create_provider
from domino_player.decision import DominoPlayer
from domino_player.models import EndRequest, MoveResponse, PlayRequest
from llm_providers.domain.ports.llm_provider import LlmProvider

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    llm_provider: LlmProvider | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime_settings = settings or Settings.from_env()
        runtime_provider = (
            llm_provider if llm_provider is not None else create_provider(runtime_settings)
        )
        application.state.domino_player = DominoPlayer.create(
            runtime_settings, runtime_provider
        )
        yield

    application = FastAPI(
        title="GenAI Domino Player",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> Response:
        return _invalid_request_response(exc.errors())

    @application.exception_handler(ValidationError)
    async def pydantic_validation_error(
        _request: Request, exc: ValidationError
    ) -> Response:
        return _invalid_request_response(exc.errors())

    @application.post("/play", response_model=MoveResponse)
    async def play(request: Request, payload: PlayRequest) -> MoveResponse:
        player: DominoPlayer = request.app.state.domino_player
        return await player.play(payload)

    @application.post("/end", status_code=200)
    async def end(payload: EndRequest) -> Response:
        logger.info("domino_game_end win=%s", payload.win)
        return Response(status_code=200)

    return application


def _invalid_request_response(errors: list[dict[str, object]]) -> Response:
    details = [
        {
            "loc": error.get("loc", ()),
            "msg": error.get("msg", "Invalid request"),
            "type": error.get("type", "validation_error"),
        }
        for error in errors
    ]
    return Response(
        content=json.dumps({"error": "invalid_request", "details": details}),
        status_code=400,
        media_type="application/json",
    )


app = create_app()
