from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Literal, TypedDict

from agents.infrastructure.adapters.structured_agent import StructuredAgent
from langchain_core.exceptions import OutputParserException
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from prompt_providers.infrastructure.adapters.static_prompt_provider import (
    StaticPromptProvider,
)
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


class TurnInput(TypedDict):
    request: PlayRequest
    request_id: str
    started: float
    deadline: float


class TurnState(TurnInput):
    legal_moves: list[LegalMove]
    player_prompt: str
    candidate: object
    move: MoveResponse | None
    attempts: int
    retry_count: int
    source: Literal["llm", "fallback"]
    failure_category: str | None


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
        # StaticPromptProvider uses f-string templates; preserve literal JSON braces.
        escaped_system = system_prompt.replace("{", "{{").replace("}", "}}")
        self.prompt_provider = StaticPromptProvider(escaped_system)
        self.agent = StructuredAgent(self.prompt_provider, llm_provider, MoveResponse)
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph[TurnState, None, TurnInput, TurnState]:
        builder = StateGraph(TurnState, input_schema=TurnInput)
        builder.add_node("prepare", self._prepare)
        builder.add_node("decide", self._decide)
        builder.add_node("validate", self._validate)
        builder.add_node("repair", self._repair)
        builder.add_node("fallback", self._fallback)
        builder.add_node("record", self._record)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "decide")
        builder.add_edge("decide", "validate")
        builder.add_conditional_edges(
            "validate", self._route_validation, ["repair", "fallback", "record"]
        )
        builder.add_edge("repair", "validate")
        builder.add_edge("fallback", "record")
        builder.add_edge("record", END)
        return builder.compile()

    @classmethod
    def create(
        cls,
        settings: Settings,
        llm_provider: LlmProvider,
    ) -> DominoPlayer:
        system_prompt, player_template = load_prompt_files(settings.prompts_dir)
        return cls(settings, llm_provider, system_prompt, player_template)

    async def play(self, request: PlayRequest) -> MoveResponse:
        started = time.monotonic()
        deadline = (
            asyncio.get_running_loop().time() + self.settings.turn_timeout_seconds
        )
        state: TurnInput = {
            "request": request,
            "request_id": uuid.uuid4().hex,
            "started": started,
            "deadline": deadline,
        }
        result = await self.graph.ainvoke(state, config={"recursion_limit": 12})
        move = result["move"]
        if not isinstance(move, MoveResponse):
            raise RuntimeError("Domino graph did not produce a validated move")
        return move

    async def _prepare(self, state: TurnInput) -> dict[str, object]:
        legal_moves = enumerate_legal_moves(state["request"])
        return {
            "legal_moves": legal_moves,
            "player_prompt": render_player_prompt(
                self.player_template, state["request"], legal_moves
            ),
            "attempts": 0,
            "retry_count": 0,
            "candidate": None,
            "move": None,
            "failure_category": None,
        }

    async def _decide(self, state: TurnState) -> dict[str, object]:
        return await self._generate(state, repair=False)

    async def _repair(self, state: TurnState) -> dict[str, object]:
        return await self._generate(state, repair=True)

    async def _generate(self, state: TurnState, *, repair: bool) -> dict[str, object]:
        if asyncio.get_running_loop().time() >= state["deadline"]:
            return {"candidate": None, "failure_category": "turn_deadline"}
        query = state["player_prompt"]
        history: list[ChatMessage] | None = None
        if repair:
            category = state["failure_category"]
            assert category is not None
            query = REPAIR_PROMPTS[category]
            history = [ChatMessage.from_user(state["player_prompt"])]
        update: dict[str, object] = {
            "attempts": state["attempts"] + 1,
            "candidate": None,
            "failure_category": None,
        }
        try:
            async with asyncio.timeout_at(state["deadline"]):
                update["candidate"] = await self.agent.invoke(
                    query, history=history
                )
        except (ValidationError, OutputParserException):
            update["failure_category"] = "invalid_response_schema"
        except TimeoutError:
            update["failure_category"] = "turn_deadline"
        except Exception as model_error:
            update["failure_category"] = type(model_error).__name__
        finally:
            update["retry_count"] = state["retry_count"] + getattr(
                self.llm_provider, "retry_count", 0
            )
        return update

    async def _validate(self, state: TurnState) -> dict[str, object]:
        if state["failure_category"] is not None:
            return {}
        if asyncio.get_running_loop().time() >= state["deadline"]:
            return {"failure_category": "turn_deadline"}
        try:
            move = MoveResponse.model_validate(state["candidate"], strict=True)
        except ValidationError:
            return {"failure_category": "invalid_response_schema"}
        legal = {(item.tile, item.position) for item in state["legal_moves"]}
        if (move.tile, move.position) not in legal:
            return {"failure_category": "move_not_in_legal_moves"}
        return {"move": move, "source": "llm"}

    @staticmethod
    async def _route_validation(
        state: TurnState,
    ) -> Literal["repair", "fallback", "record"]:
        if state["move"] is not None:
            return "record"
        if (
            state["failure_category"] in REPAIR_PROMPTS
            and state["attempts"] == 1
            and asyncio.get_running_loop().time() < state["deadline"]
        ):
            return "repair"
        return "fallback"

    async def _fallback(self, state: TurnState) -> dict[str, object]:
        return {
            "move": choose_fallback(state["legal_moves"]),
            "source": "fallback",
        }

    async def _record(self, state: TurnState) -> dict[str, object]:
        move = state["move"]
        assert move is not None
        elapsed_ms = int((time.monotonic() - state["started"]) * 1000)
        logger.info(
            "domino_decision request_id=%s player=%s tile=%d position=%s "
            "source=%s elapsed_ms=%d retry_count=%d failure_category=%s "
            "token_usage=unavailable",
            state["request_id"],
            state["request"].to_play,
            move.tile,
            move.position,
            state["source"],
            elapsed_ms,
            state["retry_count"],
            state["failure_category"] or "none",
        )
        return {}
