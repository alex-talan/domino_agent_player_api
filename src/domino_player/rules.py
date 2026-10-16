from __future__ import annotations

from dataclasses import dataclass

from domino_player.models import MoveResponse, PlayRequest

TILES: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 4),
    (4, 5),
    (4, 6),
    (5, 5),
    (5, 6),
    (6, 6),
)


@dataclass(frozen=True)
class LegalMove:
    tile: int
    position: str
    points: int
    resulting_head: int | None
    resulting_tail: int | None

    def response(self) -> MoveResponse:
        return MoveResponse(tile=self.tile, position=self.position)

    def prompt_item(self) -> dict[str, int | str | None]:
        return {
            "tile": self.tile,
            "position": self.position,
            "points": self.points,
            "resulting_head": self.resulting_head,
            "resulting_tail": self.resulting_tail,
        }


def tile_points(tile_id: int) -> int:
    left, right = TILES[tile_id]
    return left + right


def enumerate_legal_moves(request: PlayRequest) -> list[LegalMove]:
    moves: list[LegalMove] = []

    if not request.table:
        for tile_id in request.your_tiles:
            left, right = TILES[tile_id]
            moves.append(
                LegalMove(tile_id, "head", left + right, left, right)
            )
        return moves

    assert request.head is not None and request.tail is not None
    for tile_id in request.your_tiles:
        left, right = TILES[tile_id]
        points = left + right
        if left == request.head:
            moves.append(
                LegalMove(tile_id, "head", points, right, request.tail)
            )
        elif right == request.head:
            moves.append(
                LegalMove(tile_id, "head", points, left, request.tail)
            )
        if left == request.tail:
            moves.append(
                LegalMove(tile_id, "tail", points, request.head, right)
            )
        elif right == request.tail:
            moves.append(
                LegalMove(tile_id, "tail", points, request.head, left)
            )

    if not moves:
        moves.append(
            LegalMove(-1, "", 0, request.head, request.tail)
        )
    return moves


def choose_fallback(moves: list[LegalMove]) -> MoveResponse:
    best = min(
        moves,
        key=lambda move: (-move.points, move.tile, 0 if move.position == "head" else 1),
    )
    return best.response()
