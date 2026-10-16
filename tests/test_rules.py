from pydantic import ValidationError
import pytest

from domino_player.models import PlayRequest
from domino_player.rules import (
    TILES,
    choose_fallback,
    enumerate_legal_moves,
    tile_points,
)


def make_request(**overrides: object) -> PlayRequest:
    payload: dict[str, object] = {
        "table": [],
        "head": None,
        "tail": None,
        "p0": [],
        "p1": [],
        "p2": [],
        "p3": [],
        "to_play": "p0",
        "your_tiles": [0],
    }
    payload.update(overrides)
    return PlayRequest.model_validate(payload)


def test_catalogue_and_tile_zero_are_preserved() -> None:
    assert len(TILES) == 28
    assert TILES[0] == (0, 0)
    assert tile_points(0) == 0
    assert [(move.tile, move.position) for move in enumerate_legal_moves(make_request())] == [
        (0, "head")
    ]


def test_empty_table_allows_each_tile_once_at_head() -> None:
    moves = enumerate_legal_moves(make_request(your_tiles=[0, 3, 27]))

    assert [(move.tile, move.position) for move in moves] == [
        (0, "head"),
        (3, "head"),
        (27, "head"),
    ]
    assert moves[1].resulting_head == 0
    assert moves[1].resulting_tail == 3


def test_both_ends_are_enumerated_independently() -> None:
    request = make_request(
        table=[1],
        head=0,
        tail=6,
        your_tiles=[2, 12, 27],
    )

    moves = enumerate_legal_moves(request)
    actual = {(move.tile, move.position) for move in moves}

    assert actual == {(2, "head"), (12, "tail"), (27, "tail")}
    assert next(move for move in moves if move.tile == 2).resulting_head == 2


def test_double_can_play_at_both_matching_ends() -> None:
    request = make_request(
        table=[1],
        head=1,
        tail=1,
        your_tiles=[7],
    )

    assert {(move.tile, move.position) for move in enumerate_legal_moves(request)} == {
        (7, "head"),
        (7, "tail"),
    }
    assert choose_fallback(enumerate_legal_moves(request)).position == "head"


def test_no_legal_tile_has_only_pass() -> None:
    request = make_request(table=[1], head=0, tail=1, your_tiles=[27])
    moves = enumerate_legal_moves(request)

    assert [(move.tile, move.position) for move in moves] == [(-1, "")]
    assert moves[0].points == 0
    assert moves[0].resulting_head == 0
    assert moves[0].resulting_tail == 1


def test_fallback_uses_points_then_identifier_then_head() -> None:
    request = make_request(
        table=[1],
        head=0,
        tail=1,
        your_tiles=[2, 3, 8, 9],
    )
    moves = enumerate_legal_moves(request)

    # Tile 9 is a legal 1-3 tile at the tail; tile 2 is 0-2 at the head.
    # The higher-points tile wins, regardless of strategy for normal decisions.
    fallback = choose_fallback(moves)

    assert (fallback.tile, fallback.position) == (9, "tail")


@pytest.mark.parametrize(
    "overrides",
    [
        {"table": [28]},
        {"your_tiles": [True]},
        {"table": [], "head": 0},
        {"table": [1], "head": None, "tail": 1},
        {"table": [1], "your_tiles": [1]},
        {"p0": None},
        {"to_play": "p0", "p0": None},
        {"your_tiles": []},
        {"extra": "not accepted"},
    ],
)
def test_invalid_request_shapes_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_request(**overrides)
