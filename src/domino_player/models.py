from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator


class PlayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    table: list[StrictInt]
    head: StrictInt | None
    tail: StrictInt | None
    p0: list[StrictInt] | None
    p1: list[StrictInt] | None
    p2: list[StrictInt] | None
    p3: list[StrictInt] | None
    to_play: Literal["p0", "p1", "p2", "p3"]
    your_tiles: list[StrictInt] = Field(min_length=1, max_length=7)

    @model_validator(mode="after")
    def validate_game_state(self) -> PlayRequest:
        if len(set(self.table)) != len(self.table):
            raise ValueError("table tile identifiers must be distinct")
        if any(tile < 0 or tile > 27 for tile in self.table):
            raise ValueError("table tile identifiers must be between 0 and 27")
        if len(set(self.your_tiles)) != len(self.your_tiles):
            raise ValueError("your_tiles identifiers must be distinct")
        if any(tile < 0 or tile > 27 for tile in self.your_tiles):
            raise ValueError("your_tiles identifiers must be between 0 and 27")
        if set(self.table).intersection(self.your_tiles):
            raise ValueError("table and your_tiles must not overlap")

        if self.table:
            if self.head is None or self.tail is None:
                raise ValueError("head and tail are required for a nonempty table")
            if not 0 <= self.head <= 6 or not 0 <= self.tail <= 6:
                raise ValueError("head and tail must be between 0 and 6")
        elif self.head is not None or self.tail is not None:
            raise ValueError("head and tail must be null for an empty table")

        for player in ("p0", "p1", "p2", "p3"):
            history = getattr(self, player)
            if history is not None and any(move < -1 or move > 27 for move in history):
                raise ValueError(f"{player} history entries must be between -1 and 27")

        if getattr(self, self.to_play) is None:
            raise ValueError("to_play history must not be null")
        return self


class MoveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, revalidate_instances="always")

    tile: StrictInt = Field(ge=-1, le=27)
    position: Literal["head", "tail", ""]


class EndRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    win: StrictBool
