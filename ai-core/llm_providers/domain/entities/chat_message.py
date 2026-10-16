from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """A single role-tagged message in a chat conversation."""

    role: Role
    text: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported role: {self.role!r}")
        if not isinstance(self.text, str):
            raise TypeError("ChatMessage.text must be a string.")

    @classmethod
    def from_system(cls, text: str) -> ChatMessage:
        return cls(role="system", text=text)

    @classmethod
    def from_user(cls, text: str) -> ChatMessage:
        return cls(role="user", text=text)

    @classmethod
    def from_assistant(cls, text: str) -> ChatMessage:
        return cls(role="assistant", text=text)
