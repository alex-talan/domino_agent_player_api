from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class LoadedSource(Generic[T]):
    """Represent one resolved file and its path or downloaded content."""

    name: str
    content_type: str
    content: T

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        normalized_type = self.content_type.strip().lower().lstrip(".")

        if not normalized_name:
            raise ValueError("LoadedSource.name must not be empty.")

        if not normalized_type:
            raise ValueError("LoadedSource.content_type must not be empty.")

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "content_type", normalized_type)