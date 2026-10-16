from abc import ABC, abstractmethod
from typing import AsyncIterator, Generic, TypeVar

from domain.entities.content_source import ContentSource
from domain.entities.loaded_source import LoadedSource

T = TypeVar("T")


class SourceLoader(ABC, Generic[T]):
    """Resolve a source into zero or more loaded files."""

    @abstractmethod
    def load(self, source: ContentSource) -> AsyncIterator[LoadedSource[T]]:
        ...
