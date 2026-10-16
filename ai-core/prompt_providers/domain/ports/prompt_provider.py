from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ContextT_contra = TypeVar("ContextT_contra", contravariant=True)
PromptT_co = TypeVar("PromptT_co", covariant=True)


class PromptProvider(ABC, Generic[ContextT_contra, PromptT_co]):
    """Provide a prompt for an application context."""

    @abstractmethod
    async def get_prompt(self, context: ContextT_contra) -> PromptT_co:
        """Return a reusable prompt configured for *context*."""
        ...
