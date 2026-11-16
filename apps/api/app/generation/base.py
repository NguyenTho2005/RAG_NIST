"""LLM provider contract used by grounded generation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

Message = dict[str, str]


@dataclass(slots=True)
class LLMResult:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    model_id: str

    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> LLMResult:
        """Return one complete assistant response."""

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield assistant text chunks in order."""

    async def ping(self) -> bool:
        """Cheap availability/model probe; must not load weights."""

    async def aclose(self) -> None:
        """Release network clients."""
