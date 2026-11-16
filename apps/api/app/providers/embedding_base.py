"""Embedding provider contract and shared batching helpers."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

InputType = Literal["query", "passage"]


@runtime_checkable
class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        """Return one vector per input text, in order."""

    async def ping(self) -> bool:
        """Cheap availability probe; must not download weights."""

    async def aclose(self) -> None:
        """Release network clients or models."""
