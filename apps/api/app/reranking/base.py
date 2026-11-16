"""Reranker contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    model_id: str

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        """Return ``(document_index, score)`` pairs sorted by score descending."""

    async def ping(self) -> bool:
        """Cheap availability probe; must not download weights."""

    async def aclose(self) -> None:
        """Release model resources."""
