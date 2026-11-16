"""Deterministic test-only reranker; never registered in production factory."""

from __future__ import annotations


class PreferTextReranker:
    def __init__(self, prefer: str) -> None:
        self.model_id = "test-reranker"
        self.prefer = prefer
        self.calls = 0
        self.last_documents: list[str] = []

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        self.calls += 1
        self.last_documents = list(documents)
        scores = [
            1.0 if self.prefer.casefold() in document.casefold() else 0.1 for document in documents
        ]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return ranked[:top_n]

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None
