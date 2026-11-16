"""Reranker factory."""

from __future__ import annotations

from app.core.config import Settings
from app.reranking.base import Reranker
from app.reranking.local import LocalCrossEncoderReranker


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.rerank_enabled or settings.reranker_provider == "none":
        return None
    if settings.reranker_provider == "local":
        return LocalCrossEncoderReranker(
            settings.reranker_model,
            device=settings.embedding_device,
        )
    return None
