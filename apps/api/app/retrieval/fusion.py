"""Reciprocal Rank Fusion for dense and sparse candidate lists."""

from __future__ import annotations

import uuid

from app.retrieval.types import RetrievedChunk


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists without comparing incomparable score scales."""
    fused_scores: dict[uuid.UUID, float] = {}
    by_id: dict[uuid.UUID, RetrievedChunk] = {}
    component_scores: dict[uuid.UUID, dict[str, float]] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            existing = by_id.get(chunk.chunk_id)
            if existing is None:
                by_id[chunk.chunk_id] = chunk
            elif chunk.parent_text and not existing.parent_text:
                existing.parent_text = chunk.parent_text
            components = component_scores.setdefault(chunk.chunk_id, {})
            for key, value in chunk.scores.items():
                components[key] = max(components.get(key, 0.0), value)

    fused: list[RetrievedChunk] = []
    for chunk_id, score in fused_scores.items():
        chunk = by_id[chunk_id]
        chunk.score = score
        chunk.source = "hybrid"
        chunk.scores = {**component_scores[chunk_id], "rrf": score}
        fused.append(chunk)
    fused.sort(key=lambda item: item.score, reverse=True)
    return fused[:limit] if limit is not None else fused
