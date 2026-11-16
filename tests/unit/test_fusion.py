"""Reciprocal Rank Fusion unit tests."""

from __future__ import annotations

import uuid

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.types import RetrievedChunk


def _chunk(name: str, source: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, name),
        document_id=uuid.uuid4(),
        revision_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
        filename=f"{name}.md",
        text=name,
        heading_path=[],
        page_start=None,
        page_end=None,
        ordinal=0,
        token_count=1,
        score=score,
        source=source,
        scores={source: score},
    )


def test_rrf_rewards_documents_in_both_lists() -> None:
    dense = [_chunk("a", "dense", 0.9), _chunk("b", "dense", 0.8), _chunk("c", "dense", 0.7)]
    sparse = [_chunk("b", "sparse", 12.0), _chunk("d", "sparse", 9.0)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert [chunk.text for chunk in fused][:2] == ["b", "a"]
    assert fused[0].source == "hybrid"
    assert fused[0].scores["dense"] == 0.8
    assert fused[0].scores["sparse"] == 12.0
    assert fused[0].scores["rrf"] > fused[1].scores["rrf"]
    assert len(fused) == 4


def test_rrf_limit() -> None:
    dense = [_chunk("a", "dense", 0.9), _chunk("b", "dense", 0.8)]
    assert len(reciprocal_rank_fusion([dense], limit=1)) == 1
