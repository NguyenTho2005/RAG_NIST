"""Shared retrieval data types."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    revision_id: uuid.UUID
    workspace_id: uuid.UUID
    knowledge_base_id: uuid.UUID
    filename: str
    text: str
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    ordinal: int
    token_count: int
    score: float
    source: str = "dense"
    parent_chunk_id: uuid.UUID | None = None
    parent_text: str | None = None
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalOutcome:
    chunks: list[RetrievedChunk]
    timings: dict[str, float]
    candidates: dict[str, int]
    mode: str
    mode_used: str
    embedding_model: str
    warnings: list[str] = field(default_factory=list)
