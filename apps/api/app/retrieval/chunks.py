"""Load authoritative active chunks for retrieval candidates.

The database join is the publication boundary: Qdrant/BM25 candidates that do
not map to an active child chunk of the document's active revision are dropped.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.persistence.db import Database
from app.persistence.models import Chunk, Document
from app.retrieval.types import RetrievedChunk


class ActiveChunkLoader:
    def __init__(self, database: Database, embedding_model: str) -> None:
        self.database = database
        self.embedding_model = embedding_model

    async def load(
        self,
        *,
        workspace_id: uuid.UUID,
        kb_id: uuid.UUID,
        scores: dict[uuid.UUID, float],
        source: str,
        score_key: str,
        document_ids: list[uuid.UUID] | None = None,
        extra_scores: dict[uuid.UUID, dict[str, float]] | None = None,
    ) -> list[RetrievedChunk]:
        if not scores:
            return []
        conditions = [
            Chunk.id.in_(list(scores)),
            Chunk.workspace_id == workspace_id,
            Chunk.knowledge_base_id == kb_id,
            Chunk.kind == "child",
            Chunk.active.is_(True),
            Chunk.embedding_model == self.embedding_model,
            Chunk.revision_id == Document.active_revision_id,
            Document.workspace_id == workspace_id,
            Document.deleted_at.is_(None),
        ]
        if document_ids:
            conditions.append(Chunk.document_id.in_(document_ids))
        async with self.database.sessionmaker() as session:
            rows = (
                await session.execute(
                    select(Chunk, Document.filename)
                    .join(Document, Document.id == Chunk.document_id)
                    .where(*conditions)
                )
            ).all()
            parent_ids = {chunk.parent_chunk_id for chunk, _ in rows if chunk.parent_chunk_id}
            parents: dict[uuid.UUID, str] = {}
            if parent_ids:
                parent_rows = (
                    await session.execute(
                        select(Chunk.id, Chunk.text).where(Chunk.id.in_(parent_ids))
                    )
                ).all()
                parents = dict(parent_rows)

        result: list[RetrievedChunk] = []
        for chunk, filename in rows:
            score = scores.get(chunk.id)
            if score is None:
                continue
            chunk_scores = dict((extra_scores or {}).get(chunk.id, {}))
            chunk_scores[score_key] = score
            result.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    revision_id=chunk.revision_id,
                    workspace_id=chunk.workspace_id,
                    knowledge_base_id=chunk.knowledge_base_id,
                    filename=filename,
                    text=chunk.text,
                    heading_path=list(chunk.heading_path or []),
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    ordinal=chunk.ordinal,
                    token_count=chunk.token_count,
                    score=score,
                    source=source,
                    parent_chunk_id=chunk.parent_chunk_id,
                    parent_text=parents.get(chunk.parent_chunk_id)
                    if chunk.parent_chunk_id
                    else None,
                    scores=chunk_scores,
                )
            )
        return result
