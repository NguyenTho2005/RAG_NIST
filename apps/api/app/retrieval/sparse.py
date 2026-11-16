"""Explainable BM25 over PostgreSQL ``chunk_terms``.

Why not ``ts_rank``: the platform needs explicit Vietnamese syllable + bigram
terms, configurable BM25 parameters, and auditable corpus statistics. The terms
are written by the ingestion pipeline and scoped by the same active-revision
join used for dense retrieval.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict

from sqlalchemy import func, select

from app.core.config import Settings
from app.persistence.db import Database
from app.persistence.models import Chunk, ChunkTerm, Document
from app.retrieval.chunks import ActiveChunkLoader
from app.retrieval.text import tokenize
from app.retrieval.types import RetrievalOutcome

_MAX_QUERY_TERMS = 128
_MAX_TERM_ROWS = 50_000


class SparseRetriever:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        embedding_model: str,
    ) -> None:
        self.settings = settings
        self.database = database
        self.embedding_model = embedding_model
        self.loader = ActiveChunkLoader(database, embedding_model)

    async def search(
        self,
        *,
        workspace_id: uuid.UUID,
        kb_id: uuid.UUID,
        query: str,
        limit: int,
        document_ids: list[uuid.UUID] | None = None,
    ) -> RetrievalOutcome:
        started = time.perf_counter()
        terms = list(dict.fromkeys(tokenize(query)))[:_MAX_QUERY_TERMS]
        empty = RetrievalOutcome(
            chunks=[],
            timings={"sparse_ms": round((time.perf_counter() - started) * 1000, 2)},
            candidates={"sparse": 0},
            mode="hybrid",
            mode_used="sparse",
            embedding_model=self.embedding_model,
        )
        if not terms:
            return empty

        base = [
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
            base.append(Chunk.document_id.in_(document_ids))

        async with self.database.sessionmaker() as session:
            corpus = (
                await session.execute(
                    select(
                        func.count(Chunk.id),
                        func.coalesce(func.avg(Chunk.term_count), 1.0),
                    )
                    .join(Document, Document.id == Chunk.document_id)
                    .where(*base)
                )
            ).one()
            corpus_size = int(corpus[0] or 0)
            avg_length = float(corpus[1] or 1.0)
            if corpus_size == 0:
                return empty

            df_rows = (
                await session.execute(
                    select(ChunkTerm.term, func.count(func.distinct(ChunkTerm.chunk_id)))
                    .join(Chunk, Chunk.id == ChunkTerm.chunk_id)
                    .join(Document, Document.id == Chunk.document_id)
                    .where(ChunkTerm.term.in_(terms), *base)
                    .group_by(ChunkTerm.term)
                )
            ).all()
            document_frequency = {term: int(count) for term, count in df_rows}
            if not document_frequency:
                return empty

            rows = (
                await session.execute(
                    select(ChunkTerm.chunk_id, ChunkTerm.term, ChunkTerm.tf, Chunk.term_count)
                    .join(Chunk, Chunk.id == ChunkTerm.chunk_id)
                    .join(Document, Document.id == Chunk.document_id)
                    .where(ChunkTerm.term.in_(terms), *base)
                    .limit(_MAX_TERM_ROWS)
                )
            ).all()

        scores: dict[uuid.UUID, float] = defaultdict(float)
        k1 = self.settings.bm25_k1
        b = self.settings.bm25_b
        for chunk_id, term, tf, length in rows:
            df = document_frequency.get(term)
            if not df:
                continue
            idf = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
            length_ratio = (float(length) or 1.0) / avg_length
            denominator = float(tf) + k1 * (1.0 - b + b * length_ratio)
            scores[chunk_id] += idf * (float(tf) * (k1 + 1.0)) / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        chunks = await self.loader.load(
            workspace_id=workspace_id,
            kb_id=kb_id,
            scores=dict(ranked),
            source="sparse",
            score_key="sparse",
            document_ids=document_ids,
        )
        chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        return RetrievalOutcome(
            chunks=chunks,
            timings={"sparse_ms": round((time.perf_counter() - started) * 1000, 2)},
            candidates={"sparse": len(rows)},
            mode="hybrid",
            mode_used="sparse",
            embedding_model=self.embedding_model,
        )
