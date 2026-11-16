"""Dense retrieval over Qdrant with PostgreSQL as the publication authority.

Qdrant supplies candidate ids and vector scores. The shared active-chunk loader
then drops vectors from unpublished revisions, deleted documents, other
workspaces, other knowledge bases, and mismatched embedding models.
"""

from __future__ import annotations

import time
import uuid

from app.core.config import Settings
from app.core.errors import ConflictError, NotFoundError
from app.embeddings.service import EmbeddingService
from app.indexing.vector_store import VectorStore
from app.persistence.db import Database
from app.persistence.models import KnowledgeBase
from app.retrieval.chunks import ActiveChunkLoader
from app.retrieval.types import RetrievalOutcome


class DenseRetriever:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        embeddings: EmbeddingService,
        vectors: VectorStore,
    ) -> None:
        self.settings = settings
        self.database = database
        self.embeddings = embeddings
        self.vectors = vectors
        self.loader = ActiveChunkLoader(database, embeddings.model_id)

    async def _knowledge_base(self, workspace_id: uuid.UUID, kb_id: uuid.UUID) -> KnowledgeBase:
        async with self.database.sessionmaker() as session:
            kb = await session.get(KnowledgeBase, kb_id)
        if kb is None or kb.workspace_id != workspace_id or kb.deleted_at is not None:
            raise NotFoundError("Không tìm thấy knowledge base.")
        if kb.embedding_model != self.embeddings.model_id:
            raise ConflictError(
                "Knowledge base dùng embedding model khác cấu hình hiện tại.",
                code="embedding_model_mismatch",
                details={
                    "knowledge_base_model": kb.embedding_model,
                    "configured_model": self.embeddings.model_id,
                },
            )
        return kb

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
        await self._knowledge_base(workspace_id, kb_id)
        if not await self.vectors.exists():
            return RetrievalOutcome(
                chunks=[],
                timings={"dense_ms": round((time.perf_counter() - started) * 1000, 2)},
                candidates={"dense": 0},
                mode="basic",
                mode_used="dense",
                embedding_model=self.embeddings.model_id,
            )

        vector = await self.embeddings.embed_query(query)
        points = await self.vectors.search(
            workspace_id=workspace_id,
            kb_id=kb_id,
            vector=vector,
            limit=limit,
            score_threshold=self.settings.min_dense_score,
            document_ids=document_ids,
        )
        score_by_chunk: dict[uuid.UUID, float] = {}
        for point in points:
            payload = point.payload or {}
            raw_id = payload.get("chunk_id")
            if not raw_id:
                continue
            try:
                chunk_id = uuid.UUID(str(raw_id))
            except ValueError:
                continue
            score_by_chunk[chunk_id] = float(point.score)

        chunks = await self.loader.load(
            workspace_id=workspace_id,
            kb_id=kb_id,
            scores=score_by_chunk,
            source="dense",
            score_key="dense",
            document_ids=document_ids,
        )
        chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        return RetrievalOutcome(
            chunks=chunks,
            timings={"dense_ms": round((time.perf_counter() - started) * 1000, 2)},
            candidates={"dense": len(points)},
            mode="basic",
            mode_used="dense",
            embedding_model=self.embeddings.model_id,
        )
