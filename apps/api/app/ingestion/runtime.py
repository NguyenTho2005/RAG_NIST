"""Runtime wiring for ingestion dispatch (inline dev mode or arq worker)."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select

from app.core.config import Settings
from app.embeddings.service import EmbeddingService
from app.indexing.vector_store import VectorStore
from app.ingestion.pipeline import IngestionPipeline
from app.observability.logging import get_logger
from app.persistence.db import Database
from app.persistence.models import Job
from app.providers.factory import build_embedding_provider

logger = get_logger(__name__)


class IngestionRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        qdrant: AsyncQdrantClient,
        embeddings: EmbeddingService | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.embeddings = embeddings or EmbeddingService(
            build_embedding_provider(settings),
            cache_enabled=settings.embedding_cache_enabled,
            batch_size=settings.embedding_batch_size,
            max_attempts=settings.embedding_max_retries + 1,
        )
        self.vectors = VectorStore(
            qdrant,
            settings.qdrant_collection,
            settings.embedding_dimension,
            self.embeddings.model_id,
        )
        self._tasks: dict[uuid.UUID, asyncio.Task[None]] = {}
        self._arq_pool: ArqRedis | None = None

    @property
    def pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            settings=self.settings,
            database=self.database,
            embeddings=self.embeddings,
            vectors=self.vectors,
        )

    async def dispatch(self, job_id: uuid.UUID) -> None:
        if self.settings.inline_worker:
            existing = self._tasks.get(job_id)
            if existing is not None and not existing.done():
                return
            task = asyncio.create_task(self._run_inline(job_id), name=f"ingest-{job_id}")
            self._tasks[job_id] = task
            task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
            return

        if self._arq_pool is None:
            self._arq_pool = await create_pool(RedisSettings.from_dsn(self.settings.redis_url))
        await self._arq_pool.enqueue_job("process_job", str(job_id), _job_id=f"rag-job:{job_id}")

    async def _run_inline(self, job_id: uuid.UUID) -> None:
        try:
            await self.pipeline.run(job_id)
        except Exception:
            logger.exception(
                "inline_job_crashed",
                extra={"event": "inline_job_crashed", "job_id": str(job_id)},
            )

    async def resume_queued(self) -> int:
        async with self.database.sessionmaker() as session:
            rows = (
                (
                    await session.execute(
                        select(Job.id)
                        .where(Job.status == "queued")
                        .order_by(Job.created_at)
                        .limit(50)
                    )
                )
                .scalars()
                .all()
            )
        for job_id in rows:
            await self.dispatch(job_id)
        return len(rows)

    async def wait(self, job_id: uuid.UUID, timeout_seconds: float = 120.0) -> None:
        """Await an inline task; used by integration tests and CLI, not requests."""
        task = self._tasks.get(job_id)
        if task is not None:
            await asyncio.wait_for(task, timeout=timeout_seconds)

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=20)
            except TimeoutError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        if self._arq_pool is not None:
            await self._arq_pool.aclose()
            self._arq_pool = None
        await self.embeddings.aclose()

    async def enqueue_queued_to_redis(self) -> None:
        """Worker-side reconciliation: re-enqueue DB jobs that are still queued."""
        if self._arq_pool is None:
            self._arq_pool = await create_pool(RedisSettings.from_dsn(self.settings.redis_url))
        async with self.database.sessionmaker() as session:
            rows = (
                (
                    await session.execute(
                        select(Job.id)
                        .where(Job.status == "queued")
                        .order_by(Job.created_at)
                        .limit(50)
                    )
                )
                .scalars()
                .all()
            )
        for job_id in rows:
            await self._arq_pool.enqueue_job(
                "process_job", str(job_id), _job_id=f"rag-job:{job_id}"
            )
        logger.info(
            "queued_jobs_requeued",
            extra={"event": "queued_jobs_requeued", "count": len(rows)},
        )

    def stats(self) -> dict[str, Any]:
        return {
            "inline_worker": self.settings.inline_worker,
            "active_inline_tasks": len(self._tasks),
            "embedding_model": self.embeddings.model_id,
        }
