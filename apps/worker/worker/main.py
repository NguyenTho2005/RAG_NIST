"""arq worker entrypoint.

Run with: PYTHONPATH=apps/api:apps/worker python -m worker.main
"""

from __future__ import annotations

import uuid
from typing import Any

from arq.connections import RedisSettings
from arq.typing import WorkerSettingsBase
from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.embeddings.service import EmbeddingService
from app.indexing.vector_store import VectorStore
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.runtime import IngestionRuntime
from app.observability.logging import configure_logging, get_logger
from app.persistence.db import Database
from app.providers.factory import build_embedding_provider

logger = get_logger(__name__)


async def process_job(ctx: dict[str, Any], job_id: str) -> None:
    pipeline: IngestionPipeline = ctx["pipeline"]
    await pipeline.run(uuid.UUID(job_id))


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    database = Database(settings.database_url)
    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        timeout=30,
    )
    embeddings = EmbeddingService(
        build_embedding_provider(settings),
        cache_enabled=settings.embedding_cache_enabled,
        batch_size=settings.embedding_batch_size,
        max_attempts=settings.embedding_max_retries + 1,
    )
    vectors = VectorStore(
        qdrant,
        settings.qdrant_collection,
        settings.embedding_dimension,
        embeddings.model_id,
    )
    runtime = IngestionRuntime(
        settings=settings,
        database=database,
        qdrant=qdrant,
        embeddings=embeddings,
    )
    ctx.update(
        {
            "settings": settings,
            "database": database,
            "qdrant": qdrant,
            "pipeline": IngestionPipeline(
                settings=settings,
                database=database,
                embeddings=embeddings,
                vectors=vectors,
            ),
            "runtime": runtime,
        }
    )
    await runtime.enqueue_queued_to_redis()
    logger.info("worker_started", extra={"event": "worker_started"})


async def shutdown(ctx: dict[str, Any]) -> None:
    runtime: IngestionRuntime = ctx["runtime"]
    await runtime.shutdown()
    await ctx["qdrant"].close()
    await ctx["database"].dispose()
    logger.info("worker_stopped", extra={"event": "worker_stopped"})


_SETTINGS = get_settings()


class WorkerSettings(WorkerSettingsBase):
    functions = [process_job]
    on_startup = staticmethod(startup)
    on_shutdown = staticmethod(shutdown)
    max_jobs = _SETTINGS.max_concurrent_jobs
    job_timeout = 1800
    keep_result = 3600
    retry_jobs = False
    health_check_interval = 30
    redis_settings = RedisSettings.from_dsn(_SETTINGS.redis_url)


def main() -> None:
    from arq import run_worker

    run_worker(WorkerSettings)


if __name__ == "__main__":
    main()
