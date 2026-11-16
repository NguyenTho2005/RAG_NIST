"""FastAPI application factory and lifespan wiring."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.errors import install_error_handlers
from app.core.middleware import RequestContextMiddleware
from app.generation.factory import build_llm_provider
from app.ingestion.runtime import IngestionRuntime
from app.observability.logging import configure_logging, get_logger
from app.persistence.db import Database
from app.reranking.factory import build_reranker
from app.security.ratelimit import InMemoryRateLimiter, RedisRateLimiter

logger = get_logger(__name__)

VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    configure_logging(settings.log_level, settings.log_format)
    settings.ensure_dirs()

    app.state.settings = settings
    app.state.database = Database(settings.database_url)
    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=True)
    app.state.qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        timeout=5,
    )
    app.state.rate_limiter = InMemoryRateLimiter(settings.auth_rate_limit_per_minute)
    try:
        if await asyncio.wait_for(app.state.redis.ping(), timeout=2):
            app.state.rate_limiter = RedisRateLimiter(
                app.state.redis, settings.auth_rate_limit_per_minute
            )
    except Exception:
        logger.warning(
            "redis_unavailable_rate_limit_fallback",
            extra={"event": "redis_unavailable_rate_limit_fallback"},
        )
    app.state.healthy = True
    app.state.llm_provider = build_llm_provider(settings)
    app.state.reranker = build_reranker(settings)
    app.state.ingestion = IngestionRuntime(
        settings=settings,
        database=app.state.database,
        qdrant=app.state.qdrant,
    )
    if settings.inline_worker:
        resumed = await app.state.ingestion.resume_queued()
        if resumed:
            logger.info(
                "queued_jobs_resumed",
                extra={"event": "queued_jobs_resumed", "count": resumed},
            )

    logger.info(
        "application_started",
        extra={"event": "application_started"},
    )
    try:
        yield
    finally:
        logger.info("application_stopping", extra={"event": "application_stopping"})
        await app.state.ingestion.shutdown()
        if app.state.llm_provider is not None:
            await app.state.llm_provider.aclose()
        if app.state.reranker is not None:
            await app.state.reranker.aclose()
        await app.state.qdrant.close()
        await app.state.redis.aclose()
        await app.state.database.dispose()
        logger.info("application_stopped", extra={"event": "application_stopped"})


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    app = FastAPI(
        title="Advanced RAG API",
        version=VERSION,
        description=(
            "Vietnamese-first multilingual RAG platform. Retrieved content is treated as "
            "untrusted data and every answer must cite indexed evidence."
        ),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = resolved

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    install_error_handlers(app)
    app.include_router(api_router, prefix=resolved.api_prefix)

    @app.get("/", include_in_schema=False)
    async def _root() -> dict[str, str]:
        return {"name": "Advanced RAG API", "version": VERSION, "docs": "/api/docs"}

    return app


app = create_app()


async def _redis_probe(redis: Redis) -> bool:
    try:
        return bool(await asyncio.wait_for(redis.ping(), timeout=2))
    except Exception:
        return False


async def startup_services(app: FastAPI) -> dict[str, Any]:
    """Expose probes for tests without leaking client objects."""
    return {
        "redis": await _redis_probe(app.state.redis),
    }
