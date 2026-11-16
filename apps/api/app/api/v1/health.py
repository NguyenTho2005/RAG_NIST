"""Liveness and dependency readiness endpoints."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel
from sqlalchemy import text

from app.core.config import get_settings

router = APIRouter()

ComponentStatus = Literal["ok", "degraded", "down", "not_configured"]


class Component(BaseModel):
    name: str
    status: ComponentStatus
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    environment: str
    timestamp: datetime


class ReadyResponse(BaseModel):
    ready: bool
    status: Literal["ready", "degraded", "not_ready"]
    components: list[Component]
    capabilities: dict[str, bool]
    checked_at: datetime


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.environment,
        timestamp=datetime.now(UTC),
    )


async def _check_database(request: Request) -> Component:
    started = time.perf_counter()
    try:
        async with request.app.state.database.sessionmaker() as session:
            await session.execute(text("SELECT 1"))
        return Component(
            name="postgres",
            status="ok",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    except Exception as exc:
        return Component(
            name="postgres",
            status="down",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            detail=type(exc).__name__,
        )


async def _check_redis(request: Request) -> Component:
    started = time.perf_counter()
    try:
        await request.app.state.redis.ping()
        return Component(
            name="redis", status="ok", latency_ms=round((time.perf_counter() - started) * 1000, 2)
        )
    except Exception as exc:
        return Component(
            name="redis",
            status="down",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            detail=type(exc).__name__,
        )


async def _check_qdrant(request: Request) -> Component:
    started = time.perf_counter()
    try:
        await request.app.state.qdrant.get_collections()
        return Component(
            name="qdrant", status="ok", latency_ms=round((time.perf_counter() - started) * 1000, 2)
        )
    except Exception as exc:
        return Component(
            name="qdrant",
            status="down",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            detail=type(exc).__name__,
        )


def _check_llm() -> Component:
    settings = get_settings()
    if not settings.llm_configured:
        return Component(
            name="llm",
            status="not_configured",
            detail="LLM_PROVIDER=none; chat and graph extraction are unavailable.",
        )
    return Component(
        name="llm",
        status="degraded",
        detail=f"configured:{settings.llm_provider}:{settings.llm_model}; verified on first call",
    )


def _check_embeddings() -> Component:
    settings = get_settings()
    return Component(
        name="embeddings",
        status="degraded",
        detail=f"configured:{settings.embedding_provider}:{settings.embedding_model}",
    )


@router.get("/ready", response_model=ReadyResponse, summary="Dependency readiness probe")
async def ready(request: Request, response: Response) -> ReadyResponse:
    components = [
        await _check_database(request),
        await _check_qdrant(request),
        await _check_redis(request),
        _check_embeddings(),
        _check_llm(),
    ]
    by_name = {component.name: component for component in components}
    ready_flag = by_name["postgres"].status == "ok" and by_name["qdrant"].status == "ok"
    degraded = any(component.status in {"down", "degraded"} for component in components)
    if not ready_flag:
        response.status_code = 503
    status: Literal["ready", "degraded", "not_ready"] = (
        "not_ready" if not ready_flag else ("degraded" if degraded else "ready")
    )
    settings = get_settings()
    return ReadyResponse(
        ready=ready_flag,
        status=status,
        components=components,
        capabilities={
            "retrieval": ready_flag,
            "indexing": ready_flag,
            # Configured is not the same as available. Generation is only
            # advertised after a provider probe succeeds (see /diagnostics).
            "generation": False,
            "generation_configured": settings.llm_configured,
            "graph": ready_flag and settings.graph_enabled,
        },
        checked_at=datetime.now(UTC),
    )


@router.get("/diagnostics", summary="Detailed runtime diagnostics")
async def diagnostics(request: Request) -> dict[str, Any]:
    settings = get_settings()
    checks = [
        await _check_database(request),
        await _check_qdrant(request),
        await _check_redis(request),
        _check_embeddings(),
        _check_llm(),
    ]
    return {
        "version": "0.1.0",
        "environment": settings.environment,
        "components": [component.model_dump() for component in checks],
        "config": {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "reranker_provider": settings.reranker_provider,
            "default_mode": settings.retrieval_mode,
            "graph_enabled": settings.graph_enabled,
            "inline_worker": settings.inline_worker,
        },
    }
