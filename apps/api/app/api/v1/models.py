"""Model capability inspection with real, non-downloading probes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.security.auth import Principal, require_read

router = APIRouter()


class ModelInfo(BaseModel):
    role: str
    provider: str
    model_id: str
    configured: bool
    available: bool
    dimension: int | None = None
    detail: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo] = Field(default_factory=list)
    capabilities: dict[str, bool] = Field(default_factory=dict)


@router.get("/models", response_model=ModelsResponse, summary="Configured models and probes")
async def models(request: Request, principal: Principal = Depends(require_read)) -> ModelsResponse:
    settings = request.app.state.settings
    runtime = request.app.state.ingestion
    items: list[ModelInfo] = []

    embedding_available = False
    try:
        embedding_available = bool(await runtime.embeddings.ping())
    except Exception:
        embedding_available = False
    items.append(
        ModelInfo(
            role="embedding",
            provider=settings.embedding_provider,
            model_id=runtime.embeddings.model_id,
            configured=True,
            available=embedding_available,
            dimension=runtime.embeddings.dimension,
            detail=(
                "Local weights load on first real request; readiness does not download."
                if settings.embedding_provider == "local"
                else None
            ),
        )
    )

    llm = getattr(request.app.state, "llm_provider", None)
    llm_available = False
    if llm is not None:
        try:
            llm_available = bool(await llm.ping())
        except Exception:
            llm_available = False
    items.append(
        ModelInfo(
            role="generation",
            provider=settings.llm_provider,
            model_id=settings.llm_model,
            configured=llm is not None,
            available=llm_available,
            detail=None
            if llm is not None
            else "LLM_PROVIDER=none; generation is unavailable and chat returns 503.",
        )
    )

    reranker = getattr(request.app.state, "reranker", None)
    reranker_available = False
    if reranker is not None:
        try:
            reranker_available = bool(await reranker.ping())
        except Exception:
            reranker_available = False
    items.append(
        ModelInfo(
            role="reranker",
            provider=settings.reranker_provider,
            model_id=settings.reranker_model,
            configured=reranker is not None,
            available=reranker_available,
            detail=(
                "Disabled by configuration."
                if reranker is None
                else "Weights are probed only after first load."
            ),
        )
    )

    capabilities: dict[str, Any] = {
        "retrieval": True,
        "dense": True,
        "sparse": True,
        "hybrid": True,
        "rerank": reranker is not None and settings.rerank_enabled,
        "generation": llm_available,
        "generation_configured": llm is not None,
        "graph": bool(settings.graph_enabled),
    }
    return ModelsResponse(models=items, capabilities=capabilities)
