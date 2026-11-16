"""Dense retrieval/search endpoint and shared response models."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import app_database, retrieval_service
from app.api.v1.settings import workspace_retrieval_defaults
from app.core.errors import ValidationFailedError
from app.observability.traces import create_trace
from app.retrieval.context import ContextSource, source_metadata
from app.retrieval.service import RetrievalService
from app.security.auth import Principal, get_db, require_read

router = APIRouter()

RetrievalMode = Literal["basic", "hybrid", "auto", "graph_local", "graph_global"]


class SearchRequest(BaseModel):
    knowledge_base_id: uuid.UUID
    query: str = Field(min_length=1, max_length=4000)
    mode: RetrievalMode | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)


class SourceOut(BaseModel):
    marker: str
    source_index: int
    chunk_id: str
    document_id: str
    revision_id: str
    filename: str
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    score: float
    source: str


class SearchHit(SourceOut):
    text: str
    truncated: bool = False


class SearchResponse(BaseModel):
    query: str
    mode: str
    mode_used: str
    embedding_model: str
    results: list[SearchHit]
    trace_id: str
    timings: dict[str, float]
    candidates: dict[str, int]
    warnings: list[str] = Field(default_factory=list)


def source_out(source: ContextSource) -> SourceOut:
    return SourceOut(**source_metadata(source))


def search_hit(source: ContextSource) -> SearchHit:
    return SearchHit(
        **source_metadata(source),
        text=source.text,
        truncated=source.truncated,
    )


def trace_selected(sources: list[ContextSource]) -> list[dict[str, Any]]:
    return [source_metadata(source) for source in sources]


@router.post("/search", response_model=SearchResponse, summary="Dense document search")
async def search(
    request: Request,
    body: SearchRequest,
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
    retrieval: RetrievalService = Depends(retrieval_service),
) -> SearchResponse:
    query = body.query.strip()
    if not query:
        raise ValidationFailedError("Câu hỏi không được để trống.")
    default_mode, default_top_k = await workspace_retrieval_defaults(db, principal.workspace_id)
    mode = body.mode or default_mode
    top_k = body.top_k or default_top_k

    outcome, sources = await retrieval.search(
        workspace_id=principal.workspace_id,
        kb_id=body.knowledge_base_id,
        query=query,
        top_k=top_k,
        mode=mode,
        document_ids=body.document_ids,
    )
    trace_id = await create_trace(
        app_database(request),
        workspace_id=principal.workspace_id,
        knowledge_base_id=body.knowledge_base_id,
        query=query,
        processed_query=query,
        mode=outcome.mode,
        config=retrieval.trace_config(top_k=top_k, mode=mode),
        stages=[
            {
                "stage": "dense_retrieval",
                "status": "ok",
                "ms": outcome.timings.get("dense_ms", 0),
                "candidates": outcome.candidates.get("dense", 0),
            },
            {
                "stage": "context_assembly",
                "status": "ok",
                "selected": len(sources),
                "token_budget": retrieval.settings.context_token_budget,
            },
        ],
        candidates=outcome.candidates,
        selected=trace_selected(sources),
        timings=outcome.timings,
    )
    return SearchResponse(
        query=query,
        mode=outcome.mode,
        mode_used=outcome.mode_used,
        embedding_model=outcome.embedding_model,
        results=[search_hit(source) for source in sources],
        trace_id=str(trace_id),
        timings=outcome.timings,
        candidates=outcome.candidates,
        warnings=outcome.warnings,
    )
