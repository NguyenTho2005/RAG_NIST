"""Retrieval/generation trace inspection endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import Page
from app.core.errors import NotFoundError
from app.persistence.models import Trace
from app.security.auth import Principal, get_db, require_read

router = APIRouter()


class TraceOut(BaseModel):
    id: str
    request_id: str | None
    knowledge_base_id: str
    query: str
    processed_query: str
    mode: str
    config: dict[str, Any] = Field(default_factory=dict)
    stages: list[dict[str, Any]] = Field(default_factory=list)
    candidates: dict[str, Any] = Field(default_factory=dict)
    selected: list[dict[str, Any]] = Field(default_factory=list)
    timings: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class TraceSummary(BaseModel):
    id: str
    query: str
    mode: str
    created_at: datetime


def _trace_out(trace: Trace) -> TraceOut:
    return TraceOut(
        id=str(trace.id),
        request_id=trace.request_id,
        knowledge_base_id=str(trace.knowledge_base_id),
        query=trace.query,
        processed_query=trace.processed_query,
        mode=trace.mode,
        config=dict(trace.config or {}),
        stages=list(trace.stages or []),
        candidates=dict(trace.candidates or {}),
        selected=list(trace.selected or []),
        timings=dict(trace.timings or {}),
        token_usage=dict(trace.token_usage or {}),
        created_at=trace.created_at,
    )


@router.get("/traces", response_model=Page[TraceSummary], summary="List retrieval traces")
async def list_traces(
    knowledge_base_id: uuid.UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> Page[TraceSummary]:
    conditions = [Trace.workspace_id == principal.workspace_id]
    if knowledge_base_id:
        conditions.append(Trace.knowledge_base_id == knowledge_base_id)
    total = (await db.execute(select(func.count(Trace.id)).where(*conditions))).scalar_one()
    rows = (
        (
            await db.execute(
                select(Trace)
                .where(*conditions)
                .order_by(Trace.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )
    return Page(
        items=[
            TraceSummary(
                id=str(trace.id),
                query=trace.query,
                mode=trace.mode,
                created_at=trace.created_at,
            )
            for trace in rows
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/traces/{trace_id}", response_model=TraceOut, summary="Get retrieval trace detail")
async def get_trace(
    trace_id: uuid.UUID,
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> TraceOut:
    trace = await db.get(Trace, trace_id)
    if trace is None or trace.workspace_id != principal.workspace_id:
        raise NotFoundError("Không tìm thấy trace.")
    return _trace_out(trace)
