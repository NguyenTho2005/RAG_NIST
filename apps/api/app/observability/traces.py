"""Trace persistence helpers for retrieval/chat observability."""

from __future__ import annotations

import uuid
from typing import Any

from app.observability.logging import request_id_ctx
from app.persistence.db import Database
from app.persistence.models import Trace


async def create_trace(
    database: Database,
    *,
    workspace_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    query: str,
    processed_query: str,
    mode: str,
    config: dict[str, Any],
    stages: list[dict[str, Any]] | None = None,
    candidates: dict[str, Any] | None = None,
    selected: list[dict[str, Any]] | None = None,
    timings: dict[str, Any] | None = None,
    token_usage: dict[str, Any] | None = None,
) -> uuid.UUID:
    async with database.sessionmaker() as session:
        trace = Trace(
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            request_id=request_id_ctx.get(),
            query=query,
            processed_query=processed_query,
            mode=mode,
            config=config,
            stages=stages or [],
            candidates=candidates or {},
            selected=selected or [],
            timings=timings or {},
            token_usage=token_usage or {},
        )
        session.add(trace)
        await session.commit()
        await session.refresh(trace)
        return trace.id


async def update_trace(
    database: Database,
    trace_id: uuid.UUID,
    *,
    stages: list[dict[str, Any]] | None = None,
    candidates: dict[str, Any] | None = None,
    selected: list[dict[str, Any]] | None = None,
    timings: dict[str, Any] | None = None,
    token_usage: dict[str, Any] | None = None,
) -> None:
    async with database.sessionmaker() as session:
        trace = await session.get(Trace, trace_id)
        if trace is None:
            return
        if stages is not None:
            trace.stages = stages
        if candidates is not None:
            trace.candidates = candidates
        if selected is not None:
            trace.selected = selected
        if timings is not None:
            trace.timings = timings
        if token_usage is not None:
            trace.token_usage = token_usage
        await session.commit()
