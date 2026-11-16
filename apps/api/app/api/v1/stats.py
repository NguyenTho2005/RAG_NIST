"""Workspace-level operational statistics backed by real tables."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import Float, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.models import (
    ChatMessage,
    ChatSession,
    Chunk,
    Document,
    Job,
    KnowledgeBase,
    Trace,
)
from app.security.auth import Principal, get_db, require_read

router = APIRouter()


@router.get("/stats/overview", summary="Real workspace counts and latency")
async def overview(
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    ws = principal.workspace_id
    since = datetime.now(UTC) - timedelta(hours=24)

    kb_count = (
        await db.execute(
            select(func.count(KnowledgeBase.id)).where(
                KnowledgeBase.workspace_id == ws, KnowledgeBase.deleted_at.is_(None)
            )
        )
    ).scalar_one()
    doc_rows = (
        await db.execute(
            select(Document.status, func.count(Document.id))
            .where(Document.workspace_id == ws, Document.deleted_at.is_(None))
            .group_by(Document.status)
        )
    ).all()
    docs_by_status = dict(doc_rows)
    chunk_count = (
        await db.execute(
            select(func.count(Chunk.id)).where(Chunk.workspace_id == ws, Chunk.active.is_(True))
        )
    ).scalar_one()
    job_rows = (
        await db.execute(
            select(Job.status, func.count(Job.id))
            .where(Job.workspace_id == ws)
            .group_by(Job.status)
        )
    ).all()
    jobs_by_status = dict(job_rows)
    queries_24h = (
        await db.execute(
            select(func.count(Trace.id)).where(Trace.workspace_id == ws, Trace.created_at >= since)
        )
    ).scalar_one()
    avg_latency: float | None = (
        await db.execute(
            select(func.avg(cast(Trace.timings["total_ms"].astext, Float))).where(
                Trace.workspace_id == ws,
                Trace.created_at >= since,
                Trace.timings["total_ms"].astext.is_not(None),
            )
        )
    ).scalar_one()
    messages = (
        await db.execute(
            select(func.count(ChatMessage.id))
            .join(ChatSession, ChatMessage.session_id == ChatSession.id)
            .where(ChatSession.workspace_id == ws, ChatMessage.role == "assistant")
        )
    ).scalar_one_or_none()

    recent_docs = (
        await db.execute(
            select(Document.id, Document.filename, Document.status, Document.updated_at)
            .where(Document.workspace_id == ws, Document.deleted_at.is_(None))
            .order_by(Document.updated_at.desc())
            .limit(6)
        )
    ).all()

    return {
        "knowledge_bases": kb_count,
        "documents": {
            "total": sum(docs_by_status.values()),
            "ready": docs_by_status.get("ready", 0),
            "processing": sum(
                docs_by_status.get(status, 0)
                for status in (
                    "uploaded",
                    "validating",
                    "parsing",
                    "chunking",
                    "embedding",
                    "indexing",
                    "reprocessing",
                )
            ),
            "failed": docs_by_status.get("failed", 0),
            "by_status": [
                {"status": status, "count": count}
                for status, count in sorted(docs_by_status.items())
            ],
        },
        "chunks": chunk_count,
        "jobs": {
            "queued": jobs_by_status.get("queued", 0),
            "running": jobs_by_status.get("running", 0),
            "succeeded": jobs_by_status.get("succeeded", 0),
            "failed": jobs_by_status.get("failed", 0),
        },
        "queries_24h": queries_24h,
        "answers_total": messages or 0,
        "avg_latency_ms_24h": round(float(avg_latency), 1) if avg_latency is not None else None,
        "recent_documents": [
            {
                "id": str(doc_id),
                "filename": filename,
                "status": status,
                "updated_at": updated_at.isoformat(),
            }
            for doc_id, filename, status, updated_at in recent_docs
        ],
    }
