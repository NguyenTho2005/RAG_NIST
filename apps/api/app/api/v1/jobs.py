"""Job inspection, cancellation, and retry endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import Page
from app.core.errors import ConflictError, NotFoundError
from app.domain.enums import JobStatus
from app.ingestion.jobs import cancel_job, retry_job
from app.persistence.models import Document, Job, JobEvent
from app.security.auth import Principal, get_db, require_read, require_write

router = APIRouter()


class JobOut(BaseModel):
    id: str
    job_type: str
    status: str
    stage: str
    progress: int
    attempts: int
    max_attempts: int
    error_code: str | None
    error_message: str | None
    document_id: str | None
    filename: str | None = None
    knowledge_base_id: str | None
    cancel_requested: bool
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobEventOut(BaseModel):
    id: int
    stage: str
    level: str
    message: str
    data: dict[str, Any]
    created_at: datetime


class JobDetail(JobOut):
    events: list[JobEventOut] = Field(default_factory=list)


def _job_out(job: Job, filename: str | None = None) -> JobOut:
    return JobOut(
        id=str(job.id),
        job_type=job.job_type,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        error_code=job.error_code,
        error_message=job.error_message,
        document_id=str(job.document_id) if job.document_id else None,
        filename=filename,
        knowledge_base_id=str(job.knowledge_base_id) if job.knowledge_base_id else None,
        cancel_requested=job.cancel_requested,
        result=job.result or {},
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/jobs", response_model=Page[JobOut])
async def list_jobs(
    status: str | None = Query(default=None, max_length=20),
    document_id: uuid.UUID | None = None,
    knowledge_base_id: uuid.UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> Page[JobOut]:
    conditions = [Job.workspace_id == principal.workspace_id]
    if status:
        conditions.append(Job.status == status)
    if document_id:
        conditions.append(Job.document_id == document_id)
    if knowledge_base_id:
        conditions.append(Job.knowledge_base_id == knowledge_base_id)
    total = (await db.execute(select(func.count(Job.id)).where(*conditions))).scalar_one()
    rows = (
        await db.execute(
            select(Job, Document.filename)
            .outerjoin(Document, Job.document_id == Document.id)
            .where(*conditions)
            .order_by(Job.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return Page(
        items=[_job_out(job, filename) for job, filename in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/jobs/{job_id}", response_model=JobDetail)
async def get_job(
    job_id: uuid.UUID,
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> JobDetail:
    row = (
        await db.execute(
            select(Job, Document.filename)
            .outerjoin(Document, Job.document_id == Document.id)
            .where(Job.id == job_id, Job.workspace_id == principal.workspace_id)
        )
    ).one_or_none()
    if row is None:
        raise NotFoundError("Không tìm thấy job.")
    job, filename = row
    events = (
        (
            await db.execute(
                select(JobEvent).where(JobEvent.job_id == job.id).order_by(JobEvent.id).limit(500)
            )
        )
        .scalars()
        .all()
    )
    base = _job_out(job, filename)
    return JobDetail(
        **base.model_dump(),
        events=[
            JobEventOut(
                id=event.id,
                stage=event.stage,
                level=event.level,
                message=event.message,
                data=event.data or {},
                created_at=event.created_at,
            )
            for event in events
        ],
    )


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
async def cancel(
    job_id: uuid.UUID,
    principal: Principal = Depends(require_write),
    db: AsyncSession = Depends(get_db),
) -> JobOut:
    job = await cancel_job(db, job_id, principal.workspace_id)
    if job is None:
        raise NotFoundError("Không tìm thấy job.")
    if job.status not in {
        JobStatus.QUEUED.value,
        JobStatus.RUNNING.value,
        JobStatus.CANCELLED.value,
    }:
        raise ConflictError("Job đã kết thúc và không thể hủy.", code="job_not_cancellable")
    return _job_out(job)


@router.post("/jobs/{job_id}/retry", response_model=JobOut, status_code=202)
async def retry(
    request: Request,
    job_id: uuid.UUID,
    principal: Principal = Depends(require_write),
    db: AsyncSession = Depends(get_db),
) -> JobOut:
    original = (
        await db.execute(
            select(Job).where(Job.id == job_id, Job.workspace_id == principal.workspace_id)
        )
    ).scalar_one_or_none()
    if original is None:
        raise NotFoundError("Không tìm thấy job.")
    if original.status not in {JobStatus.FAILED.value, JobStatus.CANCELLED.value}:
        raise ConflictError(
            "Chỉ có thể retry job đã thất bại hoặc đã hủy.",
            code="job_not_retryable",
        )
    new = await retry_job(db, job_id, principal.workspace_id)
    if new is None:
        raise NotFoundError("Không tìm thấy job.")
    if new.status == JobStatus.QUEUED.value:
        await request.app.state.ingestion.dispatch(new.id)
    return _job_out(new)
