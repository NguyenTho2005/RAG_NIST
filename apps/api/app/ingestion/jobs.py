"""Job lifecycle helpers: deduplication, events, cancellation, claiming."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import JobStage, JobStatus, JobType
from app.persistence.db import Database
from app.persistence.models import Job, JobEvent

ACTIVE_JOB_STATUSES = (JobStatus.QUEUED.value, JobStatus.RUNNING.value)


async def find_active_job(
    session: AsyncSession,
    *,
    document_id: uuid.UUID | None,
    job_type: str,
) -> Job | None:
    if document_id is None:
        return None
    return (
        await session.execute(
            select(Job)
            .where(
                Job.document_id == document_id,
                Job.job_type == job_type,
                Job.status.in_(ACTIVE_JOB_STATUSES),
            )
            .order_by(Job.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


def new_job(
    *,
    workspace_id: uuid.UUID,
    job_type: str,
    knowledge_base_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    revision_id: uuid.UUID | None = None,
    payload: dict[str, Any] | None = None,
    max_attempts: int = 3,
) -> Job:
    return Job(
        workspace_id=workspace_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        revision_id=revision_id,
        job_type=job_type,
        status=JobStatus.QUEUED.value,
        stage=JobStage.QUEUED.value,
        progress=0,
        attempts=0,
        max_attempts=max_attempts,
        payload=payload or {},
    )


def add_event(
    session: AsyncSession,
    job_id: uuid.UUID,
    stage: str,
    message: str,
    *,
    level: str = "info",
    data: dict[str, Any] | None = None,
) -> None:
    session.add(
        JobEvent(
            job_id=job_id,
            stage=stage,
            level=level,
            message=message,
            data=data or {},
        )
    )


async def claim_job(database: Database, job_id: uuid.UUID) -> Job | None:
    """Atomically move a queued job to running. Returns None if already claimed."""
    async with database.sessionmaker() as session:
        result = await session.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == JobStatus.QUEUED.value)
            .values(
                status=JobStatus.RUNNING.value,
                attempts=Job.attempts + 1,
                started_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            .returning(Job)
        )
        job: Job | None = result.scalar_one_or_none()
        await session.commit()
        if job is None:
            return None
        await session.refresh(job)
        return job


async def cancel_job(
    session: AsyncSession, job_id: uuid.UUID, workspace_id: uuid.UUID
) -> Job | None:
    """Record cancellation in the caller's session and commit."""
    job = (
        await session.execute(select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id))
    ).scalar_one_or_none()
    if job is None:
        return None
    if job.status in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}:
        job.cancel_requested = True
        if job.status == JobStatus.QUEUED.value:
            job.status = JobStatus.CANCELLED.value
            job.stage = JobStage.DONE.value
            job.finished_at = datetime.now(UTC)
            job.progress = 100
            if job.document_id:
                from app.persistence.models import Document

                document = await session.get(Document, job.document_id)
                if document is not None and document.deleted_at is None:
                    document.status = "cancelled"
        add_event(session, job.id, job.stage, "Yêu cầu hủy đã được ghi nhận.")
        await session.commit()
        await session.refresh(job)
    return job


async def retry_job(
    session: AsyncSession, job_id: uuid.UUID, workspace_id: uuid.UUID
) -> Job | None:
    """Create a fresh queued job for the same document/revision after failure."""
    previous = (
        await session.execute(select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id))
    ).scalar_one_or_none()
    if previous is None:
        return None
    active = await find_active_job(
        session, document_id=previous.document_id, job_type=previous.job_type
    )
    if active is not None:
        return active
    job = new_job(
        workspace_id=previous.workspace_id,
        knowledge_base_id=previous.knowledge_base_id,
        document_id=previous.document_id,
        revision_id=previous.revision_id,
        job_type=previous.job_type,
        payload=previous.payload,
        max_attempts=previous.max_attempts,
    )
    session.add(job)
    await session.flush()
    add_event(session, job.id, JobStage.QUEUED.value, "Job được tạo lại từ lần chạy trước.")
    await session.commit()
    await session.refresh(job)
    return job


__all__ = [
    "JobType",
    "add_event",
    "cancel_job",
    "claim_job",
    "find_active_job",
    "new_job",
    "retry_job",
]
