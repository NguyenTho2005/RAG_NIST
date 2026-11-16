"""Document upload, URL ingestion, listing, deletion, and reprocessing."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import Page
from app.chunking.chunker import CHUNKER_VERSION
from app.core.config import Settings, get_settings
from app.core.errors import (
    ConflictError,
    IngestionError,
    NotFoundError,
    PayloadTooLargeError,
    ValidationFailedError,
)
from app.domain.enums import DocumentStatus, JobStage, JobStatus
from app.ingestion.jobs import add_event, find_active_job, new_job
from app.ingestion.storage import (
    delete_file,
    relative_storage_path,
    resolve_path,
    write_atomic,
)
from app.observability.logging import get_logger
from app.parsing.detect import DetectedType, detect_file_type, safe_filename
from app.parsing.parsers import PARSER_VERSION
from app.persistence.models import (
    Chunk,
    Document,
    DocumentRevision,
    Job,
    KnowledgeBase,
)
from app.security.auth import Principal, get_db, require_read, require_write
from app.security.url_fetch import fetch_url

logger = get_logger(__name__)
router = APIRouter()


class DocumentOut(BaseModel):
    id: str
    knowledge_base_id: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    source_type: str
    source_uri: str | None
    status: str
    error_code: str | None
    error_message: str | None
    page_count: int | None = None
    chunk_count: int = 0
    active_revision_id: str | None
    active_revision_number: int | None = None
    created_at: datetime
    updated_at: datetime


class JobSummary(BaseModel):
    id: str
    status: str
    stage: str
    progress: int
    error_code: str | None
    error_message: str | None
    created_at: datetime


class RevisionOut(BaseModel):
    id: str
    revision_number: int
    status: str
    sha256: str
    page_count: int | None
    section_count: int
    chunk_count: int
    error_code: str | None
    error_message: str | None
    created_at: datetime
    ready_at: datetime | None


class DocumentDetail(DocumentOut):
    revisions: list[RevisionOut] = Field(default_factory=list)
    latest_job: JobSummary | None = None


class UploadResponse(BaseModel):
    document: DocumentOut
    job_id: str


class UrlIngestRequest(BaseModel):
    knowledge_base_id: uuid.UUID
    url: HttpUrl
    title: str | None = Field(default=None, max_length=300)


async def _require_kb(db: AsyncSession, principal: Principal, kb_id: uuid.UUID) -> KnowledgeBase:
    kb = (
        await db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.workspace_id == principal.workspace_id,
                KnowledgeBase.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if kb is None:
        raise NotFoundError("Không tìm thấy knowledge base.")
    return kb


def _to_out(
    document: Document, chunk_count: int = 0, revision_number: int | None = None
) -> DocumentOut:
    return DocumentOut(
        id=str(document.id),
        knowledge_base_id=str(document.knowledge_base_id),
        filename=document.filename,
        mime_type=document.mime_type,
        size_bytes=document.size_bytes,
        sha256=document.sha256,
        source_type=document.source_type,
        source_uri=document.source_uri,
        status=document.status,
        error_code=document.error_code,
        error_message=document.error_message,
        page_count=document.metadata_.get("page_count"),
        chunk_count=chunk_count,
        active_revision_id=str(document.active_revision_id)
        if document.active_revision_id
        else None,
        active_revision_number=revision_number,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


async def _latest_chunk_counts(
    db: AsyncSession, document_ids: list[uuid.UUID]
) -> dict[uuid.UUID, int]:
    if not document_ids:
        return {}
    rows = (
        await db.execute(
            select(Chunk.document_id, func.count(Chunk.id))
            .where(
                Chunk.document_id.in_(document_ids),
                Chunk.active.is_(True),
                Chunk.kind == "child",
            )
            .group_by(Chunk.document_id)
        )
    ).all()
    return dict(rows)


async def _create_document(
    *,
    request: Request,
    session: AsyncSession,
    principal: Principal,
    settings: Settings,
    kb: KnowledgeBase,
    content: bytes,
    filename: str,
    declared_mime: str | None,
    source_type: str,
    source_uri: str | None,
    detected: DetectedType | None = None,
) -> tuple[Document, Job]:
    if not content:
        raise ValidationFailedError("Tệp rỗng.")
    if len(content) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            "Tệp vượt quá giới hạn cho phép.",
            details={"max_bytes": settings.max_upload_bytes},
        )
    filename = safe_filename(filename)
    detected = detected or detect_file_type(filename, content, declared_mime)
    sha256 = hashlib.sha256(content).hexdigest()
    # Serialize concurrent uploads of identical content within this workspace.
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:lock_key))"),
        {"lock_key": f"{principal.workspace_id}:{kb.id}:{sha256}"},
    )

    duplicate = (
        await session.execute(
            select(Document).where(
                Document.workspace_id == principal.workspace_id,
                Document.knowledge_base_id == kb.id,
                Document.sha256 == sha256,
                Document.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if duplicate is not None:
        raise ConflictError(
            "Tài liệu này đã tồn tại trong knowledge base (trùng SHA-256).",
            code="duplicate_document",
            details={
                "document_id": str(duplicate.id),
                "status": duplicate.status,
                "filename": duplicate.filename,
            },
        )

    document_id = uuid.uuid4()
    relative_path = relative_storage_path(principal.workspace_id, document_id, sha256, filename)
    write_atomic(settings, relative_path, content)

    document = Document(
        id=document_id,
        workspace_id=principal.workspace_id,
        knowledge_base_id=kb.id,
        filename=filename,
        source_type=source_type,
        source_uri=source_uri,
        mime_type=detected.mime_type,
        size_bytes=len(content),
        sha256=sha256,
        status=DocumentStatus.UPLOADED.value,
        metadata_={
            "storage_path": relative_path,
            "detected_kind": detected.kind,
            "declared_mime": declared_mime,
        },
    )
    revision = DocumentRevision(
        document_id=document_id,
        workspace_id=principal.workspace_id,
        revision_number=1,
        sha256=sha256,
        status="pending",
        parser_version=PARSER_VERSION,
        chunker_version=CHUNKER_VERSION,
        ingestion_version=CHUNKER_VERSION,
    )
    session.add(document)
    session.add(revision)
    await session.flush()
    document.latest_revision_id = revision.id
    job = new_job(
        workspace_id=principal.workspace_id,
        knowledge_base_id=kb.id,
        document_id=document_id,
        revision_id=revision.id,
        job_type="ingest",
        payload={"filename": filename, "source_type": source_type},
        max_attempts=settings.job_max_attempts,
    )
    session.add(job)
    await session.flush()
    add_event(session, job.id, JobStage.QUEUED.value, "Đã nhận tài liệu và xếp hàng xử lý.")
    await session.commit()
    await session.refresh(document)
    await session.refresh(job)

    runtime = request.app.state.ingestion
    try:
        await runtime.dispatch(job.id)
    except Exception as exc:
        job.status = JobStatus.QUEUED.value
        job.error_code = "dispatch_failed"
        job.error_message = "Không thể gửi job tới worker."
        await session.commit()
        raise IngestionError(
            "Tài liệu đã lưu nhưng không thể gửi job tới worker. "
            "Kiểm tra Redis/worker rồi retry job.",
            code="job_dispatch_failed",
            status_code=503,
            details={"job_id": str(job.id), "document_id": str(document.id)},
        ) from exc
    return document, job


@router.post("/documents", response_model=UploadResponse, status_code=202)
async def upload_document(
    request: Request,
    knowledge_base_id: uuid.UUID = Form(...),
    file: UploadFile = File(...),
    principal: Principal = Depends(require_write),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    kb = await _require_kb(db, principal, knowledge_base_id)
    content = bytearray()
    while chunk := await file.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > settings.max_upload_bytes:
            raise PayloadTooLargeError(
                "Tệp vượt quá giới hạn cho phép.",
                details={"max_bytes": settings.max_upload_bytes},
            )
    document, job = await _create_document(
        request=request,
        session=db,
        principal=principal,
        settings=settings,
        kb=kb,
        content=bytes(content),
        filename=file.filename or "document",
        declared_mime=file.content_type,
        source_type="upload",
        source_uri=None,
    )
    return UploadResponse(document=_to_out(document), job_id=str(job.id))


@router.post("/documents/url", response_model=UploadResponse, status_code=202)
async def ingest_url(
    request: Request,
    payload: UrlIngestRequest,
    principal: Principal = Depends(require_write),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    kb = await _require_kb(db, principal, payload.knowledge_base_id)
    page = await fetch_url(str(payload.url), settings)
    filename = page.filename
    suffix = Path(filename).suffix.lower()
    if suffix not in {".html", ".htm", ".txt", ".md", ".markdown"}:
        filename = "page.html" if "html" in page.content_type else "page.txt"
    if payload.title:
        filename = (
            f"{safe_filename(payload.title)}.html" if "html" in page.content_type else filename
        )
    try:
        detected = detect_file_type(filename, page.content, page.content_type)
    except Exception as exc:
        raise IngestionError(
            "Nội dung URL không phải định dạng văn bản được hỗ trợ.",
            code="url_content_not_supported",
            status_code=422,
        ) from exc
    document, job = await _create_document(
        request=request,
        session=db,
        principal=principal,
        settings=settings,
        kb=kb,
        content=page.content,
        filename=filename,
        declared_mime=page.content_type,
        source_type="url",
        source_uri=page.final_url,
        detected=detected,
    )
    return UploadResponse(document=_to_out(document), job_id=str(job.id))


@router.get("/documents", response_model=Page[DocumentOut])
async def list_documents(
    knowledge_base_id: uuid.UUID | None = None,
    status: str | None = Query(default=None, max_length=30),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> Page[DocumentOut]:
    conditions = [
        Document.workspace_id == principal.workspace_id,
        Document.deleted_at.is_(None),
    ]
    if knowledge_base_id is not None:
        conditions.append(Document.knowledge_base_id == knowledge_base_id)
    if status:
        conditions.append(Document.status == status)
    total = (await db.execute(select(func.count(Document.id)).where(*conditions))).scalar_one()
    documents = (
        (
            await db.execute(
                select(Document)
                .where(*conditions)
                .order_by(Document.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )
    counts = await _latest_chunk_counts(db, [document.id for document in documents])
    revision_ids = {
        document.active_revision_id for document in documents if document.active_revision_id
    }
    revisions = {}
    if revision_ids:
        rows = (
            await db.execute(
                select(DocumentRevision.id, DocumentRevision.revision_number).where(
                    DocumentRevision.id.in_(revision_ids)
                )
            )
        ).all()
        revisions = dict(rows)
    return Page(
        items=[
            _to_out(
                document,
                counts.get(document.id, 0),
                revisions.get(document.active_revision_id) if document.active_revision_id else None,
            )
            for document in documents
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: uuid.UUID,
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> DocumentDetail:
    document = (
        await db.execute(
            select(Document).where(
                Document.id == document_id,
                Document.workspace_id == principal.workspace_id,
                Document.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if document is None:
        raise NotFoundError("Không tìm thấy tài liệu.")
    counts = await _latest_chunk_counts(db, [document.id])
    revision_number = None
    if document.active_revision_id:
        active = await db.get(DocumentRevision, document.active_revision_id)
        revision_number = active.revision_number if active else None
    revisions = (
        (
            await db.execute(
                select(DocumentRevision)
                .where(DocumentRevision.document_id == document.id)
                .order_by(DocumentRevision.revision_number.desc())
            )
        )
        .scalars()
        .all()
    )
    latest_job = (
        await db.execute(
            select(Job)
            .where(Job.document_id == document.id)
            .order_by(Job.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    base = _to_out(document, counts.get(document.id, 0), revision_number)
    return DocumentDetail(
        **base.model_dump(),
        revisions=[
            RevisionOut(
                id=str(revision.id),
                revision_number=revision.revision_number,
                status=revision.status,
                sha256=revision.sha256,
                page_count=revision.page_count,
                section_count=revision.section_count,
                chunk_count=revision.chunk_count,
                error_code=revision.error_code,
                error_message=revision.error_message,
                created_at=revision.created_at,
                ready_at=revision.ready_at,
            )
            for revision in revisions
        ],
        latest_job=JobSummary(
            id=str(latest_job.id),
            status=latest_job.status,
            stage=latest_job.stage,
            progress=latest_job.progress,
            error_code=latest_job.error_code,
            error_message=latest_job.error_message,
            created_at=latest_job.created_at,
        )
        if latest_job
        else None,
    )


@router.post("/documents/{document_id}/reprocess", response_model=UploadResponse, status_code=202)
async def reprocess_document(
    request: Request,
    document_id: uuid.UUID,
    principal: Principal = Depends(require_write),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    document = (
        await db.execute(
            select(Document).where(
                Document.id == document_id,
                Document.workspace_id == principal.workspace_id,
                Document.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if document is None:
        raise NotFoundError("Không tìm thấy tài liệu.")
    active = await find_active_job(db, document_id=document.id, job_type="ingest")
    if active is not None:
        raise ConflictError(
            "Tài liệu đang có job xử lý.",
            code="job_already_active",
            details={"job_id": str(active.id)},
        )
    storage_path = document.metadata_.get("storage_path")
    if not storage_path:
        raise IngestionError("Thiếu tệp nguồn để xử lý lại.", code="source_file_missing")
    path = resolve_path(settings, storage_path)
    if not path.is_file():
        raise IngestionError("Tệp nguồn không còn tồn tại.", code="source_file_missing")

    latest_number = (
        await db.execute(
            select(func.max(DocumentRevision.revision_number)).where(
                DocumentRevision.document_id == document.id
            )
        )
    ).scalar_one() or 0
    revision = DocumentRevision(
        document_id=document.id,
        workspace_id=principal.workspace_id,
        revision_number=latest_number + 1,
        sha256=document.sha256,
        status="pending",
        parser_version=PARSER_VERSION,
        chunker_version=CHUNKER_VERSION,
        ingestion_version=CHUNKER_VERSION,
    )
    db.add(revision)
    await db.flush()
    document.latest_revision_id = revision.id
    document.status = DocumentStatus.REPROCESSING.value
    document.error_code = None
    document.error_message = None
    job = new_job(
        workspace_id=principal.workspace_id,
        knowledge_base_id=document.knowledge_base_id,
        document_id=document.id,
        revision_id=revision.id,
        job_type="reprocess",
        payload={"filename": document.filename},
        max_attempts=settings.job_max_attempts,
    )
    db.add(job)
    await db.flush()
    add_event(db, job.id, JobStage.QUEUED.value, "Xếp hàng xử lý lại tài liệu.")
    await db.commit()
    await db.refresh(document)
    await db.refresh(job)
    await request.app.state.ingestion.dispatch(job.id)
    return UploadResponse(document=_to_out(document), job_id=str(job.id))


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    request: Request,
    document_id: uuid.UUID,
    principal: Principal = Depends(require_write),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> None:
    document = (
        await db.execute(
            select(Document).where(
                Document.id == document_id,
                Document.workspace_id == principal.workspace_id,
                Document.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if document is None:
        raise NotFoundError("Không tìm thấy tài liệu.")

    # Delete vectors first: if Qdrant is unavailable, abort without pretending
    # the document was removed from retrieval.
    await request.app.state.ingestion.vectors.delete_document(principal.workspace_id, document.id)

    now = datetime.now(UTC)
    await db.execute(update(Chunk).where(Chunk.document_id == document.id).values(active=False))
    await db.execute(
        update(Job)
        .where(
            Job.document_id == document.id,
            Job.status == JobStatus.QUEUED.value,
        )
        .values(status=JobStatus.CANCELLED.value, finished_at=now, progress=100)
    )
    await db.execute(
        update(Job)
        .where(
            Job.document_id == document.id,
            Job.status == JobStatus.RUNNING.value,
        )
        .values(cancel_requested=True)
    )
    document.deleted_at = now
    document.status = DocumentStatus.CANCELLED.value
    document.updated_at = now
    await db.commit()
    storage_path = document.metadata_.get("storage_path")
    if storage_path:
        delete_file(settings, storage_path)
