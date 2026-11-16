"""Transactional ingestion pipeline.

Publication order is deliberate:
1. vectors are upserted inactive;
2. the new revision's vectors are activated in Qdrant;
3. PostgreSQL publishes the revision/chunks in one transaction;
4. old revision vectors are deactivated/deleted.

Until step 3, dense candidates from the new revision are dropped by the
retrieval-time database join because their chunks are not active. Old vectors
remain active but are likewise dropped once their chunks are inactive. This
avoids leaking unpublished content without creating a dense gap.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select, update

from app.chunking.chunker import StructureAwareChunker
from app.core.config import Settings
from app.core.errors import AppError, IngestionError
from app.domain.enums import (
    ChunkKind,
    DocumentStatus,
    JobStage,
    JobStatus,
    RevisionStatus,
)
from app.embeddings.service import EmbeddingService
from app.indexing.vector_store import VectorStore
from app.ingestion.jobs import add_event, claim_job
from app.ingestion.storage import read_verified
from app.observability.logging import get_logger
from app.parsing.detect import detect_file_type
from app.parsing.parsers import parse_document
from app.parsing.types import ParserError
from app.persistence.db import Database
from app.persistence.models import (
    Chunk,
    ChunkTerm,
    Document,
    DocumentRevision,
    Job,
    KnowledgeBase,
    Section,
)
from app.retrieval.text import term_frequencies

logger = get_logger(__name__)
_MAX_TERMS_PER_CHUNK = 300


class JobCancelled(Exception):
    pass


class IngestionPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        embeddings: EmbeddingService,
        vectors: VectorStore,
    ) -> None:
        self.settings = settings
        self.database = database
        self.embeddings = embeddings
        self.vectors = vectors

    # ------------------------------------------------------------------ state
    async def _check_cancel(self, job_id: uuid.UUID) -> None:
        async with self.database.sessionmaker() as session:
            cancel_requested = (
                await session.execute(select(Job.cancel_requested).where(Job.id == job_id))
            ).scalar_one_or_none()
        if cancel_requested:
            raise JobCancelled()

    async def _stage(
        self,
        job_id: uuid.UUID,
        stage: JobStage,
        progress: int,
        message: str,
        *,
        doc_status: DocumentStatus | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        await self._check_cancel(job_id)
        async with self.database.sessionmaker() as session:
            job = await session.get(Job, job_id)
            if job is None:
                raise JobCancelled()
            job.stage = stage.value
            job.progress = max(0, min(100, progress))
            if doc_status is not None and job.document_id:
                await session.execute(
                    update(Document)
                    .where(Document.id == job.document_id)
                    .values(status=doc_status.value, updated_at=datetime.now(UTC))
                )
            add_event(session, job_id, stage.value, message, data=data)
            await session.commit()

    async def _fail(
        self,
        job_id: uuid.UUID,
        *,
        code: str,
        message: str,
        retryable: bool,
        revision_id: uuid.UUID | None,
        cleanup_revision: bool = True,
    ) -> None:
        logger.warning(
            "job_failed",
            extra={"event": "job_failed", "job_id": str(job_id), "code": code},
        )
        if cleanup_revision and revision_id is not None:
            try:
                await self.vectors.delete_revision(revision_id)
            except Exception:
                logger.warning(
                    "failed_revision_vector_cleanup_failed",
                    extra={"event": "vector_cleanup_failed", "job_id": str(job_id)},
                )
        async with self.database.sessionmaker() as session:
            job = await session.get(Job, job_id)
            if job is None:
                return
            job.status = JobStatus.FAILED.value
            job.stage = JobStage.DONE.value
            job.error_code = code
            job.error_message = message[:2000]
            job.finished_at = datetime.now(UTC)
            job.progress = 100
            if revision_id:
                revision = await session.get(DocumentRevision, revision_id)
                if revision is not None and revision.status != RevisionStatus.READY.value:
                    revision.status = RevisionStatus.FAILED.value
                    revision.error_code = code
                    revision.error_message = message[:2000]
            if job.document_id:
                document = await session.get(Document, job.document_id)
                if document is not None and document.deleted_at is None:
                    document.status = DocumentStatus.FAILED.value
                    document.error_code = code
                    document.error_message = message[:2000]
            add_event(session, job_id, JobStage.DONE.value, message, level="error")
            await session.commit()

    async def _cancel(self, job_id: uuid.UUID, revision_id: uuid.UUID | None) -> None:
        if revision_id:
            try:
                await self.vectors.delete_revision(revision_id)
            except Exception:
                logger.warning(
                    "cancelled_revision_vector_cleanup_failed",
                    extra={"event": "vector_cleanup_failed", "job_id": str(job_id)},
                )
        async with self.database.sessionmaker() as session:
            job = await session.get(Job, job_id)
            if job is None:
                return
            job.status = JobStatus.CANCELLED.value
            job.stage = JobStage.DONE.value
            job.progress = 100
            job.finished_at = datetime.now(UTC)
            if revision_id:
                revision = await session.get(DocumentRevision, revision_id)
                if revision is not None and revision.status != RevisionStatus.READY.value:
                    revision.status = RevisionStatus.CANCELLED.value
            if job.document_id:
                document = await session.get(Document, job.document_id)
                if document is not None and document.deleted_at is None:
                    document.status = DocumentStatus.CANCELLED.value
            add_event(session, job_id, JobStage.DONE.value, "Job đã bị hủy.")
            await session.commit()

    # ------------------------------------------------------------------- run
    async def run(self, job_id: uuid.UUID) -> None:
        job = await claim_job(self.database, job_id)
        if job is None:
            return
        revision_id = job.revision_id
        started = time.perf_counter()
        try:
            async with self.database.sessionmaker() as session:
                refresh_job = await session.get(Job, job_id)
                assert refresh_job is not None
                if refresh_job.document_id is None:
                    raise IngestionError("Job thiếu document_id.", code="job_invalid")
                document = await session.get(Document, refresh_job.document_id)
                revision = await session.get(DocumentRevision, revision_id) if revision_id else None
                if document is None or revision is None:
                    raise IngestionError("Không tìm thấy document/revision.", code="job_invalid")

                if revision.status == RevisionStatus.READY.value:
                    await self._reconcile_ready(job_id, document.id, revision.id)
                    return
                if document.deleted_at is not None:
                    raise JobCancelled()

                async with self.database.sessionmaker() as stage_session:
                    job_row = await stage_session.get(Job, job_id)
                    if job_row is not None:
                        add_event(
                            stage_session,
                            job_id,
                            JobStage.QUEUED.value,
                            "Worker đã nhận job.",
                        )
                        await stage_session.commit()

                revision.status = RevisionStatus.PROCESSING.value
                await session.commit()

                result = await self._ingest(job_id, document, revision)

            await self._succeed(job_id, result, elapsed_ms=(time.perf_counter() - started) * 1000)
        except JobCancelled:
            await self._cancel(job_id, revision_id)
        except ParserError as exc:
            await self._fail(
                job_id,
                code=exc.code,
                message=exc.message,
                retryable=False,
                revision_id=revision_id,
            )
        except AppError as exc:
            detail = ""
            if exc.details:
                detail = f" | details: {exc.details}"
            await self._fail(
                job_id,
                code=exc.code,
                message=f"{exc.message}{detail}",
                retryable=exc.retryable,
                revision_id=revision_id,
            )
        except Exception as exc:
            logger.exception(
                "ingestion_unhandled_error",
                extra={"event": "ingestion_unhandled_error", "job_id": str(job_id)},
            )
            await self._fail(
                job_id,
                code="ingestion_internal_error",
                message=f"Lỗi xử lý nội bộ ({type(exc).__name__}).",
                retryable=True,
                revision_id=revision_id,
            )

    async def _reconcile_ready(
        self, job_id: uuid.UUID, document_id: uuid.UUID, revision_id: uuid.UUID
    ) -> None:
        async with self.database.sessionmaker() as session:
            document = await session.get(Document, document_id)
            revision = await session.get(DocumentRevision, revision_id)
            if document is None or revision is None:
                return
            await self.vectors.activate_revision(
                document.workspace_id, document.knowledge_base_id, revision.id
            )
            job = await session.get(Job, job_id)
            if job is None:
                return
            job.status = JobStatus.SUCCEEDED.value
            job.stage = JobStage.DONE.value
            job.progress = 100
            job.finished_at = datetime.now(UTC)
            job.result = {"reconciled": True, "chunk_count": revision.chunk_count}
            document.status = DocumentStatus.READY.value
            document.error_code = None
            document.error_message = None
            add_event(session, job_id, JobStage.DONE.value, "Đã đối soát revision READY có sẵn.")
            await session.commit()

    async def _ingest(
        self, job_id: uuid.UUID, document: Document, revision: DocumentRevision
    ) -> dict[str, Any]:
        settings = self.settings
        await self._stage(
            job_id,
            JobStage.VALIDATING,
            5,
            "Đang kiểm tra tệp nguồn và chữ ký định dạng.",
            doc_status=DocumentStatus.VALIDATING,
        )
        storage_path = document.metadata_.get("storage_path")
        if not storage_path:
            raise IngestionError("Document thiếu đường dẫn tệp nguồn.", code="source_file_missing")
        content = read_verified(settings, storage_path, revision.sha256)
        detected = detect_file_type(document.filename, content, document.mime_type)

        await self._stage(
            job_id,
            JobStage.PARSING,
            15,
            f"Đang trích xuất nội dung ({detected.kind}).",
            doc_status=DocumentStatus.PARSING,
        )
        parsed = parse_document(content, detected.kind, document.filename)

        await self._check_cancel(job_id)
        await self._stage(
            job_id,
            JobStage.CHUNKING,
            25,
            "Đang chia đoạn theo cấu trúc và ngữ nghĩa.",
            doc_status=DocumentStatus.CHUNKING,
        )
        async with self.database.sessionmaker() as session:
            kb = await session.get(KnowledgeBase, document.knowledge_base_id)
            if kb is None:
                raise IngestionError("Knowledge base không tồn tại.", code="knowledge_base_missing")
            if kb.embedding_model != settings.embedding_model:
                raise IngestionError(
                    "Embedding model của knowledge base khác cấu hình hiện tại. "
                    "Cần tạo KB mới hoặc reindex có kiểm soát.",
                    code="embedding_model_mismatch",
                    details={
                        "knowledge_base_model": kb.embedding_model,
                        "configured_model": settings.embedding_model,
                    },
                )
            target = kb.chunk_target_tokens
            chunker = StructureAwareChunker(
                target_tokens=target,
                max_tokens=max(settings.chunk_max_tokens, target),
                overlap_tokens=settings.chunk_overlap_tokens,
                parent_max_tokens=max(1024, target * 3),
            )
            result = chunker.chunk(parsed)

            # A retried job reuses its revision. Clear any partial rows from the
            # previous attempt so publication cannot activate stale chunks.
            await session.execute(delete(Chunk).where(Chunk.revision_id == revision.id))
            await session.execute(delete(Section).where(Section.revision_id == revision.id))
            await session.flush()

            section_ids: dict[int, uuid.UUID] = {}
            for section in result.sections:
                section_row = Section(
                    workspace_id=document.workspace_id,
                    document_id=document.id,
                    revision_id=revision.id,
                    parent_section_id=section_ids.get(section.parent_ordinal)
                    if section.parent_ordinal is not None
                    else None,
                    title=section.title or None,
                    level=section.level,
                    ordinal=section.ordinal,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    path=section.path,
                )
                session.add(section_row)
                await session.flush()
                section_ids[section.ordinal] = section_row.id

            draft_ids = [uuid.uuid4() for _ in result.chunks]
            term_rows: list[dict[str, Any]] = []
            child_payloads: list[dict[str, Any]] = []
            for index, draft in enumerate(result.chunks):
                chunk_id = draft_ids[index]
                parent_id = (
                    draft_ids[draft.parent_index]
                    if draft.parent_index is not None and draft.parent_index < len(draft_ids)
                    else None
                )
                term_count = 0
                if draft.kind == ChunkKind.CHILD.value:
                    frequencies = term_frequencies(draft.text)
                    top_terms = frequencies.most_common(_MAX_TERMS_PER_CHUNK)
                    term_count = sum(tf for _, tf in top_terms)
                    for term, tf in top_terms:
                        term_rows.append({"chunk_id": chunk_id, "term": term, "tf": tf})
                chunk = Chunk(
                    id=chunk_id,
                    workspace_id=document.workspace_id,
                    knowledge_base_id=document.knowledge_base_id,
                    document_id=document.id,
                    revision_id=revision.id,
                    section_id=section_ids.get(draft.section_ordinal),
                    parent_chunk_id=parent_id,
                    kind=draft.kind,
                    ordinal=draft.ordinal,
                    text=draft.text,
                    token_count=draft.token_count,
                    term_count=term_count,
                    content_hash=draft.content_hash,
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    heading_path=draft.heading_path,
                    source_spans=draft.source_spans,
                    is_embedded=False,
                    embedding_model=None,
                    embedding_dimension=None,
                    active=False,
                )
                session.add(chunk)
                if draft.kind == ChunkKind.CHILD.value:
                    child_payloads.append(
                        {
                            "chunk_id": chunk_id,
                            "text": draft.text,
                            "content_hash": draft.content_hash,
                            "token_count": draft.token_count,
                            "ordinal": draft.ordinal,
                            "page_start": draft.page_start,
                            "page_end": draft.page_end,
                        }
                    )

            for start in range(0, len(term_rows), 2000):
                session.add_all([ChunkTerm(**row) for row in term_rows[start : start + 2000]])
                await session.flush()
            chunked_revision = await session.get(DocumentRevision, revision.id)
            if chunked_revision is None:
                raise IngestionError("Không tìm thấy revision.", code="job_invalid")
            chunked_revision.section_count = len(result.sections)
            chunked_revision.chunk_count = len(result.chunks)
            chunked_revision.page_count = parsed.page_count
            await session.commit()

        if not child_payloads:
            raise IngestionError("Tài liệu không tạo ra đoạn văn bản nào.", code="no_chunks")

        await self._stage(
            job_id,
            JobStage.EMBEDDING,
            35,
            f"Đang tạo embedding cho {len(child_payloads)} đoạn.",
            doc_status=DocumentStatus.EMBEDDING,
        )

        async def progress(done: int, total: int) -> None:
            percent = 35 + int(45 * (done / max(total, 1)))
            await self._stage(
                job_id,
                JobStage.EMBEDDING,
                percent,
                f"Đã embedding {done}/{total} đoạn.",
                doc_status=DocumentStatus.EMBEDDING,
            )

        async with self.database.sessionmaker() as session:
            texts = [item["text"] for item in child_payloads]
            vectors = await self.embeddings.embed_documents(session, texts, progress=progress)

        await self._check_cancel(job_id)
        await self._stage(
            job_id,
            JobStage.INDEXING,
            82,
            "Đang ghi vector vào Qdrant.",
            doc_status=DocumentStatus.INDEXING,
        )
        await self.vectors.ensure_collection()
        # Remove vectors left by a previously failed attempt for this revision.
        await self.vectors.delete_revision(revision.id)
        async with self.database.sessionmaker() as session:
            items = []
            for payload, vector in zip(child_payloads, vectors, strict=True):
                items.append({**payload, "vector": vector})
            await self.vectors.upsert(
                workspace_id=document.workspace_id,
                kb_id=document.knowledge_base_id,
                document_id=document.id,
                revision_id=revision.id,
                filename=document.filename,
                items=items,
            )
            for start in range(0, len(items), 100):
                batch = items[start : start + 100]
                await session.execute(
                    update(Chunk)
                    .where(Chunk.id.in_([item["chunk_id"] for item in batch]))
                    .values(
                        is_embedded=True,
                        embedding_model=self.embeddings.model_id,
                        embedding_dimension=self.embeddings.dimension,
                    )
                )
            await session.commit()

        await self._check_cancel(job_id)
        await self._stage(
            job_id,
            JobStage.PUBLISHING,
            92,
            "Đang công bố revision mới.",
            doc_status=DocumentStatus.INDEXING,
        )

        # Activate new vectors before DB publication. Until DB publication they
        # are filtered out by the active-chunk join, so this is not a leak.
        await self.vectors.activate_revision(
            document.workspace_id, document.knowledge_base_id, revision.id
        )

        old_revision_id: uuid.UUID | None = None
        try:
            async with self.database.sessionmaker() as session:
                locked = (
                    await session.execute(
                        select(Document).where(Document.id == document.id).with_for_update()
                    )
                ).scalar_one()
                if locked.deleted_at is not None:
                    raise JobCancelled()
                old_revision_id = locked.active_revision_id
                now = datetime.now(UTC)
                await session.execute(
                    update(Chunk).where(Chunk.revision_id == revision.id).values(active=True)
                )
                if old_revision_id and old_revision_id != revision.id:
                    await session.execute(
                        update(Chunk)
                        .where(Chunk.revision_id == old_revision_id)
                        .values(active=False)
                    )
                    old_revision = await session.get(DocumentRevision, old_revision_id)
                    if old_revision is not None:
                        old_revision.status = RevisionStatus.SUPERSEDED.value
                published_revision = await session.get(DocumentRevision, revision.id)
                if published_revision is None:
                    raise IngestionError("Không tìm thấy revision.", code="job_invalid")
                published_revision.status = RevisionStatus.READY.value
                published_revision.ready_at = now
                published_revision.error_code = None
                published_revision.error_message = None
                locked.status = DocumentStatus.READY.value
                locked.active_revision_id = revision.id
                locked.latest_revision_id = revision.id
                locked.error_code = None
                locked.error_message = None
                locked.updated_at = now
                metadata = dict(locked.metadata_ or {})
                metadata["page_count"] = parsed.page_count
                metadata["section_count"] = len(result.sections)
                locked.metadata_ = metadata
                add_event(
                    session,
                    job_id,
                    JobStage.PUBLISHING.value,
                    "Revision đã được công bố.",
                    data={"revision_id": str(revision.id)},
                )
                await session.commit()
        except JobCancelled:
            await self.vectors.delete_revision(revision.id)
            raise
        except Exception:
            # DB publication failed: deactivate the new vectors again so the
            # previous revision stays authoritative.
            try:
                await self.vectors.delete_revision(revision.id)
            finally:
                raise

        try:
            if old_revision_id and old_revision_id != revision.id:
                await self.vectors.deactivate_revision(old_revision_id)
                await self.vectors.delete_revision(old_revision_id)
        except Exception:
            logger.warning(
                "old_revision_vector_cleanup_failed",
                extra={"event": "vector_cleanup_failed", "job_id": str(job_id)},
            )

        return {
            "page_count": parsed.page_count,
            "section_count": len(result.sections),
            "chunk_count": len(result.chunks),
            "child_chunk_count": len(child_payloads),
            "embedding_model": self.embeddings.model_id,
            "embedding_dimension": self.embeddings.dimension,
        }

    async def _succeed(
        self, job_id: uuid.UUID, result: dict[str, Any], *, elapsed_ms: float
    ) -> None:
        async with self.database.sessionmaker() as session:
            job = await session.get(Job, job_id)
            if job is None:
                return
            job.status = JobStatus.SUCCEEDED.value
            job.stage = JobStage.DONE.value
            job.progress = 100
            job.finished_at = datetime.now(UTC)
            job.result = {**result, "elapsed_ms": round(elapsed_ms, 1)}
            add_event(
                session,
                job_id,
                JobStage.DONE.value,
                "Xử lý tài liệu hoàn tất.",
                data={"elapsed_ms": round(elapsed_ms, 1)},
            )
            await session.commit()
