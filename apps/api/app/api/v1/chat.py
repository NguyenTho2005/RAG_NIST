"""Chat endpoints with grounded citations, SSE streaming and persisted sessions."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import app_database, generation_service, retrieval_service
from app.api.schemas import Page
from app.api.v1.search import RetrievalMode, SourceOut, source_out, trace_selected
from app.api.v1.settings import workspace_retrieval_defaults
from app.core.errors import AppError, NotFoundError, ValidationFailedError
from app.generation.service import GenerationOutcome, GenerationService
from app.observability.logging import get_logger
from app.observability.traces import create_trace, update_trace
from app.persistence.db import Database
from app.persistence.models import ChatMessage, ChatSession
from app.retrieval.context import ContextSource
from app.retrieval.service import RetrievalService
from app.retrieval.text import count_tokens
from app.security.auth import Principal, get_db, require_read, require_write

logger = get_logger(__name__)
router = APIRouter()

_DEFAULT_TITLE = "Cuộc trò chuyện mới"


class ChatRequest(BaseModel):
    knowledge_base_id: uuid.UUID
    message: str = Field(min_length=1, max_length=4000)
    mode: RetrievalMode | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)
    session_id: uuid.UUID | None = None


class CitationOut(BaseModel):
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
    quote: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    sources: list[SourceOut]
    grounded: bool
    refused: bool
    citation_status: str
    warnings: list[str] = Field(default_factory=list)
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    trace_id: str
    session_id: str
    timings: dict[str, float]
    mode: str
    mode_used: str
    candidates: dict[str, int]


class ChatSessionOut(BaseModel):
    id: str
    knowledge_base_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: list[dict[str, Any]]
    trace_id: str | None
    token_usage: dict[str, Any]
    created_at: datetime


class ChatSessionDetail(ChatSessionOut):
    messages: list[ChatMessageOut] = Field(default_factory=list)


def _session_out(session: ChatSession) -> ChatSessionOut:
    return ChatSessionOut(
        id=str(session.id),
        knowledge_base_id=str(session.knowledge_base_id),
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _message_out(message: ChatMessage) -> ChatMessageOut:
    return ChatMessageOut(
        id=str(message.id),
        role=message.role,
        content=message.content,
        citations=list(message.citations or []),
        trace_id=str(message.trace_id) if message.trace_id else None,
        token_usage=dict(message.token_usage or {}),
        created_at=message.created_at,
    )


async def _resolve_session(
    db: AsyncSession,
    *,
    principal: Principal,
    kb_id: uuid.UUID,
    message: str,
    session_id: uuid.UUID | None,
) -> ChatSession:
    if session_id is not None:
        session = await db.get(ChatSession, session_id)
        if (
            session is None
            or session.workspace_id != principal.workspace_id
            or session.knowledge_base_id != kb_id
        ):
            raise NotFoundError("Không tìm thấy phiên trò chuyện.")
        return session
    session = ChatSession(
        workspace_id=principal.workspace_id,
        knowledge_base_id=kb_id,
        title=message.strip()[:120] or _DEFAULT_TITLE,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def _persist_exchange(
    database: Database,
    *,
    session_id: uuid.UUID,
    user_message: str,
    outcome: GenerationOutcome,
    trace_id: uuid.UUID,
) -> None:
    now = datetime.now(UTC)
    async with database.sessionmaker() as session:
        session.add(
            ChatMessage(
                session_id=session_id,
                role="user",
                content=user_message,
                citations=[],
                trace_id=trace_id,
                token_usage={},
                created_at=now,
            )
        )
        session.add(
            ChatMessage(
                session_id=session_id,
                role="assistant",
                content=outcome.answer,
                citations=outcome.citations,
                trace_id=trace_id,
                token_usage=outcome.usage,
                created_at=now + timedelta(milliseconds=1),
            )
        )
        await session.commit()


def _chat_response(
    *,
    outcome: GenerationOutcome,
    sources: list[ContextSource],
    trace_id: uuid.UUID,
    session_id: uuid.UUID,
    timings: dict[str, float],
    mode: str,
    mode_used: str,
    candidates: dict[str, int],
) -> ChatResponse:
    return ChatResponse(
        answer=outcome.answer,
        citations=[CitationOut(**citation) for citation in outcome.citations],
        sources=[source_out(source) for source in sources],
        grounded=outcome.grounded,
        refused=outcome.refused,
        citation_status=outcome.citation_status,
        warnings=outcome.warnings,
        model=outcome.model,
        usage=outcome.usage,
        trace_id=str(trace_id),
        session_id=str(session_id),
        timings=timings,
        mode=mode,
        mode_used=mode_used,
        candidates=candidates,
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat", response_model=ChatResponse, summary="Grounded RAG chat")
async def chat(
    request: Request,
    body: ChatRequest,
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
    retrieval: RetrievalService = Depends(retrieval_service),
    generation: GenerationService = Depends(generation_service),
) -> ChatResponse:
    query = body.message.strip()
    if not query:
        raise ValidationFailedError("Tin nhắn không được để trống.")
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
    generation_started = time.perf_counter()
    generated = await generation.generate(query, sources)
    generation_ms = round((time.perf_counter() - generation_started) * 1000, 2)
    timings = {**outcome.timings, "generation_ms": generation_ms}
    stages: list[dict[str, Any]] = [
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
    ]
    if generated.refused:
        stages.append(
            {"stage": "generation", "status": "skipped", "reason": "insufficient_context"}
        )
    else:
        stages.append(
            {
                "stage": "generation",
                "status": "ok",
                "ms": generation_ms,
                "attempts": generated.attempts,
                "grounded": generated.grounded,
                "citation_status": generated.citation_status,
            }
        )
    trace_id = await create_trace(
        app_database(request),
        workspace_id=principal.workspace_id,
        knowledge_base_id=body.knowledge_base_id,
        query=query,
        processed_query=query,
        mode=outcome.mode,
        config=retrieval.trace_config(top_k=top_k, mode=mode),
        stages=stages,
        candidates=outcome.candidates,
        selected=trace_selected(sources),
        timings=timings,
        token_usage=generated.usage,
    )
    session = await _resolve_session(
        db,
        principal=principal,
        kb_id=body.knowledge_base_id,
        message=query,
        session_id=body.session_id,
    )
    await _persist_exchange(
        app_database(request),
        session_id=session.id,
        user_message=query,
        outcome=generated,
        trace_id=trace_id,
    )
    return _chat_response(
        outcome=generated,
        sources=sources,
        trace_id=trace_id,
        session_id=session.id,
        timings=timings,
        mode=outcome.mode,
        mode_used=outcome.mode_used,
        candidates=outcome.candidates,
    )


@router.post("/chat/stream", summary="Grounded RAG chat over Server-Sent Events")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
    retrieval: RetrievalService = Depends(retrieval_service),
    generation: GenerationService = Depends(generation_service),
) -> StreamingResponse:
    query = body.message.strip()
    if not query:
        raise ValidationFailedError("Tin nhắn không được để trống.")
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
    session = await _resolve_session(
        db,
        principal=principal,
        kb_id=body.knowledge_base_id,
        message=query,
        session_id=body.session_id,
    )
    generation_started = time.perf_counter()
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
            {"stage": "generation", "status": "streaming"},
        ],
        candidates=outcome.candidates,
        selected=trace_selected(sources),
        timings=outcome.timings,
    )
    database = app_database(request)

    async def event_stream() -> AsyncIterator[str]:
        yield _sse(
            "meta",
            {
                "trace_id": str(trace_id),
                "session_id": str(session.id),
                "mode": outcome.mode,
                "mode_used": outcome.mode_used,
                "sources": [source_out(source).model_dump() for source in sources],
                "warnings": outcome.warnings,
            },
        )
        parts: list[str] = []
        try:
            if sources:
                async for token in generation.stream(query, sources):
                    parts.append(token)
                    yield _sse("token", {"text": token})
                answer = "".join(parts)
                usage = generation.prompt_usage(query, sources)
                usage["output_tokens"] = count_tokens(answer)
                generated = generation.finalize(
                    answer,
                    sources,
                    usage=usage,
                    attempts=1,
                )
            else:
                generated = generation.refusal()
                yield _sse("token", {"text": generated.answer})
            generation_ms = round((time.perf_counter() - generation_started) * 1000, 2)
            timings = {**outcome.timings, "generation_ms": generation_ms}
            stages: list[dict[str, Any]] = [
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
                {
                    "stage": "generation",
                    "status": "skipped" if generated.refused else "ok",
                    "ms": generation_ms,
                    "grounded": generated.grounded,
                    "citation_status": generated.citation_status,
                },
            ]
            await update_trace(
                database,
                trace_id,
                stages=stages,
                timings=timings,
                token_usage=generated.usage,
            )
            await _persist_exchange(
                database,
                session_id=session.id,
                user_message=query,
                outcome=generated,
                trace_id=trace_id,
            )
            yield _sse(
                "citations",
                {
                    "citations": generated.citations,
                    "grounded": generated.grounded,
                    "refused": generated.refused,
                    "citation_status": generated.citation_status,
                    "warnings": generated.warnings,
                    "model": generated.model,
                    "usage": generated.usage,
                },
            )
            yield _sse(
                "done",
                {
                    "trace_id": str(trace_id),
                    "session_id": str(session.id),
                    "timings": timings,
                },
            )
        except AppError as exc:
            await update_trace(
                database,
                trace_id,
                stages=[
                    {
                        "stage": "generation",
                        "status": "error",
                        "code": exc.code,
                    }
                ],
            )
            yield _sse(
                "error",
                {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "trace_id": str(trace_id),
                },
            )
        except Exception:
            logger.exception(
                "chat_stream_failed",
                extra={"event": "chat_stream_failed", "trace_id": str(trace_id)},
            )
            yield _sse(
                "error",
                {
                    "code": "generation_internal_error",
                    "message": "Lỗi khi tạo câu trả lời.",
                    "retryable": True,
                    "trace_id": str(trace_id),
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/sessions", response_model=Page[ChatSessionOut], summary="List chat sessions")
async def list_sessions(
    knowledge_base_id: uuid.UUID | None = None,
    limit: int = 50,
    offset: int = 0,
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> Page[ChatSessionOut]:
    conditions = [ChatSession.workspace_id == principal.workspace_id]
    if knowledge_base_id:
        conditions.append(ChatSession.knowledge_base_id == knowledge_base_id)
    total = (await db.execute(select(func.count(ChatSession.id)).where(*conditions))).scalar_one()
    rows = (
        (
            await db.execute(
                select(ChatSession)
                .where(*conditions)
                .order_by(ChatSession.updated_at.desc())
                .limit(min(max(limit, 1), 200))
                .offset(max(offset, 0))
            )
        )
        .scalars()
        .all()
    )
    return Page(
        items=[_session_out(session) for session in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/chat/sessions/{session_id}",
    response_model=ChatSessionDetail,
    summary="Get one chat session with messages",
)
async def get_session(
    session_id: uuid.UUID,
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> ChatSessionDetail:
    session = await db.get(ChatSession, session_id)
    if session is None or session.workspace_id != principal.workspace_id:
        raise NotFoundError("Không tìm thấy phiên trò chuyện.")
    messages = (
        (
            await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session.id)
                .order_by(ChatMessage.created_at, ChatMessage.id)
            )
        )
        .scalars()
        .all()
    )
    base = _session_out(session)
    return ChatSessionDetail(
        **base.model_dump(),
        messages=[_message_out(message) for message in messages],
    )


@router.delete(
    "/chat/sessions/{session_id}",
    status_code=204,
    summary="Delete a chat session",
)
async def delete_session(
    session_id: uuid.UUID,
    principal: Principal = Depends(require_write),
    db: AsyncSession = Depends(get_db),
) -> None:
    session = await db.get(ChatSession, session_id)
    if session is None or session.workspace_id != principal.workspace_id:
        raise NotFoundError("Không tìm thấy phiên trò chuyện.")
    await db.delete(session)
    await db.commit()
