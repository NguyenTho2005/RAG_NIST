"""Knowledge base endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import KnowledgeBaseCreate, KnowledgeBaseOut, Page
from app.core.config import Settings, get_settings
from app.core.errors import ConflictError, NotFoundError
from app.persistence.models import Chunk, Document, KnowledgeBase
from app.security.auth import Principal, get_db, require_read, require_write

router = APIRouter()


def _to_out(kb: KnowledgeBase, document_count: int = 0, chunk_count: int = 0) -> KnowledgeBaseOut:
    return KnowledgeBaseOut(
        id=str(kb.id),
        name=kb.name,
        description=kb.description,
        embedding_model=kb.embedding_model,
        embedding_dimension=kb.embedding_dimension,
        chunk_target_tokens=kb.chunk_target_tokens,
        document_count=document_count,
        chunk_count=chunk_count,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


@router.post("/knowledge-bases", response_model=KnowledgeBaseOut, status_code=201)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    principal: Principal = Depends(require_write),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> KnowledgeBaseOut:
    duplicate = (
        await db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.workspace_id == principal.workspace_id,
                func.lower(KnowledgeBase.name) == payload.name.strip().lower(),
                KnowledgeBase.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if duplicate is not None:
        raise ConflictError(
            "Đã tồn tại knowledge base với tên này.",
            code="knowledge_base_exists",
            details={"knowledge_base_id": str(duplicate.id)},
        )
    kb = KnowledgeBase(
        workspace_id=principal.workspace_id,
        name=payload.name.strip(),
        description=payload.description,
        embedding_model=settings.embedding_model,
        embedding_dimension=settings.embedding_dimension,
        chunk_target_tokens=payload.chunk_target_tokens or settings.chunk_target_tokens,
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return _to_out(kb)


@router.get("/knowledge-bases", response_model=Page[KnowledgeBaseOut])
async def list_knowledge_bases(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> Page[KnowledgeBaseOut]:
    base_where = (
        KnowledgeBase.workspace_id == principal.workspace_id,
        KnowledgeBase.deleted_at.is_(None),
    )
    total = (await db.execute(select(func.count(KnowledgeBase.id)).where(*base_where))).scalar_one()
    rows = (
        (
            await db.execute(
                select(KnowledgeBase)
                .where(*base_where)
                .order_by(KnowledgeBase.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )

    doc_counts = dict(
        (
            await db.execute(
                select(Document.knowledge_base_id, func.count(Document.id))
                .where(
                    Document.workspace_id == principal.workspace_id,
                    Document.deleted_at.is_(None),
                )
                .group_by(Document.knowledge_base_id)
            )
        ).all()
    )
    chunk_counts = dict(
        (
            await db.execute(
                select(Chunk.knowledge_base_id, func.count(Chunk.id))
                .where(Chunk.workspace_id == principal.workspace_id, Chunk.active.is_(True))
                .group_by(Chunk.knowledge_base_id)
            )
        ).all()
    )
    items = [_to_out(kb, doc_counts.get(kb.id, 0), chunk_counts.get(kb.id, 0)) for kb in rows]
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseOut)
async def get_knowledge_base(
    kb_id: uuid.UUID,
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeBaseOut:
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
    docs = (
        await db.execute(
            select(func.count(Document.id)).where(
                Document.knowledge_base_id == kb.id, Document.deleted_at.is_(None)
            )
        )
    ).scalar_one()
    chunks = (
        await db.execute(
            select(func.count(Chunk.id)).where(
                Chunk.knowledge_base_id == kb.id, Chunk.active.is_(True)
            )
        )
    ).scalar_one()
    return _to_out(kb, docs, chunks)
