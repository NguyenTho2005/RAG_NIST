"""Workspace retrieval settings and effective defaults."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ValidationFailedError
from app.persistence.models import WorkspaceSettings
from app.security.auth import Principal, get_db, require_read, require_write

router = APIRouter()

Mode = Literal["basic", "hybrid", "auto"]


class WorkspaceSettingsOut(BaseModel):
    default_mode: str
    top_k: int
    rerank_enabled: bool
    rerank_available: bool
    updated_at: datetime


class WorkspaceSettingsUpdate(BaseModel):
    default_mode: Mode | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank_enabled: bool | None = None


async def workspace_retrieval_defaults(
    db: AsyncSession, workspace_id: uuid.UUID
) -> tuple[str, int]:
    row = await db.get(WorkspaceSettings, workspace_id)
    if row is None:
        return "hybrid", 8
    return row.default_mode, row.top_k


def _settings_out(request: Request, row: WorkspaceSettings) -> WorkspaceSettingsOut:
    settings = request.app.state.settings
    return WorkspaceSettingsOut(
        default_mode=row.default_mode,
        top_k=row.top_k,
        rerank_enabled=row.rerank_enabled,
        rerank_available=bool(settings.rerank_enabled and settings.reranker_provider != "none"),
        updated_at=row.updated_at,
    )


async def _get_or_create(
    request: Request, db: AsyncSession, workspace_id: uuid.UUID
) -> WorkspaceSettings:
    row = await db.get(WorkspaceSettings, workspace_id)
    if row is not None:
        return row
    settings = request.app.state.settings
    row = WorkspaceSettings(
        workspace_id=workspace_id,
        default_mode=settings.retrieval_mode
        if settings.retrieval_mode in {"basic", "hybrid", "auto"}
        else "hybrid",
        top_k=min(max(settings.rerank_top_n, 1), 50),
        rerank_enabled=settings.rerank_enabled,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


@router.get("/settings", response_model=WorkspaceSettingsOut, summary="Workspace settings")
async def get_workspace_settings(
    request: Request,
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> WorkspaceSettingsOut:
    row = await _get_or_create(request, db, principal.workspace_id)
    return _settings_out(request, row)


@router.patch("/settings", response_model=WorkspaceSettingsOut, summary="Update settings")
async def update_workspace_settings(
    request: Request,
    body: WorkspaceSettingsUpdate,
    principal: Principal = Depends(require_write),
    db: AsyncSession = Depends(get_db),
) -> WorkspaceSettingsOut:
    row = await _get_or_create(request, db, principal.workspace_id)
    if body.default_mode is not None:
        row.default_mode = body.default_mode
    if body.top_k is not None:
        row.top_k = body.top_k
    if body.rerank_enabled is not None:
        settings = request.app.state.settings
        if not settings.rerank_enabled or settings.reranker_provider == "none":
            raise ValidationFailedError(
                "Reranker chưa được bật trên máy chủ.",
                code="reranker_not_available",
            )
        row.rerank_enabled = body.rerank_enabled
    row.updated_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(row)
    return _settings_out(request, row)
