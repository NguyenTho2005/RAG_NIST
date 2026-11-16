"""API-key authentication and workspace-scoped authorization."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ForbiddenError, UnauthorizedError
from app.observability.logging import workspace_id_ctx
from app.persistence.db import Database
from app.persistence.models import ApiKey

_bearer = HTTPBearer(auto_error=False, description="Workspace API key")

KEY_PREFIX = "rag"


@dataclass(frozen=True, slots=True)
class Principal:
    workspace_id: uuid.UUID
    api_key_id: str
    scopes: tuple[str, ...]
    workspace_name: str

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "admin" in self.scopes


def generate_api_key() -> tuple[str, str, str]:
    """Return (plaintext key, display prefix, sha256 hash)."""
    secret = secrets.token_urlsafe(32)
    key = f"{KEY_PREFIX}_{secret}"
    prefix = key[:14]
    return key, prefix, hash_api_key(key)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def resolve_principal(db: AsyncSession, key: str) -> Principal:
    key_hash = hash_api_key(key)
    row = (
        await db.execute(
            select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.revoked_at.is_(None))
        )
    ).scalar_one_or_none()
    if row is None:
        raise UnauthorizedError("API key không hợp lệ hoặc đã bị thu hồi.")
    now = datetime.now(UTC)
    if row.last_used_at is None or now - row.last_used_at > timedelta(minutes=5):
        row.last_used_at = now
        await db.commit()
    from app.persistence.models import Workspace

    workspace = await db.get(Workspace, row.workspace_id)
    if workspace is None:
        raise UnauthorizedError("Không tìm thấy workspace của API key.")
    workspace_id_ctx.set(str(row.workspace_id))
    return Principal(
        workspace_id=row.workspace_id,
        api_key_id=str(row.id),
        scopes=tuple(row.scopes),
        workspace_name=workspace.name,
    )


async def get_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise UnauthorizedError("Thiếu API key. Gửi header Authorization: Bearer <key>.")
    database: Database = request.app.state.database
    async with database.sessionmaker() as db:
        principal = await resolve_principal(db, credentials.credentials)
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None:
        await limiter.check(principal.api_key_id)
    return principal


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    database: Database = request.app.state.database
    async with database.sessionmaker() as session:
        yield session


def require_scope(
    scope: str,
) -> Callable[..., Awaitable[Principal]]:
    async def _dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.has_scope(scope):
            raise ForbiddenError(f"API key không có quyền '{scope}'.")
        return principal

    return _dependency


require_read = require_scope("read")
require_write = require_scope("write")
require_admin = require_scope("admin")
