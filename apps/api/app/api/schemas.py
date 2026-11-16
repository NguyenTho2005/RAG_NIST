"""Shared API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Page[T](BaseModel):
    items: list[T]
    total: int
    limit: int
    offset: int


class PaginationParams(BaseModel):
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    chunk_target_tokens: int | None = Field(default=None, ge=128, le=1024)


class KnowledgeBaseOut(ORMModel):
    id: str
    name: str
    description: str | None
    embedding_model: str
    embedding_dimension: int
    chunk_target_tokens: int
    document_count: int = 0
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str | None = None
