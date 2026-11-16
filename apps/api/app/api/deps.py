"""Request-scoped service dependencies."""

from __future__ import annotations

from fastapi import Request

from app.core.config import Settings
from app.generation.base import LLMProvider
from app.generation.service import GenerationService
from app.persistence.db import Database
from app.reranking.base import Reranker
from app.retrieval.service import RetrievalService


def app_database(request: Request) -> Database:
    database: Database = request.app.state.database
    return database


def retrieval_service(request: Request) -> RetrievalService:
    settings: Settings = request.app.state.settings
    runtime = request.app.state.ingestion
    reranker: Reranker | None = getattr(request.app.state, "reranker", None)
    return RetrievalService(
        settings=settings,
        database=request.app.state.database,
        embeddings=runtime.embeddings,
        vectors=runtime.vectors,
        reranker=reranker,
    )


def generation_service(request: Request) -> GenerationService:
    settings: Settings = request.app.state.settings
    provider: LLMProvider | None = getattr(request.app.state, "llm_provider", None)
    return GenerationService(settings=settings, provider=provider)
