"""Bounded retrieval-evaluation runner endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import retrieval_service
from app.api.schemas import Page
from app.core.errors import NotFoundError
from app.evaluation.dataset import list_datasets, load_dataset
from app.evaluation.runner import evaluate_retrieval
from app.persistence.models import EvaluationRun, KnowledgeBase
from app.retrieval.service import RetrievalService
from app.security.auth import Principal, get_db, require_read, require_write

router = APIRouter()

EvaluationMode = Literal["basic", "hybrid", "auto"]


class EvaluationDatasetOut(BaseModel):
    id: str
    title: str
    description: str
    version: str
    sha256: str
    case_count: int
    answerable_count: int
    unanswerable_count: int
    categories: dict[str, int]
    review_status: str
    authoring: str
    languages: list[str]
    limitations: list[str]


class EvaluationRunCreate(BaseModel):
    knowledge_base_id: uuid.UUID
    dataset_id: str = Field(default="grc-vi-v1", min_length=1, max_length=64)
    mode: EvaluationMode = "hybrid"
    top_k: int = Field(default=10, ge=1, le=50)
    limit: int | None = Field(default=None, ge=1, le=200)


class EvaluationRunSummary(BaseModel):
    id: str
    dataset_id: str
    dataset_sha256: str
    knowledge_base_id: str
    mode: str
    top_k: int
    status: str
    case_count: int
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    finished_at: datetime | None


class EvaluationRunOut(EvaluationRunSummary):
    category_metrics: dict[str, Any] = Field(default_factory=dict)
    unanswerable: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    cases: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None


def _summary(row: EvaluationRun) -> EvaluationRunSummary:
    return EvaluationRunSummary(
        id=str(row.id),
        dataset_id=row.dataset_id,
        dataset_sha256=row.dataset_sha256,
        knowledge_base_id=str(row.knowledge_base_id),
        mode=row.mode,
        top_k=row.top_k,
        status=row.status,
        case_count=row.case_count,
        metrics=dict(row.metrics or {}),
        created_at=row.created_at,
        finished_at=row.finished_at,
    )


def _detail(row: EvaluationRun) -> EvaluationRunOut:
    return EvaluationRunOut(
        **_summary(row).model_dump(),
        category_metrics=dict(row.category_metrics or {}),
        unanswerable=dict(row.unanswerable or {}),
        config=dict(row.config or {}),
        cases=list(row.cases or []),
        error_message=row.error_message,
    )


@router.get(
    "/evaluations/datasets",
    response_model=list[EvaluationDatasetOut],
    summary="List bundled evaluation datasets",
)
async def datasets(
    request: Request,
    principal: Principal = Depends(require_read),
) -> list[EvaluationDatasetOut]:
    settings = request.app.state.settings
    return [EvaluationDatasetOut(**item) for item in list_datasets(settings.evaluation_dir)]


@router.post(
    "/evaluations/runs",
    response_model=EvaluationRunOut,
    status_code=201,
    summary="Run retrieval evaluation on a knowledge base",
)
async def create_run(
    request: Request,
    body: EvaluationRunCreate,
    principal: Principal = Depends(require_write),
    db: AsyncSession = Depends(get_db),
    retrieval: RetrievalService = Depends(retrieval_service),
) -> EvaluationRunOut:
    kb = await db.get(KnowledgeBase, body.knowledge_base_id)
    if kb is None or kb.workspace_id != principal.workspace_id or kb.deleted_at is not None:
        raise NotFoundError("Không tìm thấy knowledge base.")
    settings = request.app.state.settings
    dataset = load_dataset(settings.evaluation_dir, body.dataset_id)

    run = EvaluationRun(
        workspace_id=principal.workspace_id,
        knowledge_base_id=body.knowledge_base_id,
        dataset_id=dataset.dataset_id,
        dataset_sha256=dataset.sha256,
        mode=body.mode,
        top_k=body.top_k,
        status="running",
        case_count=0,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    try:
        result = await evaluate_retrieval(
            retrieval,
            workspace_id=principal.workspace_id,
            kb_id=body.knowledge_base_id,
            dataset=dataset,
            mode=body.mode,
            top_k=body.top_k,
            limit=body.limit,
        )
    except Exception as exc:
        run.status = "failed"
        run.error_message = f"{type(exc).__name__}: {exc}"[:2000]
        run.finished_at = datetime.now(UTC)
        await db.commit()
        raise
    report = result.to_report()
    run.status = "completed"
    run.case_count = len(result.cases)
    run.metrics = report["metrics"]
    run.category_metrics = report["category_metrics"]
    run.unanswerable = report["unanswerable"]
    run.config = {
        **report["config"],
        "retrieval_mode_used": report["retrieval_mode_used"],
        "embedding_model": report["embedding_model"],
        "dataset_review_status": report["dataset"]["review_status"],
        "disclaimer": report["disclaimer"],
    }
    run.cases = report["cases"]
    run.finished_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(run)
    return _detail(run)


@router.get(
    "/evaluations/runs",
    response_model=Page[EvaluationRunSummary],
    summary="List evaluation runs",
)
async def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> Page[EvaluationRunSummary]:
    where = (EvaluationRun.workspace_id == principal.workspace_id,)
    total = (await db.execute(select(func.count(EvaluationRun.id)).where(*where))).scalar_one()
    rows = (
        (
            await db.execute(
                select(EvaluationRun)
                .where(*where)
                .order_by(EvaluationRun.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )
    return Page(
        items=[_summary(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/evaluations/runs/{run_id}",
    response_model=EvaluationRunOut,
    summary="Get evaluation run detail",
)
async def get_run(
    run_id: uuid.UUID,
    principal: Principal = Depends(require_read),
    db: AsyncSession = Depends(get_db),
) -> EvaluationRunOut:
    row = await db.get(EvaluationRun, run_id)
    if row is None or row.workspace_id != principal.workspace_id:
        raise NotFoundError("Không tìm thấy lần đánh giá.")
    return _detail(row)
