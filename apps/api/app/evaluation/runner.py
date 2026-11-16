"""Retrieval evaluation runner over a versioned dataset and live knowledge base."""

from __future__ import annotations

import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.metrics import (
    hit_at_k,
    mean,
    ndcg_from_ranks,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.retrieval.service import RetrievalService


@dataclass(slots=True)
class CaseMetrics:
    recall_at_k: float
    precision_at_k: float
    hit_at_k: float
    mrr: float
    ndcg_at_k: float


@dataclass(slots=True)
class CaseOutcome:
    id: str
    question: str
    category: str
    answerable: bool
    expected_sources: list[dict[str, str]]
    matched_sources: list[dict[str, Any]]
    metrics: CaseMetrics | None
    retrieved: list[dict[str, Any]]
    latency_ms: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(slots=True)
class EvaluationResult:
    dataset: EvaluationDataset
    mode: str
    top_k: int
    retrieval_mode_used: str
    embedding_model: str
    metrics: dict[str, Any]
    category_metrics: dict[str, dict[str, Any]]
    unanswerable: dict[str, Any]
    cases: list[CaseOutcome]
    timings: dict[str, Any]
    config: dict[str, Any]

    def to_report(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.summary(),
            "mode": self.mode,
            "top_k": self.top_k,
            "retrieval_mode_used": self.retrieval_mode_used,
            "embedding_model": self.embedding_model,
            "metrics": self.metrics,
            "category_metrics": self.category_metrics,
            "unanswerable": self.unanswerable,
            "timings": self.timings,
            "config": self.config,
            "cases": [case.to_dict() for case in self.cases],
            "disclaimer": (
                "AI-authored dataset, human review NOT_VERIFIED. Retrieval metrics only; "
                "faithfulness and citation support are not measured by this run."
            ),
        }


def _contains(text: str, evidence: str) -> bool:
    return (
        unicodedata.normalize("NFC", evidence).casefold()
        in unicodedata.normalize("NFC", text).casefold()
    )


def _case_metrics(
    *,
    expected: int,
    ranks: list[int],
    k: int,
) -> CaseMetrics:
    matched = len(ranks)
    first_rank = min(ranks) if ranks else None
    return CaseMetrics(
        recall_at_k=round(recall_at_k(matched, expected), 4),
        precision_at_k=round(precision_at_k(matched, k), 4),
        hit_at_k=hit_at_k(matched),
        mrr=round(reciprocal_rank(first_rank), 4),
        ndcg_at_k=round(ndcg_from_ranks(ranks, expected, k), 4),
    )


def _category_summary(cases: list[CaseOutcome]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[CaseOutcome]] = {}
    for case in cases:
        if case.answerable and case.metrics is not None:
            grouped.setdefault(case.category, []).append(case)
    summary: dict[str, dict[str, Any]] = {}
    for category, items in sorted(grouped.items()):
        summary[category] = {
            "cases": len(items),
            "recall_at_k": mean(item.metrics.recall_at_k for item in items if item.metrics),
            "precision_at_k": mean(item.metrics.precision_at_k for item in items if item.metrics),
            "hit_at_k": mean(item.metrics.hit_at_k for item in items if item.metrics),
            "mrr": mean(item.metrics.mrr for item in items if item.metrics),
            "ndcg_at_k": mean(item.metrics.ndcg_at_k for item in items if item.metrics),
        }
    return summary


def _retrieved_summary(chunk: Any) -> dict[str, Any]:
    text = chunk.text.strip().replace("\n", " ")
    return {
        "chunk_id": str(chunk.chunk_id),
        "document_id": str(chunk.document_id),
        "filename": chunk.filename,
        "heading_path": list(chunk.heading_path),
        "score": round(float(chunk.score), 6),
        "source": chunk.source,
        "snippet": text[:240],
    }


async def evaluate_retrieval(
    retrieval: RetrievalService,
    *,
    workspace_id: uuid.UUID,
    kb_id: uuid.UUID,
    dataset: EvaluationDataset,
    mode: str,
    top_k: int,
    limit: int | None = None,
) -> EvaluationResult:
    cases = list(dataset.cases[:limit] if limit is not None else dataset.cases)
    outcomes: list[CaseOutcome] = []
    latencies: list[float] = []
    modes_used: dict[str, int] = {}
    started_all = time.perf_counter()

    for case in cases:
        started = time.perf_counter()
        outcome, _ = await retrieval.search(
            workspace_id=workspace_id,
            kb_id=kb_id,
            query=case.question,
            top_k=top_k,
            mode=mode,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        latencies.append(latency_ms)
        modes_used[outcome.mode_used] = modes_used.get(outcome.mode_used, 0) + 1

        chunks = outcome.chunks[:top_k]
        ranks: list[int] = []
        matched: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for rank, chunk in enumerate(chunks, start=1):
            for source in case.expected_sources:
                key = (source.filename, source.evidence)
                if key in seen:
                    continue
                if chunk.filename == source.filename and _contains(chunk.text, source.evidence):
                    seen.add(key)
                    ranks.append(rank)
                    matched.append(
                        {
                            "filename": source.filename,
                            "evidence": source.evidence,
                            "rank": rank,
                            "chunk_id": str(chunk.chunk_id),
                        }
                    )

        case_metrics: CaseMetrics | None = None
        if case.answerable:
            case_metrics = _case_metrics(expected=len(case.expected_sources), ranks=ranks, k=top_k)

        outcomes.append(
            CaseOutcome(
                id=case.id,
                question=case.question,
                category=case.category,
                answerable=case.answerable,
                expected_sources=[
                    {"filename": source.filename, "evidence": source.evidence}
                    for source in case.expected_sources
                ],
                matched_sources=matched,
                metrics=case_metrics,
                retrieved=[_retrieved_summary(chunk) for chunk in chunks],
                latency_ms=latency_ms,
                warnings=list(outcome.warnings),
            )
        )

    answerable = [item for item in outcomes if item.answerable and item.metrics is not None]
    unanswerable = [item for item in outcomes if not item.answerable]
    total_seconds = round(time.perf_counter() - started_all, 3)

    aggregate_metrics: dict[str, Any] = {
        "cases": len(outcomes),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "recall_at_k": mean(item.metrics.recall_at_k for item in answerable if item.metrics),
        "precision_at_k": mean(item.metrics.precision_at_k for item in answerable if item.metrics),
        "hit_at_k": mean(item.metrics.hit_at_k for item in answerable if item.metrics),
        "mrr": mean(item.metrics.mrr for item in answerable if item.metrics),
        "ndcg_at_k": mean(item.metrics.ndcg_at_k for item in answerable if item.metrics),
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "total_seconds": total_seconds,
    }

    returned_any = sum(1 for item in unanswerable if item.retrieved)
    unanswerable_summary = {
        "cases": len(unanswerable),
        "returned_any_results": returned_any,
        "returned_any_rate": round(returned_any / len(unanswerable), 4) if unanswerable else None,
        "note": (
            "Truy hồi luôn có thể trả ứng viên; tỷ lệ này chỉ ghi nhận hành vi, "
            "không phải độ chính xác từ chối. Từ chối được đo ở tầng sinh câu trả lời."
        ),
    }

    return EvaluationResult(
        dataset=dataset,
        mode=mode,
        top_k=top_k,
        retrieval_mode_used=", ".join(
            f"{name} ({count})" for name, count in sorted(modes_used.items())
        ),
        embedding_model=retrieval.embeddings.model_id,
        metrics=aggregate_metrics,
        category_metrics=_category_summary(outcomes),
        unanswerable=unanswerable_summary,
        cases=outcomes,
        timings={"total_seconds": total_seconds},
        config=retrieval.trace_config(top_k=top_k, mode=mode),
    )


def summarize_case(case: EvaluationCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "question": case.question,
        "category": case.category,
        "answerable": case.answerable,
        "expected_source_count": len(case.expected_sources),
        "prompt_injection": case.prompt_injection,
    }
