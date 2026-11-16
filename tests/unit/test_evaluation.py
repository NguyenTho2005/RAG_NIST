"""Unit tests for retrieval metrics and dataset validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.errors import ValidationFailedError
from app.evaluation.dataset import list_datasets, load_corpus, load_dataset
from app.evaluation.metrics import (
    dcg,
    hit_at_k,
    mean,
    ndcg_at_k,
    ndcg_from_ranks,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_precision_hit_and_mrr() -> None:
    assert recall_at_k(2, 4) == 0.5
    assert recall_at_k(0, 4) == 0.0
    assert recall_at_k(1, 0) == 0.0
    assert precision_at_k(2, 10) == 0.2
    assert precision_at_k(1, 0) == 0.0
    assert hit_at_k(1) == 1.0
    assert hit_at_k(0) == 0.0
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0


def test_ndcg_from_ranks_discounts_late_hits() -> None:
    ideal = ndcg_from_ranks([1], 1, 5)
    late = ndcg_from_ranks([3], 1, 5)
    assert ideal == 1.0
    assert 0.0 < late < ideal
    assert ndcg_from_ranks([1, 2], 2, 5) == 1.0
    assert ndcg_from_ranks([6], 1, 5) == 0.0
    assert ndcg_from_ranks([], 1, 5) == 0.0
    assert ndcg_from_ranks([1], 0, 5) == 0.0


def test_ndcg_flags_and_dcg() -> None:
    assert dcg([1.0, 1.0]) > dcg([1.0, 0.0])
    assert ndcg_at_k([True, False], expected=2, k=2) == pytest.approx(0.6131, abs=1e-3)
    assert ndcg_at_k([], expected=0, k=2) == 0.0


def test_percentiles_and_mean() -> None:
    assert percentile([], 0.5) is None
    assert percentile([7.0], 0.95) == 7.0
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert mean([]) is None
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_bundled_dataset_is_large_and_honest() -> None:
    root = Path(__file__).resolve().parents[2] / "evaluation"
    dataset = load_dataset(root, "grc-vi-v1")
    assert dataset.case_count >= 50
    assert dataset.answerable_count >= 40
    assert dataset.unanswerable_count >= 5
    assert dataset.summary()["review_status"] == "NOT_VERIFIED"
    assert dataset.summary()["authoring"].startswith("AI-authored")
    categories = dataset.categories
    for expected in ("fact", "multi_hop", "terminology", "prompt_injection", "unanswerable"):
        assert categories[expected] >= 2
    assert len({case.id for case in dataset.cases}) == dataset.case_count
    corpus = load_corpus(root)
    assert len(corpus) == 8
    for case in dataset.cases:
        for source in case.expected_sources:
            assert source.filename in corpus
            assert source.evidence in corpus[source.filename]
    assert list_datasets(root)[0]["id"] == "grc-vi-v1"


def test_dataset_validation_rejects_bad_records(tmp_path: Path) -> None:
    directory = tmp_path / "datasets"
    directory.mkdir()
    (directory / "bad.jsonl").write_text(
        json.dumps(
            {
                "id": "x",
                "question": "Câu hỏi?",
                "category": "fact",
                "answerable": True,
                "reference_answer": "Trả lời",
                "expected_sources": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationFailedError):
        load_dataset(tmp_path, "bad")

    (directory / "dup.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "id": "x",
                    "question": "Câu hỏi?",
                    "category": "fact",
                    "answerable": True,
                    "reference_answer": "Trả lời",
                    "expected_sources": [{"filename": "a.md", "evidence": "a"}],
                }
            )
            for _ in range(2)
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationFailedError):
        load_dataset(tmp_path, "dup")

    with pytest.raises(ValidationFailedError):
        load_dataset(tmp_path, "../escape")
