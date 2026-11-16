"""Versioned evaluation datasets and corpus loading.

The bundled dataset is AI-authored and has not been human-reviewed. That status
is part of the dataset metadata and is repeated in every report so synthetic
labels are never presented as gold data.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.core.errors import NotFoundError, ValidationFailedError

Category = Literal[
    "fact",
    "multi_hop",
    "terminology",
    "prompt_injection",
    "unanswerable",
]

_DATASET_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_CATEGORIES = {"fact", "multi_hop", "terminology", "prompt_injection", "unanswerable"}


@dataclass(frozen=True, slots=True)
class ExpectedSource:
    filename: str
    evidence: str


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    question: str
    category: str
    answerable: bool
    reference_answer: str
    expected_sources: tuple[ExpectedSource, ...]
    review_status: str
    prompt_injection: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    dataset_id: str
    title: str
    description: str
    version: str
    path: Path
    sha256: str
    cases: tuple[EvaluationCase, ...]
    metadata: dict[str, Any]

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def answerable_count(self) -> int:
        return sum(1 for case in self.cases if case.answerable)

    @property
    def unanswerable_count(self) -> int:
        return sum(1 for case in self.cases if not case.answerable)

    @property
    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.category] = counts.get(case.category, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.dataset_id,
            "title": self.title,
            "description": self.description,
            "version": self.version,
            "sha256": self.sha256,
            "case_count": self.case_count,
            "answerable_count": self.answerable_count,
            "unanswerable_count": self.unanswerable_count,
            "categories": self.categories,
            "review_status": self.metadata.get("review_status", "NOT_VERIFIED"),
            "authoring": self.metadata.get("authoring", "AI-authored draft"),
            "languages": self.metadata.get("languages", ["vi"]),
            "limitations": self.metadata.get("limitations", []),
        }


def dataset_dir(settings_root: Path | str) -> Path:
    return Path(settings_root) / "datasets"


def corpus_dir(settings_root: Path | str) -> Path:
    return Path(settings_root) / "corpus"


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _validate_case(raw: dict[str, Any], line_no: int) -> EvaluationCase:
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValidationFailedError(f"Dòng {line_no}: thiếu id câu hỏi.")
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValidationFailedError(f"Dòng {line_no}: câu hỏi rỗng.")
    category = raw.get("category")
    if category not in _CATEGORIES:
        raise ValidationFailedError(
            f"Dòng {line_no}: category '{category}' không hợp lệ.",
            details={"allowed": sorted(_CATEGORIES)},
        )
    answerable = raw.get("answerable")
    if not isinstance(answerable, bool):
        raise ValidationFailedError(f"Dòng {line_no}: answerable phải là boolean.")
    reference_answer = raw.get("reference_answer")
    if not isinstance(reference_answer, str) or not reference_answer.strip():
        raise ValidationFailedError(f"Dòng {line_no}: thiếu reference_answer.")
    sources_raw = raw.get("expected_sources")
    if not isinstance(sources_raw, list):
        raise ValidationFailedError(f"Dòng {line_no}: expected_sources phải là mảng.")
    sources: list[ExpectedSource] = []
    for source in sources_raw:
        if not isinstance(source, dict):
            raise ValidationFailedError(f"Dòng {line_no}: expected source không hợp lệ.")
        filename = source.get("filename")
        evidence = source.get("evidence")
        if not isinstance(filename, str) or not filename.strip():
            raise ValidationFailedError(f"Dòng {line_no}: expected source thiếu filename.")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValidationFailedError(f"Dòng {line_no}: expected source thiếu evidence.")
        sources.append(ExpectedSource(filename=filename.strip(), evidence=_normalize(evidence)))
    if answerable and not sources:
        raise ValidationFailedError(f"Dòng {line_no}: câu trả lời được phải có expected_sources.")
    if not answerable and sources:
        raise ValidationFailedError(
            f"Dòng {line_no}: câu không trả lời được không nên có expected_sources."
        )
    review_status = raw.get("review_status", "ai_authored_not_human_reviewed")
    if not isinstance(review_status, str):
        raise ValidationFailedError(f"Dòng {line_no}: review_status không hợp lệ.")
    return EvaluationCase(
        id=case_id.strip(),
        question=question.strip(),
        category=category,
        answerable=answerable,
        reference_answer=reference_answer.strip(),
        expected_sources=tuple(sources),
        review_status=review_status,
        prompt_injection=bool(raw.get("prompt_injection", False)),
    )


def load_dataset(root: Path | str, dataset_id: str) -> EvaluationDataset:
    if not _DATASET_ID.fullmatch(dataset_id):
        raise ValidationFailedError("dataset_id không hợp lệ.")
    directory = dataset_dir(root)
    path = directory / f"{dataset_id}.jsonl"
    if not path.is_file():
        raise NotFoundError(f"Không tìm thấy dataset '{dataset_id}'.")
    payload = path.read_bytes()
    seen: set[str] = set()
    cases: list[EvaluationCase] = []
    for line_no, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationFailedError(
                f"Dòng {line_no}: JSON không hợp lệ.",
                details={"error": str(exc)},
            ) from exc
        if not isinstance(raw, dict):
            raise ValidationFailedError(f"Dòng {line_no}: JSON phải là object.")
        case = _validate_case(raw, line_no)
        if case.id in seen:
            raise ValidationFailedError(f"Trùng id câu hỏi: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValidationFailedError(f"Dataset '{dataset_id}' không có câu hỏi.")

    meta_path = directory / f"{dataset_id}.meta.json"
    metadata: dict[str, Any] = {}
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValidationFailedError(f"Metadata dataset '{dataset_id}' không hợp lệ.")
    return EvaluationDataset(
        dataset_id=dataset_id,
        title=str(metadata.get("title", dataset_id)),
        description=str(metadata.get("description", "")),
        version=str(metadata.get("version", "0")),
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        cases=tuple(cases),
        metadata=metadata,
    )


def list_datasets(root: Path | str) -> list[dict[str, Any]]:
    directory = dataset_dir(root)
    if not directory.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        try:
            summaries.append(load_dataset(root, path.stem).summary())
        except ValidationFailedError:
            continue
    return summaries


def load_corpus(root: Path | str) -> dict[str, str]:
    """Return filename -> NFC text for the bundled evaluation corpus."""
    directory = corpus_dir(root)
    if not directory.is_dir():
        return {}
    return {
        path.name: _normalize(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.md"))
    }
