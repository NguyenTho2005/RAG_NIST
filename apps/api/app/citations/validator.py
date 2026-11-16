"""Citation extraction and validation.

The validator proves that a produced answer points at sources that were
actually supplied to the model. It does **not** prove factual entailment: that
requires evaluation/human review, which is reported separately.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.context import ContextSource, source_metadata

_MARKER_RE = re.compile(r"\[\s*S(\d+)\s*\]", re.IGNORECASE)
_QUOTE_LIMIT = 320


@dataclass(slots=True)
class CitationValidation:
    answer: str
    citations: list[dict[str, Any]]
    invalid_markers: list[str] = field(default_factory=list)
    missing: bool = False
    citation_status: str = "missing"
    grounded: bool = False
    warnings: list[str] = field(default_factory=list)


def _quote(source: ContextSource) -> str:
    text = source.text.strip()
    return text if len(text) <= _QUOTE_LIMIT else text[: _QUOTE_LIMIT - 1].rstrip() + "…"


def validate_answer(answer: str, sources: Sequence[ContextSource]) -> CitationValidation:
    by_index = {source.index: source for source in sources}
    valid_indices: list[int] = []
    invalid_markers: list[str] = []
    for match in _MARKER_RE.finditer(answer):
        index = int(match.group(1))
        marker = match.group(0)
        if index not in by_index:
            if marker not in invalid_markers:
                invalid_markers.append(marker)
            continue
        if index not in valid_indices:
            valid_indices.append(index)

    def _replace(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in by_index else ""

    cleaned = _MARKER_RE.sub(_replace, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    citations = [
        {**source_metadata(by_index[index]), "quote": _quote(by_index[index])}
        for index in sorted(valid_indices)
    ]
    warnings: list[str] = []
    if invalid_markers:
        warnings.append("Một số trích dẫn không tồn tại trong ngữ cảnh đã gửi và đã bị loại bỏ.")
    if not citations:
        warnings.append("Câu trả lời không có trích dẫn nguồn hợp lệ.")
        status = "invalid" if invalid_markers else "missing"
    else:
        status = "valid"
    return CitationValidation(
        answer=cleaned,
        citations=citations,
        invalid_markers=invalid_markers,
        missing=not citations,
        citation_status=status,
        grounded=status == "valid",
        warnings=warnings,
    )


__all__ = ["CitationValidation", "validate_answer"]
