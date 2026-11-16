"""Typed parser contracts shared by every document format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BlockKind = Literal["heading", "paragraph", "list_item", "table", "code", "caption"]


@dataclass(slots=True)
class ParsedBlock:
    kind: BlockKind
    text: str
    ordinal: int
    heading_level: int | None = None
    page: int | None = None
    char_start: int = 0
    char_end: int = 0
    table_id: str | None = None
    table_row: int | None = None
    table_header: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_heading(self) -> bool:
        return self.kind == "heading"


@dataclass(slots=True)
class ParsedDocument:
    title: str
    blocks: list[ParsedBlock]
    page_count: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    parser_version: str = "1"

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)


class ParserError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class EmptyDocumentError(ParserError):
    def __init__(self, detail: str = "Tài liệu không có nội dung văn bản trích xuất được.") -> None:
        super().__init__("empty_document", detail)


class OcrRequiredError(ParserError):
    def __init__(self) -> None:
        super().__init__(
            "ocr_required",
            "PDF có quá ít văn bản trích xuất được (có thể là bản scan). "
            "Bật pipeline OCR tùy chọn trước khi index; hệ thống không tạo index rỗng.",
        )
