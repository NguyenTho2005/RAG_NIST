"""Format-specific parsers producing page/section provenance."""

from __future__ import annotations

import io
import re
from collections.abc import Iterable
from typing import Any

from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
from docx import Document as DocxDocument
from pypdf import PdfReader

from app.parsing.types import (
    EmptyDocumentError,
    OcrRequiredError,
    ParsedBlock,
    ParsedDocument,
    ParserError,
)
from app.retrieval.text import normalize_text

PARSER_VERSION = "2026.11.1"


def _add(
    blocks: list[ParsedBlock],
    text: str,
    *,
    kind: str = "paragraph",
    heading_level: int | None = None,
    page: int | None = None,
    table_id: str | None = None,
    table_row: int | None = None,
    table_header: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    cleaned = normalize_text(text)
    if not cleaned:
        return
    blocks.append(
        ParsedBlock(
            kind=kind,  # type: ignore[arg-type]
            text=cleaned,
            ordinal=len(blocks),
            heading_level=heading_level,
            page=page,
            table_id=table_id,
            table_row=table_row,
            table_header=table_header,
            metadata=metadata or {},
        )
    )


def _finalize(doc: ParsedDocument) -> ParsedDocument:
    cursor = 0
    for block in doc.blocks:
        block.char_start = cursor
        block.char_end = cursor + len(block.text)
        cursor = block.char_end + 2
    if not doc.blocks or sum(len(block.text.strip()) for block in doc.blocks) == 0:
        raise EmptyDocumentError()
    return doc


def parse_pdf(content: bytes, filename: str) -> ParsedDocument:
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ParserError("pdf_encrypted", "PDF được mã hóa và không thể đọc.") from exc
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
    except ParserError:
        raise
    except Exception as exc:
        raise ParserError("pdf_corrupted", "Không thể đọc cấu trúc PDF.") from exc

    total_chars = sum(len(normalize_text(text)) for text in pages)
    if total_chars < max(10, len(pages) * 10):
        raise OcrRequiredError()

    blocks: list[ParsedBlock] = []
    for page_number, raw_text in enumerate(pages, start=1):
        normalized = normalize_text(raw_text)
        if not normalized:
            continue
        paragraphs = [p for p in re.split(r"\n\s*\n", normalized) if p.strip()]
        if len(paragraphs) == 1 and "\n" in normalized:
            paragraphs = [line for line in normalized.split("\n") if line.strip()]
        for paragraph in paragraphs:
            _add(blocks, paragraph, page=page_number)

    metadata: dict[str, Any] = {}
    if reader.metadata:
        metadata = {
            key: str(value)
            for key, value in {
                "title": reader.metadata.title,
                "author": reader.metadata.author,
                "subject": reader.metadata.subject,
            }.items()
            if value
        }
    title = str(metadata.get("title") or filename)
    return _finalize(
        ParsedDocument(
            title=title,
            blocks=blocks,
            page_count=len(pages),
            metadata=metadata,
            parser_version=PARSER_VERSION,
        )
    )


def _docx_table_rows(table: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.rows:
        rows.append([normalize_text(cell.text) for cell in row.cells])
    return rows


def parse_docx(content: bytes, filename: str) -> ParsedDocument:
    try:
        document = DocxDocument(io.BytesIO(content))
    except Exception as exc:
        raise ParserError("docx_corrupted", "Không thể đọc cấu trúc DOCX.") from exc

    blocks: list[ParsedBlock] = []
    table_index = 0
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            text = normalize_text(paragraph.text)
            if not text:
                continue
            style = (paragraph.style.name if paragraph.style else "") or ""
            heading_level = None
            if style.lower().startswith("heading"):
                match = re.search(r"(\d+)", style)
                heading_level = int(match.group(1)) if match else 1
            elif style.lower() in {"title", "subtitle"}:
                heading_level = 1
            _add(
                blocks,
                text,
                kind="heading" if heading_level else "paragraph",
                heading_level=heading_level,
            )
        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            rows = [row for row in _docx_table_rows(table) if any(cell for cell in row)]
            if not rows:
                continue
            table_id = f"tbl-{table_index}"
            table_index += 1
            header = rows[0]
            for row_number, row in enumerate(rows):
                _add(
                    blocks,
                    " | ".join(cell for cell in row if cell),
                    kind="table",
                    table_id=table_id,
                    table_row=row_number,
                    table_header=header,
                    metadata={"rows": rows, "row_index": row_number, "header": header},
                )

    title = filename
    try:
        core_title = document.core_properties.title
        if core_title:
            title = normalize_text(core_title)
    except Exception:  # optional metadata must never block parsing
        core_title = None
    return _finalize(
        ParsedDocument(
            title=title,
            blocks=blocks,
            page_count=None,
            metadata={"format": "docx"},
            parser_version=PARSER_VERSION,
        )
    )


def parse_text(content: bytes, filename: str) -> ParsedDocument:
    match = from_bytes(content).best()
    if match is None:
        raise ParserError("text_decode_failed", "Không thể xác định bảng mã văn bản.")
    text = str(match)
    normalized = normalize_text(text)
    blocks: list[ParsedBlock] = []
    for paragraph in re.split(r"\n\s*\n", normalized):
        for line in paragraph.split("\n"):
            if line.strip():
                _add(blocks, line)
    return _finalize(
        ParsedDocument(
            title=filename,
            blocks=blocks,
            page_count=None,
            metadata={"encoding": match.encoding or "unknown"},
            parser_version=PARSER_VERSION,
        )
    )


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_FENCE = re.compile(r"^```")
_MD_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")


def parse_markdown(content: bytes, filename: str) -> ParsedDocument:
    match = from_bytes(content).best()
    text = str(match) if match else content.decode("utf-8", errors="replace")
    lines = text.splitlines()
    blocks: list[ParsedBlock] = []
    paragraph: list[str] = []
    table_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False
    table_index = 0

    def flush_paragraph() -> None:
        if paragraph:
            _add(blocks, " ".join(part.strip() for part in paragraph))
            paragraph.clear()

    def flush_table() -> None:
        nonlocal table_index
        if not table_lines:
            return
        rows: list[list[str]] = []
        for line in table_lines:
            cells = [normalize_text(cell) for cell in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells if cell):
                continue
            rows.append(cells)
        table_lines.clear()
        if rows:
            table_id = f"tbl-{table_index}"
            table_index += 1
            header = rows[0]
            for row_number, row in enumerate(rows):
                _add(
                    blocks,
                    " | ".join(cell for cell in row if cell),
                    kind="table",
                    table_id=table_id,
                    table_row=row_number,
                    table_header=header,
                    metadata={"rows": rows, "row_index": row_number, "header": header},
                )

    for line in lines:
        if _MD_FENCE.match(line.strip()):
            flush_paragraph()
            flush_table()
            if in_code:
                _add(blocks, "\n".join(code_lines), kind="code")
                code_lines.clear()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        heading = _MD_HEADING.match(line)
        if heading:
            flush_paragraph()
            flush_table()
            _add(blocks, heading.group(2), kind="heading", heading_level=len(heading.group(1)))
            continue
        if line.strip().startswith("|") and line.strip().endswith("|"):
            flush_paragraph()
            table_lines.append(line.strip())
            continue
        flush_table()
        list_match = _MD_LIST.match(line)
        if list_match:
            flush_paragraph()
            _add(blocks, list_match.group(1), kind="list_item")
            continue
        if not line.strip():
            flush_paragraph()
            continue
        paragraph.append(line)
    flush_paragraph()
    flush_table()
    if code_lines:
        _add(blocks, "\n".join(code_lines), kind="code")

    # Skip YAML front matter as content but keep it as metadata is out of scope.
    return _finalize(
        ParsedDocument(
            title=filename,
            blocks=blocks,
            page_count=None,
            metadata={"format": "markdown"},
            parser_version=PARSER_VERSION,
        )
    )


def _html_table_rows(table: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        cells = [normalize_text(cell.get_text(" ")) for cell in row.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    return rows


def parse_html(content: bytes, filename: str) -> ParsedDocument:
    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form", "nav", "footer"]):
        tag.decompose()
    blocks: list[ParsedBlock] = []
    table_index = 0
    seen_tables: set[int] = set()

    body = soup.body or soup
    for element in body.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table", "blockquote"]
    ):
        if element.find_parent("table") is not None and element.name != "table":
            continue
        if element.name == "table":
            rows = _html_table_rows(element)
            if not rows:
                continue
            table_id = f"tbl-{table_index}"
            table_index += 1
            seen_tables.add(id(element))
            header = rows[0]
            for row_number, row in enumerate(rows):
                _add(
                    blocks,
                    " | ".join(cell for cell in row if cell),
                    kind="table",
                    table_id=table_id,
                    table_row=row_number,
                    table_header=header,
                    metadata={"rows": rows, "row_index": row_number, "header": header},
                )
            continue
        text = normalize_text(element.get_text(" "))
        if not text:
            continue
        if element.name.startswith("h"):
            _add(blocks, text, kind="heading", heading_level=int(element.name[1]))
        elif element.name == "li":
            _add(blocks, text, kind="list_item")
        elif element.name == "pre":
            _add(blocks, text, kind="code")
        else:
            _add(blocks, text)

    title_tag = soup.title
    title = normalize_text(title_tag.get_text()) if title_tag else filename
    return _finalize(
        ParsedDocument(
            title=title,
            blocks=blocks,
            page_count=None,
            metadata={"format": "html", "language": soup.html.get("lang") if soup.html else None},
            parser_version=PARSER_VERSION,
        )
    )


def parse_document(content: bytes, kind: str, filename: str) -> ParsedDocument:
    parsers = {
        "pdf": parse_pdf,
        "docx": parse_docx,
        "text": parse_text,
        "markdown": parse_markdown,
        "html": parse_html,
    }
    parser = parsers.get(kind)
    if parser is None:
        raise ParserError("unsupported_parser", f"Không có parser cho định dạng '{kind}'.")
    result = parser(content, filename)
    if not result.blocks:
        raise EmptyDocumentError()
    return result


def iter_sections(blocks: Iterable[ParsedBlock]) -> list[tuple[str, int, list[ParsedBlock]]]:
    """Group blocks by heading path (exported for chunker tests)."""
    sections: list[tuple[str, int, list[ParsedBlock]]] = []
    current_title = ""
    current_level = 0
    current: list[ParsedBlock] = []
    for block in blocks:
        if block.is_heading:
            if current:
                sections.append((current_title, current_level, current))
            current_title = block.text
            current_level = block.heading_level or 1
            current = [block]
        else:
            current.append(block)
    if current:
        sections.append((current_title, current_level, current))
    return sections
