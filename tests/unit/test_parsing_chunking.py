"""Unit tests for Vietnamese text handling, parsers, and chunking."""

from __future__ import annotations

import pytest
from tests.fixtures.builders import make_docx, make_pdf

from app.chunking.chunker import StructureAwareChunker, build_sections
from app.core.errors import UnsupportedMediaError
from app.parsing.detect import detect_file_type
from app.parsing.parsers import (
    parse_docx,
    parse_html,
    parse_markdown,
    parse_pdf,
    parse_text,
)
from app.parsing.types import OcrRequiredError, ParserError
from app.retrieval.text import count_tokens, normalize_text, tokenize


def test_nfc_and_diacritics_preserved() -> None:
    decomposed = "Ma\u0303 so\u0302\u0301 123"
    normalized = normalize_text(decomposed)
    assert normalized == "Mã số 123"
    assert "Mã" in normalized


def test_zero_width_and_whitespace_normalized() -> None:
    assert normalize_text("A\u200bB\u00a0 C\r\n\r\n\r\nD") == "AB C\n\nD"


def test_vietnamese_tokenizer_keeps_diacritics_and_identifiers() -> None:
    terms = tokenize("Thông tư số 30/2020/TT-BTC quy định mã số thuế")
    assert "thông" in terms
    assert "mã" in terms
    assert "ma" not in terms
    assert any(term.startswith("id:30") or "30" in term for term in terms)
    assert "mã số" in terms
    assert count_tokens("Xin chào Việt Nam") > 0


def test_text_parser_windows1258_roundtrip() -> None:
    content = "Chính sách nhân sự\n\nNhân viên có 12 ngày phép.".encode()
    parsed = parse_text(content, "policy.txt")
    assert len(parsed.blocks) == 2
    assert parsed.blocks[0].char_start == 0
    assert (
        parsed.text[parsed.blocks[0].char_start : parsed.blocks[0].char_end]
        == parsed.blocks[0].text
    )


def test_pdf_parser_preserves_pages() -> None:
    parsed = parse_pdf(make_pdf(["Page one content", "Page two content"]), "sample.pdf")
    assert parsed.page_count == 2
    pages = {block.page for block in parsed.blocks}
    assert pages == {1, 2}
    assert "Page two" in parsed.text


def test_scanned_pdf_fails_loudly() -> None:
    with pytest.raises(OcrRequiredError):
        parse_pdf(make_pdf(["", ""]), "scan.pdf")


def test_markdown_sections_and_tables() -> None:
    markdown = """# Chính sách

Một đoạn văn.

## Bảng lương

| Chức danh | Lương |
| --- | --- |
| Kế toán | 20 |
| Kỹ sư | 30 |
""".encode()
    parsed = parse_markdown(markdown, "policy.md")
    sections = build_sections(parsed)
    assert [section.title for section in sections] == ["Chính sách", "Bảng lương"]
    assert sections[1].path == ["Chính sách", "Bảng lương"]
    table_blocks = [block for block in parsed.blocks if block.kind == "table"]
    assert len(table_blocks) == 3
    assert table_blocks[0].table_header == ["Chức danh", "Lương"]


def test_docx_headings_and_table() -> None:
    parsed = parse_docx(make_docx(), "policy.docx")
    headings = [block.text for block in parsed.blocks if block.kind == "heading"]
    assert headings == ["Chính sách nghỉ phép", "Quy trình phê duyệt"]
    assert any(block.kind == "table" for block in parsed.blocks)


def test_html_strips_scripts_and_keeps_structure() -> None:
    html = """<html lang="vi"><head><title>Tiêu đề</title><script>alert(1)</script></head>
    <body><h1>Chính sách</h1><p>Nội dung <b>quan trọng</b>.</p>
    <ul><li>Mục một</li></ul>
    <table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
    <script>evil()</script></body></html>""".encode()
    parsed = parse_html(html, "page.html")
    assert parsed.title == "Tiêu đề"
    assert "alert" not in parsed.text
    assert "evil" not in parsed.text
    assert any(block.kind == "table" for block in parsed.blocks)
    assert parsed.metadata["language"] == "vi"


def test_detect_rejects_spoofed_pdf() -> None:
    with pytest.raises(UnsupportedMediaError):
        detect_file_type("fake.pdf", b"not a pdf")
    with pytest.raises(UnsupportedMediaError):
        detect_file_type("evil.exe", b"MZ")


def test_detect_docx_zip_without_word_structure() -> None:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "hello")
    with pytest.raises(UnsupportedMediaError):
        detect_file_type("fake.docx", buffer.getvalue())


def test_chunker_respects_limits_and_provenance() -> None:
    paragraphs = "\n\n".join(
        f"Điều {index}. " + ("Nội dung quy định chi tiết về nghĩa vụ và quyền hạn. " * 12)
        for index in range(1, 12)
    )
    markdown = f"# Quy chế nội bộ\n\n## Nghĩa vụ\n\n{paragraphs}\n".encode()
    parsed = parse_markdown(markdown, "rules.md")
    chunker = StructureAwareChunker(target_tokens=120, max_tokens=180, overlap_tokens=20)
    result = chunker.chunk(parsed)
    children = [chunk for chunk in result.chunks if chunk.kind == "child"]
    assert len(children) > 3
    assert all(chunk.token_count <= 180 for chunk in children)
    assert all(chunk.heading_path == ["Quy chế nội bộ", "Nghĩa vụ"] for chunk in children)
    document_text = parsed.text
    for child in children:
        assert child.source_spans
        for span in child.source_spans:
            assert 0 <= span["char_start"] <= span["char_end"] <= len(document_text) + 2
    parents = [chunk for chunk in result.chunks if chunk.kind == "parent"]
    assert parents
    for child in children:
        if child.parent_index is not None:
            assert result.chunks[child.parent_index].kind == "parent"


def test_chunker_repeats_table_header_and_keeps_rows() -> None:
    rows = "\n".join(f"| Nhóm {index} | {index * 10} |" for index in range(40))
    markdown = ("# Bảng lương\n\n| Nhóm | Giá trị |\n| --- | --- |\n" + rows + "\n").encode()
    parsed = parse_markdown(markdown, "table.md")
    chunker = StructureAwareChunker(target_tokens=80, max_tokens=120, overlap_tokens=10)
    result = chunker.chunk(parsed)
    children = [chunk for chunk in result.chunks if chunk.kind == "child"]
    table_chunks = [chunk for chunk in children if "|" in chunk.text]
    assert len(table_chunks) > 1
    for chunk in table_chunks:
        assert "Nhóm | Giá trị" in chunk.text
        # Every emitted row is complete: no line without a separator cell.
        for line in chunk.text.splitlines():
            if "|" in line and not line.startswith("Bảng lương"):
                assert line.count("|") == 1


def test_chunker_handles_pathological_single_block() -> None:
    parsed = parse_text(("từ " * 2000).encode(), "long.txt")
    chunker = StructureAwareChunker(target_tokens=100, max_tokens=120, overlap_tokens=10)
    result = chunker.chunk(parsed)
    children = [chunk for chunk in result.chunks if chunk.kind == "child"]
    assert children
    assert all(chunk.token_count <= 140 for chunk in children)
    parents = [chunk for chunk in result.chunks if chunk.kind == "parent"]
    assert all(chunk.token_count <= 1300 for chunk in parents)


def test_empty_text_rejected() -> None:
    with pytest.raises(ParserError):
        parse_text(b"   \n\n  ", "empty.txt")
