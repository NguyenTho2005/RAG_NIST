"""Deterministic binary fixtures generated at test time."""

from __future__ import annotations

import io


def make_pdf(pages: list[str]) -> bytes:
    """Build a small valid PDF with Helvetica text, without extra dependencies."""
    objects: list[bytes] = []

    def add(content: bytes) -> int:
        objects.append(content)
        return len(objects)

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    content_ids: list[int] = []
    for text in pages:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1", errors="replace")
        content_ids.append(
            add(
                b"<< /Length "
                + str(len(stream)).encode()
                + b" >>\nstream\n"
                + stream
                + b"\nendstream"
            )
        )
        page_ids.append(0)

    pages_id = len(objects) + len(pages) + 1
    for index, content_id in enumerate(content_ids):
        page_ids[index] = add(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
            ).encode()
        )
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    buffer = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number, body in enumerate(objects, start=1):
        offsets[number] = len(buffer)
        buffer += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_offset = len(buffer)
    buffer += f"xref\n0 {len(objects) + 1}\n".encode()
    buffer += b"0000000000 65535 f \n"
    for number in range(1, len(objects) + 1):
        buffer += f"{offsets[number]:010d} 00000 n \n".encode()
    buffer += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return bytes(buffer)


def make_docx() -> bytes:
    from docx import Document

    document = Document()
    document.add_heading("Chính sách nghỉ phép", level=1)
    document.add_paragraph("Nhân viên được nghỉ 12 ngày mỗi năm.")
    document.add_heading("Quy trình phê duyệt", level=2)
    document.add_paragraph("Đơn nghỉ phép phải được quản lý trực tiếp phê duyệt trước 3 ngày.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Loại nghỉ"
    table.cell(0, 1).text = "Số ngày"
    table.cell(1, 0).text = "Nghỉ phép năm"
    table.cell(1, 1).text = "12"
    stream = io.BytesIO()
    document.save(stream)
    return stream.getvalue()
