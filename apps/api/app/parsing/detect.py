"""File type and MIME signature validation."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from app.core.errors import UnsupportedMediaError

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm"}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
HTML_EXTENSIONS = {".html", ".htm"}


@dataclass(frozen=True, slots=True)
class DetectedType:
    extension: str
    mime_type: str
    kind: str


def safe_filename(filename: str) -> str:
    name = Path(filename).name.replace("\x00", "").strip()
    if not name or name in {".", ".."}:
        raise UnsupportedMediaError("Tên tệp không hợp lệ.")
    return name[:500]


def detect_file_type(
    filename: str, content: bytes, declared_mime: str | None = None
) -> DetectedType:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedMediaError(
            f"Định dạng '{suffix or 'không rõ'}' không được hỗ trợ.",
            details={"supported": sorted(SUPPORTED_EXTENSIONS)},
        )
    if len(content) < 4:
        raise UnsupportedMediaError("Tệp quá nhỏ hoặc rỗng.")
    if declared_mime:
        declared = declared_mime.split(";")[0].strip().lower()
        if declared in {"application/octet-stream", "binary/octet-stream", ""}:
            declared = ""
    else:
        declared = ""

    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise UnsupportedMediaError("Nội dung tệp không phải PDF hợp lệ.")
        return DetectedType(suffix, "application/pdf", "pdf")

    if suffix == ".docx":
        if not content.startswith(b"PK\x03\x04"):
            raise UnsupportedMediaError("Nội dung tệp không phải DOCX hợp lệ.")
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                names = set(archive.namelist())
                if "word/document.xml" not in names or "[Content_Types].xml" not in names:
                    raise UnsupportedMediaError("Tệp ZIP không chứa cấu trúc DOCX.")
        except zipfile.BadZipFile as exc:
            raise UnsupportedMediaError("Tệp DOCX bị hỏng.") from exc
        return DetectedType(
            suffix,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "docx",
        )

    if suffix in HTML_EXTENSIONS:
        sample = content[:2048].lstrip().lower()
        if not sample.startswith((b"<!doctype html", b"<html", b"<head", b"<body", b"<div", b"<p")):
            raise UnsupportedMediaError("Nội dung không giống HTML hợp lệ.")
        return DetectedType(suffix, "text/html", "html")

    # Plain text and Markdown: reject NUL bytes and undecodable binary blobs.
    if b"\x00" in content[:4096]:
        raise UnsupportedMediaError("Tệp văn bản chứa dữ liệu nhị phân.")
    try:
        content[:65536].decode("utf-8")
    except UnicodeDecodeError:
        # charset-normalizer handles legacy encodings later; try latin-1 as a
        # conservative decodability gate.
        try:
            content.decode("latin-1")
        except Exception as exc:  # pragma: no cover - latin-1 never fails
            raise UnsupportedMediaError("Không thể giải mã tệp văn bản.") from exc
    mime = "text/markdown" if suffix in {".md", ".markdown"} else "text/plain"
    return DetectedType(suffix, mime, "markdown" if mime == "text/markdown" else "text")
