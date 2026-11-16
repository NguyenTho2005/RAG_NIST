"""Atomic upload storage with traversal-safe relative paths."""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from pathlib import Path

from app.core.config import Settings
from app.core.errors import IngestionError


def relative_storage_path(
    workspace_id: uuid.UUID, document_id: uuid.UUID, sha256: str, filename: str
) -> str:
    suffix = Path(filename).suffix.lower()
    safe = Path(filename).stem[:80].replace("/", "_").replace("\\", "_") or "document"
    return f"uploads/{workspace_id}/{document_id}/{sha256[:16]}_{safe}{suffix}"


def resolve_path(settings: Settings, relative_path: str) -> Path:
    root = settings.data_dir.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise IngestionError("Đường dẫn lưu trữ không hợp lệ.", code="unsafe_storage_path")
    return candidate


def write_atomic(settings: Settings, relative_path: str, content: bytes) -> str:
    destination = resolve_path(settings, relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".upload-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return relative_path


def read_verified(settings: Settings, relative_path: str, expected_sha256: str) -> bytes:
    path = resolve_path(settings, relative_path)
    if not path.is_file():
        raise IngestionError("Không tìm thấy tệp nguồn đã lưu.", code="source_file_missing")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise IngestionError(
            "Tệp nguồn đã thay đổi so với bản ghi (hash mismatch).",
            code="source_hash_mismatch",
        )
    return data


def delete_file(settings: Settings, relative_path: str) -> None:
    try:
        path = resolve_path(settings, relative_path)
    except IngestionError:
        return
    path.unlink(missing_ok=True)
