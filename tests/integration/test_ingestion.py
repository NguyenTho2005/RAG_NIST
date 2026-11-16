"""End-to-end ingestion tests against real PostgreSQL and Qdrant.

Embeddings use an explicitly labelled deterministic test double; parsing,
chunking, persistence, FTS terms, job state, deletion, and Qdrant operations are
real.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from tests.fixtures.builders import make_pdf

from app.ingestion.storage import resolve_path
from app.parsing.parsers import parse_markdown
from app.persistence.models import Chunk, DocumentRevision

pytestmark = pytest.mark.integration

MARKDOWN = """# Quy chế nghỉ phép

## Phạm vi áp dụng

Quy chế này áp dụng cho toàn bộ nhân viên chính thức và nhân viên thử việc.

## Số ngày nghỉ

Nhân viên được nghỉ 12 ngày phép năm. Số ngày nghỉ tăng thêm một ngày cho mỗi
năm làm việc liên tục, tối đa 18 ngày.

## Quy trình phê duyệt

Nhân viên gửi đơn trước ít nhất 3 ngày làm việc. Quản lý trực tiếp phê duyệt
trên hệ thống nhân sự. Trường hợp khẩn cấp có thể bổ sung sau nhưng phải có
xác nhận của trưởng phòng.

## Bảng tóm tắt

| Loại nghỉ | Số ngày | Ghi chú |
| --- | --- | --- |
| Phép năm | 12 | Tăng theo thâm niên |
| Nghỉ ốm | 10 | Cần giấy xác nhận |
| Nghỉ thai sản | 180 | Theo quy định |

## Điều khoản khác

Mọi thay đổi về quy chế phải được thông báo trước 15 ngày. Dữ liệu cá nhân
trong đơn nghỉ phép được bảo mật theo chính sách bảo vệ dữ liệu.
"""


async def wait_job(
    client: AsyncClient, headers: dict[str, str], job_id: str, limit_seconds: float = 30
) -> dict:
    deadline = time.monotonic() + limit_seconds
    last = None
    while time.monotonic() < deadline:
        response = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in {"succeeded", "failed", "cancelled"}:
            return last
        await asyncio.sleep(0.05)
    raise AssertionError(f"job did not finish: {last}")


async def upload(
    client: AsyncClient,
    headers: dict[str, str],
    kb_id: str,
    filename: str,
    content: bytes,
    mime: str,
) -> dict:
    response = await client.post(
        "/api/v1/documents",
        headers=headers,
        data={"knowledge_base_id": kb_id},
        files={"file": (filename, content, mime)},
    )
    assert response.status_code == 202, response.text
    return response.json()


async def active_chunks(db, document_id: uuid.UUID) -> list[Chunk]:
    async with db.sessionmaker() as session:
        return (
            (
                await session.execute(
                    select(Chunk).where(Chunk.document_id == document_id, Chunk.active.is_(True))
                )
            )
            .scalars()
            .all()
        )


async def test_upload_ready_provenance_and_vectors(
    client, auth_headers, kb, fake_runtime, db
) -> None:
    body = await upload(
        client, auth_headers, str(kb.id), "quy-che.md", MARKDOWN.encode(), "text/markdown"
    )
    document_id = body["document"]["id"]
    job = await wait_job(client, auth_headers, body["job_id"])
    assert job["status"] == "succeeded", job
    assert job["stage"] == "done"
    assert job["result"]["child_chunk_count"] > 0

    detail = (await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)).json()
    assert detail["status"] == "ready"
    assert detail["chunk_count"] > 0
    assert detail["active_revision_number"] == 1
    assert detail["revisions"][0]["status"] == "ready"

    chunks = await active_chunks(db, uuid.UUID(document_id))
    children = [chunk for chunk in chunks if chunk.kind == "child"]
    parents = [chunk for chunk in chunks if chunk.kind == "parent"]
    assert children and parents
    parent_ids = {chunk.id for chunk in parents}
    for child in children:
        assert child.parent_chunk_id in parent_ids
        assert child.heading_path
        assert child.source_spans
        assert child.token_count <= 160
        assert child.embedding_model == fake_runtime.embeddings.model_id
        assert child.embedding_dimension == 8
    for parent in parents:
        assert parent.is_embedded is False

    # Source spans must map back to the parsed source text.
    async with db.sessionmaker() as session:
        from app.persistence.models import Document

        document = await session.get(Document, uuid.UUID(document_id))
    storage_path = document.metadata_["storage_path"]
    source = resolve_path(fake_runtime.settings, storage_path).read_bytes()
    parsed = parse_markdown(source, "quy-che.md")
    for child in children:
        for span in child.source_spans:
            assert 0 <= span["char_start"] <= span["char_end"] <= len(parsed.text) + 2

    vector = await fake_runtime.embeddings.embed_query("bao nhiêu ngày phép năm")
    points = await fake_runtime.vectors.search(
        workspace_id=kb.workspace_id,
        kb_id=kb.id,
        vector=vector,
        limit=5,
    )
    assert points
    assert all(point.payload["active"] is True for point in points)
    assert all(point.payload["revision_id"] == detail["revisions"][0]["id"] for point in points)

    events = (await client.get(f"/api/v1/jobs/{body['job_id']}", headers=auth_headers)).json()[
        "events"
    ]
    assert any(event["stage"] == "publishing" for event in events)
    for event in events:
        assert "Nhân viên được nghỉ 12 ngày" not in event["message"]


async def test_duplicate_upload_is_predictable(client, auth_headers, kb, fake_runtime) -> None:
    first = await upload(
        client, auth_headers, str(kb.id), "a.md", MARKDOWN.encode(), "text/markdown"
    )
    await wait_job(client, auth_headers, first["job_id"])
    response = await client.post(
        "/api/v1/documents",
        headers=auth_headers,
        data={"knowledge_base_id": str(kb.id)},
        files={"file": ("copy.md", MARKDOWN.encode(), "text/markdown")},
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "duplicate_document"
    assert error["details"]["document_id"] == first["document"]["id"]


async def test_scanned_pdf_fails_without_empty_index(
    client, auth_headers, kb, fake_runtime, db
) -> None:
    body = await upload(
        client, auth_headers, str(kb.id), "scan.pdf", make_pdf(["", ""]), "application/pdf"
    )
    job = await wait_job(client, auth_headers, body["job_id"])
    assert job["status"] == "failed"
    assert job["error_code"] == "ocr_required"
    detail = (
        await client.get(f"/api/v1/documents/{body['document']['id']}", headers=auth_headers)
    ).json()
    assert detail["status"] == "failed"
    assert detail["chunk_count"] == 0
    assert await active_chunks(db, uuid.UUID(body["document"]["id"])) == []

    vector = await fake_runtime.embeddings.embed_query("scan")
    points = await fake_runtime.vectors.search(
        workspace_id=kb.workspace_id, kb_id=kb.id, vector=vector, limit=10
    )
    assert points == []


async def test_invalid_and_unsupported_files_rejected(
    client, auth_headers, kb, fake_runtime
) -> None:
    fake = await client.post(
        "/api/v1/documents",
        headers=auth_headers,
        data={"knowledge_base_id": str(kb.id)},
        files={"file": ("fake.pdf", b"this is not a pdf", "application/pdf")},
    )
    assert fake.status_code == 415

    unsupported = await client.post(
        "/api/v1/documents",
        headers=auth_headers,
        data={"knowledge_base_id": str(kb.id)},
        files={"file": ("evil.exe", b"MZ binary", "application/octet-stream")},
    )
    assert unsupported.status_code == 415

    empty = await client.post(
        "/api/v1/documents",
        headers=auth_headers,
        data={"knowledge_base_id": str(kb.id)},
        files={"file": ("empty.md", b"", "text/markdown")},
    )
    assert empty.status_code == 422


async def test_delete_removes_retrievable_data(client, auth_headers, kb, fake_runtime, db) -> None:
    body = await upload(
        client, auth_headers, str(kb.id), "delete-me.md", MARKDOWN.encode(), "text/markdown"
    )
    await wait_job(client, auth_headers, body["job_id"])
    document_id = uuid.UUID(body["document"]["id"])
    response = await client.delete(f"/api/v1/documents/{document_id}", headers=auth_headers)
    assert response.status_code == 204

    assert (
        await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)
    ).status_code == 404
    chunks = await active_chunks(db, document_id)
    assert chunks == []
    vector = await fake_runtime.embeddings.embed_query("ngày phép")
    points = await fake_runtime.vectors.search(
        workspace_id=kb.workspace_id, kb_id=kb.id, vector=vector, limit=10
    )
    assert points == []


async def test_reprocess_is_idempotent_and_supersedes(
    client, auth_headers, kb, fake_runtime, db
) -> None:
    body = await upload(
        client, auth_headers, str(kb.id), "versioned.md", MARKDOWN.encode(), "text/markdown"
    )
    first_job = await wait_job(client, auth_headers, body["job_id"])
    assert first_job["status"] == "succeeded"
    document_id = uuid.UUID(body["document"]["id"])

    response = await client.post(f"/api/v1/documents/{document_id}/reprocess", headers=auth_headers)
    assert response.status_code == 202, response.text
    second_job = await wait_job(client, auth_headers, response.json()["job_id"])
    assert second_job["status"] == "succeeded", second_job

    detail = (await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)).json()
    assert detail["status"] == "ready"
    assert detail["active_revision_number"] == 2
    statuses = {revision["revision_number"]: revision["status"] for revision in detail["revisions"]}
    assert statuses == {1: "superseded", 2: "ready"}

    async with db.sessionmaker() as session:
        revisions = (
            (
                await session.execute(
                    select(DocumentRevision).where(DocumentRevision.document_id == document_id)
                )
            )
            .scalars()
            .all()
        )
        old_chunks = (
            (
                await session.execute(
                    select(Chunk).where(
                        Chunk.revision_id == next(r.id for r in revisions if r.revision_number == 1)
                    )
                )
            )
            .scalars()
            .all()
        )
    assert old_chunks
    assert all(not chunk.active for chunk in old_chunks)

    vector = await fake_runtime.embeddings.embed_query("ngày phép")
    points = await fake_runtime.vectors.search(
        workspace_id=kb.workspace_id, kb_id=kb.id, vector=vector, limit=20
    )
    assert points
    assert {point.payload["revision_id"] for point in points} == {
        next(revision.id for revision in revisions if revision.revision_number == 2).__str__()
    }


async def test_cancel_queued_job(client, auth_headers, kb, fake_runtime, monkeypatch) -> None:
    async def _no_dispatch(_job_id):
        return None

    monkeypatch.setattr(fake_runtime, "dispatch", _no_dispatch)
    body = await upload(
        client, auth_headers, str(kb.id), "queued.md", MARKDOWN.encode(), "text/markdown"
    )
    job_id = body["job_id"]
    queued = (await client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)).json()
    assert queued["status"] == "queued"
    response = await client.post(f"/api/v1/jobs/{job_id}/cancel", headers=auth_headers)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled"
    detail = (
        await client.get(f"/api/v1/documents/{body['document']['id']}", headers=auth_headers)
    ).json()
    assert detail["status"] == "cancelled"


async def test_running_job_observes_cancellation(
    app, client, auth_headers, kb, test_settings
) -> None:
    from app.embeddings.service import EmbeddingService
    from app.ingestion.runtime import IngestionRuntime
    from app.providers.fake import DeterministicFakeEmbeddingProvider

    provider = DeterministicFakeEmbeddingProvider(8, delay_seconds=0.15)
    runtime = IngestionRuntime(
        settings=test_settings,
        database=app.state.database,
        qdrant=app.state.qdrant,
        embeddings=EmbeddingService(provider, cache_enabled=False, batch_size=1, max_attempts=1),
    )
    await runtime.vectors.ensure_collection()
    await app.state.ingestion.shutdown()
    app.state.ingestion = runtime

    many = "# Chính sách\n\n" + "\n\n".join(
        f"Điều {index}. Nội dung chi tiết về quy trình và trách nhiệm của bộ phận."
        for index in range(80)
    )
    body = await upload(client, auth_headers, str(kb.id), "slow.md", many.encode(), "text/markdown")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = (await client.get(f"/api/v1/jobs/{body['job_id']}", headers=auth_headers)).json()
        if job["stage"] == "embedding" or job["status"] in {"succeeded", "failed"}:
            break
        await asyncio.sleep(0.05)
    response = await client.post(f"/api/v1/jobs/{body['job_id']}/cancel", headers=auth_headers)
    assert response.status_code == 200
    job = await wait_job(client, auth_headers, body["job_id"], limit_seconds=30)
    assert job["status"] == "cancelled", job
    detail = (
        await client.get(f"/api/v1/documents/{body['document']['id']}", headers=auth_headers)
    ).json()
    assert detail["status"] == "cancelled"
    assert detail["chunk_count"] == 0


async def test_retry_failed_job(client, auth_headers, kb, fake_runtime) -> None:
    body = await upload(
        client, auth_headers, str(kb.id), "scan2.pdf", make_pdf([""]), "application/pdf"
    )
    first = await wait_job(client, auth_headers, body["job_id"])
    assert first["status"] == "failed"
    response = await client.post(f"/api/v1/jobs/{body['job_id']}/retry", headers=auth_headers)
    assert response.status_code == 202, response.text
    new_job = response.json()
    assert new_job["id"] != body["job_id"]
    second = await wait_job(client, auth_headers, new_job["id"])
    assert second["status"] == "failed"


async def test_url_ingestion_disabled_by_default(client, auth_headers, kb, fake_runtime) -> None:
    response = await client.post(
        "/api/v1/documents/url",
        headers=auth_headers,
        json={"knowledge_base_id": str(kb.id), "url": "http://example.com/page.html"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "url_ingestion_disabled"


async def test_url_ingestion_with_faked_transport(
    client, auth_headers, kb, fake_runtime, monkeypatch
) -> None:
    from app.api.v1 import documents as documents_module
    from app.security.url_fetch import FetchedPage

    async def _fake_fetch(_url: str, _settings) -> FetchedPage:
        html = (
            "<html><head><title>Chính sách</title></head><body>"
            "<h1>Chính sách</h1><p>Nội dung từ web.</p></body></html>"
        )
        return FetchedPage(
            content=html.encode(),
            final_url="https://example.com/policy.html",
            content_type="text/html",
            filename="policy.html",
        )

    monkeypatch.setattr(documents_module, "fetch_url", _fake_fetch)
    response = await client.post(
        "/api/v1/documents/url",
        headers=auth_headers,
        json={"knowledge_base_id": str(kb.id), "url": "https://example.com/policy.html"},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    job = await wait_job(client, auth_headers, body["job_id"])
    assert job["status"] == "succeeded", job
    detail = (
        await client.get(f"/api/v1/documents/{body['document']['id']}", headers=auth_headers)
    ).json()
    assert detail["source_type"] == "url"
    assert detail["source_uri"] == "https://example.com/policy.html"
    assert detail["status"] == "ready"


async def test_jobs_list_and_cross_tenant_job_access(
    client, auth_headers, second_workspace, kb, fake_runtime
) -> None:
    body = await upload(
        client, auth_headers, str(kb.id), "jobs.md", MARKDOWN.encode(), "text/markdown"
    )
    await wait_job(client, auth_headers, body["job_id"])
    listing = await client.get("/api/v1/jobs", headers=auth_headers)
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    other_headers = {"Authorization": f"Bearer {second_workspace[1]}"}
    cross = await client.get(f"/api/v1/jobs/{body['job_id']}", headers=other_headers)
    assert cross.status_code == 404
