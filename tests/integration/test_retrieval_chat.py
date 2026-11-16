"""Integration tests for dense search, chat, citations, streaming and traces."""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration

MARKDOWN = """# Chính sách nhân sự

## Nghỉ phép năm

Nhân viên chính thức được nghỉ 12 ngày phép năm. Mỗi năm làm việc liên tục
tăng thêm một ngày, tối đa 18 ngày.

## Quy trình phê duyệt

Nhân viên gửi đơn trước ít nhất 3 ngày làm việc. Quản lý trực tiếp phê duyệt
trên hệ thống nhân sự.

## Bảo mật dữ liệu

Dữ liệu cá nhân trong đơn nghỉ phép được bảo mật theo chính sách bảo vệ dữ liệu.
"""


async def wait_job(client: AsyncClient, headers: dict[str, str], job_id: str) -> dict:
    deadline = time.monotonic() + 30
    last: dict | None = None
    while time.monotonic() < deadline:
        response = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in {"succeeded", "failed", "cancelled"}:
            return last
        await asyncio.sleep(0.05)
    raise AssertionError(f"job did not finish: {last}")


async def ingest(client: AsyncClient, headers: dict[str, str], kb_id: str) -> dict:
    response = await client.post(
        "/api/v1/documents",
        headers=headers,
        data={"knowledge_base_id": kb_id},
        files={"file": ("chinh-sach.md", MARKDOWN.encode(), "text/markdown")},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    job = await wait_job(client, headers, body["job_id"])
    assert job["status"] == "succeeded", job
    return body


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((event, json.loads(data)))
    return events


async def test_search_returns_sources_and_trace(
    client: AsyncClient, auth_headers: dict[str, str], kb, fake_runtime
) -> None:
    await ingest(client, auth_headers, str(kb.id))
    response = await client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={
            "knowledge_base_id": str(kb.id),
            "query": "Nhân viên được nghỉ bao nhiêu ngày phép năm?",
            "top_k": 3,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode_used"] == "hybrid"
    assert body["candidates"]["sparse"] >= 1
    assert body["embedding_model"] == "test-embedding"
    assert body["results"], body
    first = body["results"][0]
    assert first["filename"] == "chinh-sach.md"
    assert first["chunk_id"] and first["document_id"] and first["revision_id"]
    assert first["text"]
    assert first["marker"] == "[S1]"
    assert isinstance(first["score"], float)

    trace = await client.get(f"/api/v1/traces/{body['trace_id']}", headers=auth_headers)
    assert trace.status_code == 200, trace.text
    trace_body = trace.json()
    assert trace_body["id"] == body["trace_id"]
    assert trace_body["selected"]
    assert trace_body["candidates"]["dense"] >= 1
    assert trace_body["candidates"]["sparse"] >= 1
    assert trace_body["config"]["sparse_enabled"] is True
    assert trace_body["timings"]["dense_ms"] >= 0

    listing = await client.get("/api/v1/traces", headers=auth_headers)
    assert listing.status_code == 200
    assert listing.json()["total"] == 1


async def test_basic_mode_skips_sparse(
    client: AsyncClient, auth_headers: dict[str, str], kb, fake_runtime
) -> None:
    await ingest(client, auth_headers, str(kb.id))
    response = await client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={
            "knowledge_base_id": str(kb.id),
            "query": "ngày phép năm",
            "mode": "basic",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode_used"] == "dense"
    assert "sparse" not in body["candidates"]


async def test_reranker_reorders_hybrid_candidates(
    app, client: AsyncClient, auth_headers: dict[str, str], kb, fake_runtime
) -> None:
    from tests.fixtures.fake_reranker import PreferTextReranker

    await ingest(client, auth_headers, str(kb.id))
    reranker = PreferTextReranker("bảo mật")
    app.state.reranker = reranker
    app.state.settings.rerank_enabled = True
    response = await client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={
            "knowledge_base_id": str(kb.id),
            "query": "chính sách dữ liệu nhân sự",
            "mode": "hybrid",
            "top_k": 3,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode_used"] == "hybrid+rerank"
    assert reranker.calls == 1
    assert "bảo mật" in body["results"][0]["text"].casefold()
    assert body["results"][0]["score"] == 1.0
    trace = await client.get(f"/api/v1/traces/{body['trace_id']}", headers=auth_headers)
    assert trace.status_code == 200
    assert trace.json()["config"]["rerank_enabled"] is True


async def test_search_is_workspace_scoped(
    client: AsyncClient,
    auth_headers: dict[str, str],
    kb,
    second_workspace,
    fake_runtime,
) -> None:
    await ingest(client, auth_headers, str(kb.id))
    other_headers = {"Authorization": f"Bearer {second_workspace[1]}"}
    response = await client.post(
        "/api/v1/search",
        headers=other_headers,
        json={"knowledge_base_id": str(kb.id), "query": "nghỉ phép"},
    )
    assert response.status_code == 404

    own = await client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={"knowledge_base_id": str(kb.id), "query": "nghỉ phép"},
    )
    trace_id = own.json()["trace_id"]
    cross_trace = await client.get(f"/api/v1/traces/{trace_id}", headers=other_headers)
    assert cross_trace.status_code == 404


async def test_chat_without_context_refuses_without_llm(
    client: AsyncClient, auth_headers: dict[str, str], kb, fake_runtime
) -> None:
    response = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"knowledge_base_id": str(kb.id), "message": "Chính sách nghỉ phép?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["refused"] is True
    assert body["grounded"] is False
    assert body["citations"] == []
    assert body["sources"] == []
    assert body["citation_status"] == "not_applicable"
    assert "không tìm thấy" in body["answer"].lower()


async def test_chat_with_sources_requires_configured_llm(
    client: AsyncClient, auth_headers: dict[str, str], kb, fake_runtime
) -> None:
    await ingest(client, auth_headers, str(kb.id))
    response = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"knowledge_base_id": str(kb.id), "message": "Bao nhiêu ngày phép?"},
    )
    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "generation_unavailable"


async def test_chat_validates_citations_and_persists_session(
    app, client: AsyncClient, auth_headers: dict[str, str], kb, fake_runtime
) -> None:
    from tests.fixtures.fake_llm import DeterministicFakeLLM

    await ingest(client, auth_headers, str(kb.id))
    fake = DeterministicFakeLLM()
    app.state.llm_provider = fake

    response = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"knowledge_base_id": str(kb.id), "message": "Bao nhiêu ngày phép năm?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["grounded"] is True
    assert body["citation_status"] == "valid"
    assert body["citations"][0]["marker"] == "[S1]"
    assert body["citations"][0]["filename"] == "chinh-sach.md"
    assert body["citations"][0]["quote"]
    assert body["session_id"]
    assert fake.calls == 1
    assert "dữ liệu không đáng tin" in fake.last_messages[0]["content"]
    assert "chinh-sach.md" in fake.last_messages[1]["content"]

    session = await client.get(f"/api/v1/chat/sessions/{body['session_id']}", headers=auth_headers)
    assert session.status_code == 200
    messages = session.json()["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["citations"][0]["marker"] == "[S1]"

    trace = await client.get(f"/api/v1/traces/{body['trace_id']}", headers=auth_headers)
    assert trace.status_code == 200
    assert trace.json()["token_usage"]["input_tokens"] == 100


async def test_invalid_and_missing_citations_are_reported(
    app, client: AsyncClient, auth_headers: dict[str, str], kb, fake_runtime
) -> None:
    from tests.fixtures.fake_llm import DeterministicFakeLLM

    await ingest(client, auth_headers, str(kb.id))
    invalid = DeterministicFakeLLM(answer="Thông tin [S99] và [S1].")
    app.state.llm_provider = invalid
    response = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"knowledge_base_id": str(kb.id), "message": "Ngày phép?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["citation_status"] == "valid"
    assert "[S99]" not in body["answer"]
    assert any("không tồn tại" in warning for warning in body["warnings"])

    missing = DeterministicFakeLLM(answer="Không có thông tin trích dẫn.")
    app.state.llm_provider = missing
    response = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"knowledge_base_id": str(kb.id), "message": "Ngày phép?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["citation_status"] == "missing"
    assert body["grounded"] is False
    assert missing.calls == 2  # one corrective rewrite


async def test_chat_stream_emits_tokens_citations_and_done(
    app, client: AsyncClient, auth_headers: dict[str, str], kb, fake_runtime
) -> None:
    from tests.fixtures.fake_llm import DeterministicFakeLLM

    await ingest(client, auth_headers, str(kb.id))
    app.state.llm_provider = DeterministicFakeLLM()
    response = await client.post(
        "/api/v1/chat/stream",
        headers=auth_headers,
        json={"knowledge_base_id": str(kb.id), "message": "Ngày phép năm?"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert "token" in names
    assert names[-1] == "done"
    tokens = "".join(data["text"] for name, data in events if name == "token")
    assert "12 ngày phép" in tokens
    citations = next(data for name, data in events if name == "citations")
    assert citations["grounded"] is True
    assert citations["citations"][0]["marker"] == "[S1]"
    done = events[-1][1]
    trace = await client.get(f"/api/v1/traces/{done['trace_id']}", headers=auth_headers)
    assert trace.status_code == 200
    assert trace.json()["stages"][-1]["status"] == "ok"


async def test_chat_session_workspace_isolation(
    app,
    client: AsyncClient,
    auth_headers: dict[str, str],
    kb,
    second_workspace,
    fake_runtime,
) -> None:
    from tests.fixtures.fake_llm import DeterministicFakeLLM

    await ingest(client, auth_headers, str(kb.id))
    app.state.llm_provider = DeterministicFakeLLM()
    response = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"knowledge_base_id": str(kb.id), "message": "Nghỉ phép?"},
    )
    assert response.status_code == 200
    session_id = response.json()["session_id"]
    other_headers = {"Authorization": f"Bearer {second_workspace[1]}"}
    cross = await client.get(f"/api/v1/chat/sessions/{session_id}", headers=other_headers)
    assert cross.status_code == 404
    assert uuid.UUID(session_id)
