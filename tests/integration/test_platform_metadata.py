"""Integration tests for model probes, workspace settings, graph gate and defaults."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from tests.integration.test_retrieval_chat import ingest

pytestmark = pytest.mark.integration


async def test_models_reports_configured_and_honest_availability(
    client: AsyncClient, auth_headers: dict[str, str], fake_runtime
) -> None:
    response = await client.get("/api/v1/models", headers=auth_headers)
    assert response.status_code == 200, response.text
    body = response.json()

    roles = {item["role"]: item for item in body["models"]}
    assert set(roles) == {"embedding", "generation", "reranker"}

    embedding = roles["embedding"]
    assert embedding["model_id"] == "test-embedding"
    assert embedding["dimension"] == 8
    assert embedding["configured"] is True
    assert embedding["available"] is True

    generation = roles["generation"]
    assert generation["provider"] == "none"
    assert generation["configured"] is False
    assert generation["available"] is False
    assert "503" in (generation["detail"] or "")

    reranker = roles["reranker"]
    assert reranker["configured"] is False
    assert reranker["available"] is False

    assert body["capabilities"]["retrieval"] is True
    assert body["capabilities"]["generation"] is False
    assert body["capabilities"]["graph"] is False


async def test_settings_defaults_and_patch(
    client: AsyncClient, auth_headers: dict[str, str], fake_runtime
) -> None:
    response = await client.get("/api/v1/settings", headers=auth_headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["default_mode"] == "hybrid"
    assert body["top_k"] == 8
    assert body["rerank_enabled"] is False
    assert body["rerank_available"] is False

    rejected = await client.patch(
        "/api/v1/settings",
        headers=auth_headers,
        json={"rerank_enabled": True},
    )
    assert rejected.status_code == 422, rejected.text
    assert rejected.json()["error"]["code"] == "reranker_not_available"

    updated = await client.patch(
        "/api/v1/settings",
        headers=auth_headers,
        json={"default_mode": "basic", "top_k": 3},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["default_mode"] == "basic"
    assert updated.json()["top_k"] == 3

    reread = await client.get("/api/v1/settings", headers=auth_headers)
    assert reread.status_code == 200
    assert reread.json()["default_mode"] == "basic"
    assert reread.json()["top_k"] == 3


async def test_settings_are_workspace_scoped(
    client: AsyncClient, auth_headers: dict[str, str], second_workspace, fake_runtime
) -> None:
    updated = await client.patch(
        "/api/v1/settings",
        headers=auth_headers,
        json={"default_mode": "basic", "top_k": 2},
    )
    assert updated.status_code == 200, updated.text

    other_headers = {"Authorization": f"Bearer {second_workspace[1]}"}
    other = await client.get("/api/v1/settings", headers=other_headers)
    assert other.status_code == 200
    assert other.json()["default_mode"] == "hybrid"
    assert other.json()["top_k"] == 8


async def test_search_uses_workspace_defaults(
    client: AsyncClient, auth_headers: dict[str, str], kb, fake_runtime
) -> None:
    await ingest(client, auth_headers, str(kb.id))
    await client.patch(
        "/api/v1/settings",
        headers=auth_headers,
        json={"default_mode": "basic", "top_k": 1},
    )

    response = await client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={
            "knowledge_base_id": str(kb.id),
            "query": "Nhân viên được nghỉ bao nhiêu ngày phép năm?",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode_used"] == "dense"
    assert "sparse" not in body["candidates"]
    assert len(body["results"]) <= 1

    explicit = await client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={
            "knowledge_base_id": str(kb.id),
            "query": "Nhân viên được nghỉ bao nhiêu ngày phép năm?",
            "mode": "hybrid",
            "top_k": 3,
        },
    )
    assert explicit.status_code == 200, explicit.text
    assert explicit.json()["mode_used"] == "hybrid"


async def test_graph_reports_disabled_and_enabled_unavailable(
    app, client: AsyncClient, auth_headers: dict[str, str], fake_runtime
) -> None:
    disabled = await client.get("/api/v1/graph", headers=auth_headers)
    assert disabled.status_code == 200, disabled.text
    body = disabled.json()
    assert body["enabled"] is False
    assert body["available"] is False
    assert body["entities"] == []
    assert body["relations"] == []
    assert "GRAPH_ENABLED" in body["reason"]

    app.state.settings.graph_enabled = True
    enabled = await client.get("/api/v1/graph", headers=auth_headers)
    assert enabled.status_code == 200, enabled.text
    body = enabled.json()
    assert body["enabled"] is True
    assert body["available"] is False
    assert body["entities"] == []
    assert "chưa được triển khai" in body["reason"]


async def test_metadata_endpoints_require_auth(client: AsyncClient) -> None:
    for path in ("/api/v1/models", "/api/v1/settings", "/api/v1/graph"):
        response = await client.get(path)
        assert response.status_code == 401, (path, response.text)
