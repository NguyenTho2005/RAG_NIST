"""Foundation API tests (real Postgres/Qdrant/Redis from docker-compose)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def test_health_is_open_and_structured(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert response.headers["x-request-id"]


async def test_ready_reports_real_dependencies(client: AsyncClient) -> None:
    response = await client.get("/api/v1/ready")
    body = response.json()
    assert response.status_code in (200, 503)
    names = {component["name"] for component in body["components"]}
    assert {"postgres", "qdrant", "redis", "embeddings", "llm"} <= names
    postgres = next(c for c in body["components"] if c["name"] == "postgres")
    assert postgres["status"] == "ok"
    assert body["capabilities"]["generation"] is False
    assert body["capabilities"]["generation_configured"] is False


async def test_openapi_loads(client: AsyncClient) -> None:
    response = await client.get("/api/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/api/v1/knowledge-bases" in schema["paths"]
    assert schema["info"]["title"] == "Advanced RAG API"


async def test_request_id_is_echoed(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health", headers={"X-Request-ID": "trace-me-123"})
    assert response.headers["x-request-id"] == "trace-me-123"


async def test_unsafe_request_id_is_replaced(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health", headers={"X-Request-ID": "../../etc/passwd"})
    assert response.headers["x-request-id"] != "../../etc/passwd"


async def test_missing_api_key_rejected(client: AsyncClient) -> None:
    response = await client.get("/api/v1/knowledge-bases")
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "unauthorized"
    assert body["request_id"]


async def test_invalid_api_key_rejected(client: AsyncClient) -> None:
    response = await client.get(
        "/api/v1/knowledge-bases", headers={"Authorization": "Bearer rag_invalid"}
    )
    assert response.status_code == 401


async def test_knowledge_base_crud_and_validation(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await client.post(
        "/api/v1/knowledge-bases",
        headers=auth_headers,
        json={"name": "Chính sách nhân sự", "description": "Tài liệu nội bộ"},
    )
    assert created.status_code == 201, created.text
    kb = created.json()
    assert kb["name"] == "Chính sách nhân sự"
    assert kb["embedding_dimension"] == 8
    assert kb["document_count"] == 0

    listed = await client.get("/api/v1/knowledge-bases", headers=auth_headers)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    fetched = await client.get(f"/api/v1/knowledge-bases/{kb['id']}", headers=auth_headers)
    assert fetched.status_code == 200
    assert fetched.json()["id"] == kb["id"]

    duplicate = await client.post(
        "/api/v1/knowledge-bases", headers=auth_headers, json={"name": "chính sách nhân sự"}
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "knowledge_base_exists"

    invalid = await client.post("/api/v1/knowledge-bases", headers=auth_headers, json={"name": "x"})
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_failed"


async def test_workspace_isolation(
    client: AsyncClient,
    auth_headers: dict[str, str],
    second_workspace: tuple,
) -> None:
    created = await client.post(
        "/api/v1/knowledge-bases", headers=auth_headers, json={"name": "Private KB"}
    )
    assert created.status_code == 201
    kb_id = created.json()["id"]

    other_headers = {"Authorization": f"Bearer {second_workspace[1]}"}
    other_list = await client.get("/api/v1/knowledge-bases", headers=other_headers)
    assert other_list.status_code == 200
    assert other_list.json()["total"] == 0

    cross = await client.get(f"/api/v1/knowledge-bases/{kb_id}", headers=other_headers)
    assert cross.status_code == 404


async def test_pagination_limits(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    for index in range(3):
        response = await client.post(
            "/api/v1/knowledge-bases",
            headers=auth_headers,
            json={"name": f"KB {index}"},
        )
        assert response.status_code == 201
    response = await client.get("/api/v1/knowledge-bases?limit=2&offset=0", headers=auth_headers)
    body = response.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2


async def test_read_scope_is_insufficient_for_write(client: AsyncClient, db, workspace) -> None:
    from app.persistence.models import ApiKey
    from app.security.auth import generate_api_key

    ws, _, _ = workspace
    key, prefix, key_hash = generate_api_key()
    async with db.sessionmaker() as session:
        session.add(
            ApiKey(
                workspace_id=ws.id,
                name="read-only",
                prefix=prefix,
                key_hash=key_hash,
                scopes=["read"],
            )
        )
        await session.commit()
    response = await client.post(
        "/api/v1/knowledge-bases",
        headers={"Authorization": f"Bearer {key}"},
        json={"name": "Nope"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


async def test_unknown_route_returns_json_error(client: AsyncClient) -> None:
    response = await client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
