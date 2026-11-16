"""Integration tests for the evaluation runner endpoints."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.evaluation.dataset import load_corpus

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2] / "evaluation"


async def _wait_job(client: AsyncClient, headers: dict[str, str], job_id: str) -> dict:
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


async def _ingest_corpus(client: AsyncClient, headers: dict[str, str], kb_id: str) -> None:
    for filename, text in load_corpus(ROOT).items():
        response = await client.post(
            "/api/v1/documents",
            headers=headers,
            data={"knowledge_base_id": kb_id},
            files={"file": (filename, text.encode(), "text/markdown")},
        )
        assert response.status_code == 202, response.text
        job = await _wait_job(client, headers, response.json()["job_id"])
        assert job["status"] == "succeeded", (
            filename,
            job.get("error_code"),
            job.get("error_message"),
        )


async def test_evaluation_datasets_are_listed(
    client: AsyncClient, auth_headers: dict[str, str], fake_runtime
) -> None:
    response = await client.get("/api/v1/evaluations/datasets", headers=auth_headers)
    assert response.status_code == 200, response.text
    items = response.json()
    assert len(items) == 1
    dataset = items[0]
    assert dataset["id"] == "grc-vi-v1"
    assert dataset["case_count"] >= 50
    assert dataset["review_status"] == "NOT_VERIFIED"
    assert dataset["authoring"].startswith("AI-authored")
    assert dataset["limitations"]


async def test_evaluation_run_completes_with_real_retrieval(
    client: AsyncClient, auth_headers: dict[str, str], kb, fake_runtime
) -> None:
    await _ingest_corpus(client, auth_headers, str(kb.id))
    response = await client.post(
        "/api/v1/evaluations/runs",
        headers=auth_headers,
        json={
            "knowledge_base_id": str(kb.id),
            "dataset_id": "grc-vi-v1",
            "mode": "hybrid",
            "top_k": 5,
            "limit": 12,
        },
    )
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["status"] == "completed"
    assert run["case_count"] == 12
    assert run["dataset_sha256"]
    assert run["metrics"]["cases"] == 12
    assert run["metrics"]["answerable_cases"] == 12
    assert 0.0 <= run["metrics"]["recall_at_k"] <= 1.0
    assert 0.0 <= run["metrics"]["precision_at_k"] <= 1.0
    assert 0.0 <= run["metrics"]["hit_at_k"] <= 1.0
    assert 0.0 <= run["metrics"]["mrr"] <= 1.0
    assert 0.0 <= run["metrics"]["ndcg_at_k"] <= 1.0
    assert run["metrics"]["latency_ms_p50"] is not None
    assert run["config"]["dataset_review_status"] == "NOT_VERIFIED"
    assert "NOT_VERIFIED" in run["config"]["disclaimer"]
    assert "AI-authored" in run["config"]["disclaimer"]
    assert len(run["cases"]) == 12
    assert run["cases"][0]["retrieved"]
    assert run["cases"][0]["metrics"]["hit_at_k"] in (0.0, 1.0)
    assert isinstance(run["cases"][0]["matched_sources"], list)

    listing = await client.get("/api/v1/evaluations/runs", headers=auth_headers)
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["items"][0]["id"] == run["id"]

    detail = await client.get(f"/api/v1/evaluations/runs/{run['id']}", headers=auth_headers)
    assert detail.status_code == 200
    assert detail.json()["metrics"] == run["metrics"]


async def test_evaluation_run_is_workspace_scoped(
    client: AsyncClient,
    auth_headers: dict[str, str],
    kb,
    second_workspace,
    fake_runtime,
) -> None:
    other_headers = {"Authorization": f"Bearer {second_workspace[1]}"}
    forbidden = await client.post(
        "/api/v1/evaluations/runs",
        headers=other_headers,
        json={"knowledge_base_id": str(kb.id), "limit": 1},
    )
    assert forbidden.status_code == 404, forbidden.text

    await _ingest_corpus(client, auth_headers, str(kb.id))
    created = await client.post(
        "/api/v1/evaluations/runs",
        headers=auth_headers,
        json={"knowledge_base_id": str(kb.id), "limit": 1},
    )
    assert created.status_code == 201, created.text
    run_id = created.json()["id"]
    cross = await client.get(f"/api/v1/evaluations/runs/{run_id}", headers=other_headers)
    assert cross.status_code == 404


async def test_evaluation_rejects_unknown_dataset(
    client: AsyncClient, auth_headers: dict[str, str], kb, fake_runtime
) -> None:
    response = await client.post(
        "/api/v1/evaluations/runs",
        headers=auth_headers,
        json={"knowledge_base_id": str(kb.id), "dataset_id": "missing-dataset"},
    )
    assert response.status_code == 404, response.text
