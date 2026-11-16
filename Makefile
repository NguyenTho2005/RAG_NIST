.PHONY: install infra-up infra-down infra-logs migrate revision api worker web test test-unit test-integration test-security lint typecheck check eval seed-demo create-workspace qdrant-init models pull-llm

install:
	uv sync --extra dev

infra-up:
	docker compose up -d postgres redis qdrant ollama

infra-down:
	docker compose down

infra-logs:
	docker compose logs -f --tail=100

migrate:
	uv run alembic upgrade head

revision:
	uv run alembic revision --autogenerate -m "$(m)"

api:
	PYTHONPATH=apps/api:apps/worker uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:
	PYTHONPATH=apps/api:apps/worker uv run python -m worker.main

web:
	cd apps/web && npm run dev

test:
	PYTHONPATH=apps/api:apps/worker uv run pytest -q

test-unit:
	PYTHONPATH=apps/api:apps/worker uv run pytest tests/unit -q

test-integration:
	PYTHONPATH=apps/api:apps/worker uv run pytest tests/integration -q

test-security:
	PYTHONPATH=apps/api:apps/worker uv run pytest tests/security -q

lint:
	uv run ruff check apps tests evals scripts
	uv run ruff format --check apps tests evals scripts

typecheck:
	uv run mypy apps/api/app apps/worker/worker

check: lint typecheck test

eval:
	PYTHONPATH=apps/api:apps/worker uv run python -m app.evaluation.cli --config evals/configs/default.json

seed-demo:
	PYTHONPATH=apps/api:apps/worker uv run python -m app.cli seed-demo

create-workspace:
	PYTHONPATH=apps/api:apps/worker uv run python -m app.cli create-workspace --name "$(name)"

qdrant-init:
	PYTHONPATH=apps/api:apps/worker uv run python -m app.cli init-qdrant

pull-llm:
	docker exec advanced-rag-ollama-1 ollama pull $(model)
