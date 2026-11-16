# ADR 0001 — Monorepo layout and runtime

**Status:** Accepted
**Date:** 2026-11-16

## Context

The system needs an API, a background worker, shared domain logic, and a web
dashboard, while remaining runnable on one developer machine.

## Decision

- One repository with `apps/api`, `apps/worker`, `apps/web`.
- A single Python package `app` owns domain, ingestion, retrieval, providers,
  and persistence. The `worker` package is a thin arq entrypoint importing
  `app`.
- Python 3.12, `uv` lockfile, FastAPI, SQLAlchemy 2 async, Alembic.
- Development infra runs in Docker Compose; app processes can run locally.

## Consequences

- No duplicate ingestion code between API and worker.
- Worker and API must be deployed from the same version.
- Python path manipulation is required in dev (`PYTHONPATH=apps/api:apps/worker`),
  handled by the Makefile and pytest config.

## Rejected alternatives

- Separate installable packages per module: unnecessary overhead at this scale.
- Celery: arq is lighter and async-native for this workload.
