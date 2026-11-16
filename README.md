# Advanced RAG

A Vietnamese-first, multilingual Retrieval-Augmented Generation platform built from scratch:
traceable hybrid retrieval, grounded answers with validated citations, and reproducible evaluation.

> **Reference note.** The requested reference repository
> `NguynHung-VN/GRC-RAG` returned HTTP 404 during read-only inspection (the
> account exists and has one unrelated public repository). This project is an
> independent implementation; no reference source code was cloned, downloaded,
> or copied. See `docs/01-reference-analysis.md`.

## Status

Under active construction. See `docs/PROJECT_STATE.md` for the current phase,
verified functionality, and blockers.

## Quick start

```bash
cp .env.example .env
make infra-up
make install
make migrate
make create-workspace name="Demo"
# copy the printed API key
make api        # http://localhost:8000/api/docs
```

The API starts even when no LLM is configured; `/api/v1/ready` reports the
capability status instead of pretending generation works.

Detailed setup, testing, evaluation, and operational runbooks live in `docs/`.
