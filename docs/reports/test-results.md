# Test Results

All results below were executed locally on 2026-11-16. A test is only marked PASS when the command actually ran and returned success. No mocked success is reported as production evidence.

## Full suite

```bash
PYTHONPATH=apps/api:apps/worker uv run pytest tests -q
```

Result: **77 passed** (56.66s)

Coverage of executed tests (real PostgreSQL/Qdrant/Redis; deterministic fake embeddings dimension 8 and fake LLM/reranker where required):

- **Foundation:** auth, tenant isolation, health/readiness, error envelope, KB CRUD/stats.
- **Parsing/chunking:** Vietnamese normalization, spans/tables, structure-aware chunking, parent links.
- **Ingestion E2E:** upload/parse/chunk/embed/publish, provenance, duplicate conflict, scanned-PDF failure without empty index, invalid/unsupported/empty rejection, delete/reprocess/revision supersession, queued/running cancel, retry, URL disabled/enabled (faked transport), job listing/cross-tenant denial.
- **Retrieval/generation:** dense search, BM25 sparse, hybrid RRF, optional reranker reorder path, context assembly, grounding/refusal, citation validation/missing/invalid handling, SSE streaming events, trace/session persistence and workspace isolation.
- **Platform metadata:** `/models` probes and capability flags, `/settings` defaults/patch/validation and workspace scoping, `/graph` gated honest status.
- **Evaluation harness:** dataset loader/validation and bundled dataset integrity (74 AI-authored questions, `NOT_VERIFIED`), metrics unit tests, `/evaluations/*` dataset listing/run/list/detail and workspace scoping over live ingested corpus.

## Targeted suites

```bash
PYTHONPATH=apps/api:apps/worker uv run pytest tests/integration/test_ingestion.py -q
PYTHONPATH=apps/api:apps/worker uv run pytest tests/integration/test_retrieval_chat.py -q
PYTHONPATH=apps/api:apps/worker uv run pytest tests/integration/test_platform_metadata.py -q
PYTHONPATH=apps/api:apps/worker uv run pytest tests/unit/test_evaluation.py tests/integration/test_evaluation_api.py -q
```

Latest combined targeted evidence in this session: **16 passed** for evaluation+platform subset (7.44s); full suite includes all above.

## Lint and format

```bash
uv run ruff check apps tests
uv run ruff format --check apps tests
```

Result: **All checks passed**, **105 files already formatted**.

## Type checking

```bash
PYTHONPATH=apps/api:apps/worker uv run mypy apps/api/app apps/worker/worker
```

Result: **Success: no issues found in 86 source files**.

## Explicit non-claims / limitations

- Embeddings in tests are deterministic fakes (dimension 8); **no real embedding model downloaded or semantically benchmarked**.
- **No real LLM generation E2E**: Ollama runs but `qwen2.5:1.5b-instruct` not pulled; answer-faithfulness/citation-support LLM-judge metrics not run.
- Evaluation dataset `grc-vi-v1` is **AI-authored draft, human review NOT_VERIFIED**; corpus is synthetic. Harness metrics are verified, semantic quality of production models is not.
- **No frontend** and no frontend/build tests (`apps/web` absent).
- **GraphRAG pipeline not implemented** (endpoint reports gated/unavailable honestly).
- Reference parity remains **UNVERIFIED** (`REFERENCE_UNAVAILABLE`).
