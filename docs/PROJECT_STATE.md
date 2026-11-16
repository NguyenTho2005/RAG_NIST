# Project State

Last updated: 2026-11-16 (Asia/Ho_Chi_Minh session)

## Resume Here — CRITICAL COURSE CORRECTION (Policy RAG / NIST CSF Evidence Platform)

- Project root: `/home/tho/Documents/RAG` (no git repo, no commits). **Do not restart; do not delete/overwrite working code destructively.**
- **Current corrective phase:** **P0 — Workspace audit and migration plan: COMPLETED (documentation/verification evidence below).**
- **Next incomplete critical phase:** **P1 — Verified policy parsing and provenance** (then P2 chunking ~600/80, P3 Gemini+indexing+storage interfaces/local adapter, P4 retrieval parity, P5 structured Q&A + exact quote verification, P6+ NIST assessment, etc.).
- **Overall status:** **PARTIALLY COMPLETE**. The existing general RAG baseline is healthy and reusable, but it is **not functionally aligned** with the corrected Policy RAG / NIST CSF evidence-platform specification.
- **Functional reference:** user-supplied README/spec treated as **reference description only** (not verified by executing the original repository). Reference repo remains **`REFERENCE_UNAVAILABLE`** (404; owner public repos only unrelated `maison`); no parity claimed.

## Corrected objective (supersedes prior broad spec)
- Build a **Policy RAG system** for security-policy Q&A and **preliminary NIST CSF 2.0 design-coverage assessment** based on document evidence only (not operational effectiveness/compliance/certification/audit). All assessments preliminary and subject to human review.
- Required direction: provenance-preserving parsing, heading-aware chunking (~600 tokens, ~80 overlap), Gemini doc/query embeddings + L2/cosine correctness, independent vector+BM25 top-20 with RRF (MIN_SIM on cosine only), structured grounded answers with **exact quote verification** (Unicode quote/controlled ellipsis normalization only; paraphrases rejected), official NIST CSF 2.0 (6/22/106) with deterministic app-computed statuses and persistence/review, optional license-aware CIS v8.1/crosswalk, human-review evaluation toolkit.
- **Deferred:** GraphRAG/autonomous multi-agent/broad enterprise features, decorative/fabricated UI stats.

## P0 Audit outcomes (executed)

### Verification evidence (2026-11-16, non-destructive)
| Check | Command | Result |
|---|---|---|
| Lint | `uv run ruff check apps tests` | **PASS** — All checks passed |
| Types | `PYTHONPATH=apps/api:apps/worker uv run mypy apps/api/app apps/worker/worker` | **PASS** — no issues in 86 source files |
| Tests | `PYTHONPATH=apps/api:apps/worker uv run pytest -q` | **PASS** — **77 passed in 60.67s** |

### Deliverables produced
- `docs/POLICY_RAG_GAP_ANALYSIS.md` — current implementation status, reusable/modify/defer/missing classification, incorrect assumptions, missing functionality, risks, revised P0–P12 sequence, files requiring changes.
- `docs/POLICY_RAG_MIGRATION_PLAN.md` — incremental corrective migration plan, component disposition, phase deliverables/gates, verification protocol, ADRs to add, immediate next target (P1).
- `docs/PROJECT_STATE.md` — updated (this file).

### Key findings (summary)
- **REUSE:** FastAPI/config/logging/auth/rate-limit foundations; Postgres/Alembic revision+job+trace models; Qdrant vector store with cosine thresholding and revision activation; explicit BM25 over `chunk_terms`; RRF fusion; parsing/chunking/ingestion primitives; provider abstractions/cache/retry; pytest harness with real Postgres/Qdrant/Redis.
- **MODIFY:** parsing provenance/heading recovery and locators; chunking defaults/invariants (~600/80) and versioning; embedding/generation providers for Gemini task types, L2/cosine discipline, OpenRouter/structured outputs, diagnostics; retrieval inspector/explainability outputs; citations toward exact quote verification/evidence records; storage interfaces + index manifests + local lightweight adapter (`chunks.jsonl`/`vectors.npy`/BM25 state); evaluation toward labeling/kappa/quote validity/safety metrics.
- **MISSING/DEFER:** NIST CSF 2.0 official data/loader/assessment/scoring/persistence/review; CIS/crosswalk (optional, license-aware); policy/NIST UI and retrieval inspector UI (`apps/web` absent); exact-quote structured Q&A; human-review evaluation toolkit per spec. GraphRAG/multi-agent remain deferred.
- **Incorrect assumptions to correct:** general-RAG product framing; non-Gemini default model stack; marker citations instead of exact-quote verified evidence; incomplete retrieval evidence surfacing; no index manifests/local adapter parity; assessment/compliance framing risks (CIS inconsistency must be resolved via ADR/tests: CIS corroborating/contradicting only, prefer NEEDS_REVIEW on contradiction, never auto-upgrade).

## Verified reusable baseline (pre-correction, still passing)
- General RAG stack: FastAPI, auth/workspace isolation, Postgres/Qdrant/Redis integrations, ingestion/revisions/jobs, dense+BM25+RRF retrieval, grounded chat/SSE with marker citations, traces, models/settings/graph-placeholder/evaluations APIs, retrieval-eval harness (74 AI-authored questions, human review `NOT_VERIFIED`).
- Evidence: 77 tests, ruff, mypy passing as recorded above (real Postgres/Qdrant/Redis; deterministic fake embeddings dim 8 and fake LLM/reranker where needed).

## Critical Gaps / Blocked Work (corrective roadmap)
- **P1–P5:** policy-grade provenance/locators, chunking alignment, Gemini embeddings+task types+manifests+local adapter, retrieval inspector parity, structured answers with exact quote verification and insufficient-evidence handling.
- **P6–P7:** official NIST CSF 2.0 (6/22/106) verified import/licensing/provenance, cached subcategory embeddings, resumable per-category assessments, deterministic app-computed statuses, human review workflow.
- **P8:** usable UI (docs, policy Q&A, NIST assessment/review, retrieval inspector, evaluation) — `apps/web` missing.
- **P9:** labeling/kappa/confusion matrix/citation & quote accuracy/Recall@K/false COVERED/MISSING/abstention-NEEDS_REVIEW, held-out docs, model comparison.
- **P10:** optional CIS v8.1 + versioned crosswalk (official identifier mappings only, relationship types preserved, license-aware), ADR-compliant inconsistency handling.
- **Provider/data dependencies (likely BLOCKED until available):** Gemini/OpenRouter credentials/quota and verified/licensed official NIST/CIS/crosswalk source data; live model-dependent tests must not be marked PASS without execution.
- **Prior blockers still relevant:** no real embedding/LLM pulled for true semantic/generation E2E; human review `NOT_VERIFIED` for existing eval labels; broader security/backup-restore hardening pending.

## ADRs/actions to record next
- Draft/accept: storage interfaces & index manifests/local adapter (ADR-0007), evidence/quote-verification contract (ADR-0008), deterministic NIST scoring & LLM boundary (ADR-0009), CIS/crosswalk inconsistency & licensing (ADR-0010).

## Commands

```bash
# Infra
docker compose up -d postgres redis qdrant ollama

# DB migrations
PYTHONPATH=apps/api:apps/worker uv run alembic upgrade head

# Tests (real Postgres/Qdrant/Redis; fake embeddings/LLM/reranker only)
PYTHONPATH=apps/api:apps/worker uv run pytest tests -q

# Quality gates
uv run ruff check apps tests
uv run ruff format --check apps tests
PYTHONPATH=apps/api:apps/worker uv run mypy apps/api/app apps/worker/worker

# Local API
PYTHONPATH=apps/api:apps/worker uv run uvicorn app.main:app --reload
```

Do not commit `.env`, `/tmp/opencode/workspace.out`, API keys, generated data, or restricted/licensed mapping datasets.
