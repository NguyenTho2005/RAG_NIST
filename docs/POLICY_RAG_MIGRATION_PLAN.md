# Policy RAG / NIST CSF Evidence Platform — Migration Plan

**Date:** 2026-11-16  
**Based on:** `docs/POLICY_RAG_GAP_ANALYSIS.md`, current workspace state (`docs/PROJECT_STATE.md`), and corrected user-supplied specification.  
**Principles:** do **not** restart, do **not** delete/overwrite working code destructively, preserve verified behavior (77 passing tests, ruff, mypy), prefer adapters/interfaces over rewrites, and record every phase as **PASS / FAIL / BLOCKED / NOT_RUN** with executed evidence.

## 1. Audit summary and decision

- **Current state:** healthy general RAG foundation (parsing/chunking/ingestion, Postgres+Qdrant, BM25+RRF, chat/traces/evals scaffolding) but misaligned with required Policy RAG + NIST CSF 2.0 evidence workflow.
- **Decision:** **incremental corrective migration**. Reuse app shell, DB/revision/job/vector/BM25/test infrastructure; refactor/extend parsing, chunking, embeddings, retrieval outputs, generation/citations, storage abstractions, and add NIST/CIS/evaluation/UI modules in priority order.
- **Deferred (explicit):** GraphRAG/autonomous multi-agent workflows, broad unrelated enterprise features, decorative dashboard pages/statistics, any compliance/certification claims.

## 2. Component disposition

| Component | Disposition | Notes |
|---|---|---|
| FastAPI core, auth, logging, errors, middleware, rate limit | **REUSE** | Extend config/diagnostics; keep behavior compatible. |
| Postgres models/migrations (docs/chunks/jobs/traces/chat) | **MODIFY/EXTEND** | Add evidence/assessment/framework/review/index-manifest entities via additive migrations. |
| Qdrant vector store, revision activation | **REUSE/MODIFY** | Keep; add manifest/version metadata and Gemini model/task-type handling. |
| BM25 `chunk_terms` + sparse retriever | **REUSE/MODIFY** | Keep independent lane; ensure ranking exposure and tokenizer adequacy for policy text. |
| RRF fusion | **REUSE** | Expose per-lane ranks/RRF contributions. |
| Parsers/chunker | **MODIFY** | Harden provenance, heading recovery, no unrelated merges, ~600/80 targets, versioning. |
| Embedding providers/service/cache | **MODIFY/EXTEND** | Add Gemini (doc/query task types, consistent L2/cosine), preserve existing providers behind interfaces, enforce no-mix and reindex requirements. |
| Generation providers/service/citations | **MODIFY/REPLACE-INCREMENTALLY** | Add structured outputs, exact quote verification, Gemini/OpenRouter, provider diagnostics/fallback; keep/retire marker-only path behind compatibility where needed. |
| Existing eval dataset/harness | **MODIFY** | Keep metrics/runner plumbing; replace/extend datasets and metrics to spec (labeling/kappa/quote validity/safety errors/held-out). |
| Graph endpoint/UI stubs | **DEFER** | Unrelated placeholder; do not expand. |
| Frontend (`apps/web`) | **MISSING → BUILD (P8)** | Minimal usable policy UI; no mock/fabricated outputs. |
| NIST/CIS/crosswalk/assessments | **MISSING → BUILD (P6,P7,P10)** | Official sources, license-aware, deterministic, optional CIS. |
| Local lightweight index adapter | **MISSING → BUILD (P3/P4 support)** | `chunks.jsonl`, `vectors.npy`, BM25 state + manifests via storage interfaces. |

## 3. Phase plan, deliverables, and gates

### P0 — Workspace audit and migration plan (CURRENT)
- **Deliverables:** `docs/POLICY_RAG_GAP_ANALYSIS.md`, `docs/POLICY_RAG_MIGRATION_PLAN.md`, updated `docs/PROJECT_STATE.md`, ADR stubs/decisions list (storage/index manifest, CIS inconsistency resolution, deterministic assessment contract).
- **Verification:** non-destructive inspection, `uv run ruff check apps tests`, `PYTHONPATH=apps/api:apps/worker uv run mypy apps/api/app apps/worker/worker`, `PYTHONPATH=apps/api:apps/worker uv run pytest -q`.
- **Gate:** all executed checks recorded; no destructive changes. **Status target:** PASS (audit artifacts) / evidence recorded.
- **Risks:** none functional; ensure docs reflect reality.

### P1 — Verified policy parsing and provenance
- **Scope:** robust DOCX heading/paragraph/table preservation and PDF page numbers + heading recovery where possible; exact source locators (document/revision/page/heading/char/block identifiers) per chunk/evidence; prevent unrelated heading merges; scanned-PDF behavior explicit.
- **Deliverables:** parser/provenance model updates, unit tests (DOCX/PDF fixtures), integration tests with real sample docs (non-sensitive), provenance serialization.
- **Gate:** real DOCX/PDF parsing tests PASS; source locations preserved and resolvable; no-regression on existing parser/chunking tests or justified updates recorded.
- **Risk:** parser edge cases; mitigate with fixture corpus and conservative recovery.

### P2 — Heading-aware chunking (~600/80)
- **Scope:** target ~600 tokens, configurable overlap ~80, heading-aware boundaries, no unrelated heading merges, parent/child/context suitability for evidence, chunking version recorded.
- **Deliverables:** chunker config/default alignment, invariants, unit/integration tests (boundary, overlap, heading isolation, token bounds).
- **Gate:** chunking invariants and versioning tests PASS; existing ingestion tests updated only where behavior intentionally changed and documented.

### P3 — Gemini embedding and persistent indexing
- **Scope:** Gemini embedding provider with **separate document/query task types** (when supported), consistent L2 normalization, cosine-correct usage, no mixing incompatible configs; index manifests recording embedding model/dims/chunking version/source doc version/index schema version; compatibility rejection + clear reindex procedure.
- **Storage interfaces + lightweight adapter:** define interfaces for Document/Chunk/Vector/Sparse/Assessment/Framework stores; implement local dev adapter (`chunks.jsonl`, `vectors.npy`, BM25 state) where practical without breaking Postgres/Qdrant path.
- **Deliverables:** provider(s), manifest module, adapters, config/env docs, unit/integration tests (including compatibility rejection). Provider-dependent live Gemini tests **BLOCKED** if credentials/quota unavailable.
- **Gate:** deterministic/unit tests PASS; live embedding tests PASS or BLOCKED with evidence; reindex procedure documented.

### P4 — Vector + BM25 + RRF retrieval parity
- **Scope:** independent vector top-20 and BM25 top-20, RRF fusion, top-K evidence selection, **MIN_SIM applied to cosine scores only (not RRF)**, exposed individual rankings/cosine/BM25/RRF/selected chunks/filtering explanations (API/trace payloads for Retrieval Inspector).
- **Deliverables:** retrieval service/API/schema updates, inspector data contract, tests for ranking independence, threshold placement, fusion correctness, explainability fields.
- **Gate:** retrieval unit/integration tests PASS; behavior verified against stored/indexed fixtures.

### P5 — Structured policy Q&A and quote verification
- **Scope:** structured grounded answers (Pydantic-validated), exact quote verification against actual source chunks/locations; quote matching may normalize Unicode quotation marks and controlled ellipsis only; **reject semantic paraphrases** as exact quotations; insufficient-evidence answers; provider-independent validation.
- **Deliverables:** evidence/answer schemas, quote verifier, generation pipeline changes, tests for valid/invalid quotes, normalization boundaries, out-of-document abstention.
- **Gate:** quote verification and abstention tests PASS; provider-dependent generation tests PASS or BLOCKED.

### P6 — Official NIST CSF 2.0 loader and evidence assessment
- **Scope:** verify official machine-readable source and licensing/provenance before import; load 6 Functions/22 Categories/106 Subcategories (no invented data); lookups/groupings; evidence retrieval per subcategory; cached subcategory query embeddings; structured assessment (one LLM call per Category when context allows; fallback splitting); resumable, independently persisted results; failed Category must not destroy successful results. Support individual subcategory/category, one/multiple functions, full framework.
- **Deliverables:** `frameworks` module, verified data provenance record, cache/persistence, assessment runner/API, tests (loader integrity, persistence/resume, isolation on failure).
- **Gate:** loader/count/provenance tests PASS with verified local data; live assessment calls PASS or BLOCKED.

### P7 — Deterministic scoring and persistence
- **Scope:** LLM returns evidence analysis only (coverage FULL/PARTIAL/NONE, language strength MUST/SHOULD/MAY/UNKNOWN, exception_present, citations, supporting quotes, rationale, uncertainty). App verifies evidence and computes **COVERED/PARTIAL/NEEDS_REVIEW/MISSING/ERROR** per specified safeguards (retrieval failure ≠ proof absent, low score alone ≠ conclusive MISSING, failed/incomplete LLM ≠ MISSING, quote absence ≠ verified absence, confidence/limitations separate, no fabricated probabilities). Human review workflow fields/state.
- **Deliverables:** deterministic evaluator module, status/conflict matrix, persistence/review model, **explicit unit tests for every scoring rule and conflict case**.
- **Gate:** all deterministic rule tests PASS; integration persistence/review tests PASS.

### P8 — Application UI
- **Scope:** usable UI over real APIs: Document Management, Policy Q&A (grounded answers/exact quotes/source refs/retrieval results), NIST CSF Assessment/review, Retrieval Inspector, Evaluation, optional CIS placeholder gated. No fabricated stats, no mock model outputs.
- **Deliverables:** `apps/web` (Next.js/TS/Tailwind) or minimal equivalent aligned with repo stack, API wiring, build/run docs.
- **Gate:** build/tests PASS where feasible; manual/automated smoke against real backend recorded.

### P9 — Evaluation framework
- **Scope:** labeling template generation/validation, reviewer agreement, Cohen’s kappa, confusion matrix, evidence citation accuracy, exact quote validity, Retrieval Recall@K, false COVERED/MISSING, abstention/NEEDS_REVIEW behavior, held-out docs, model comparison, dataset limitations/safety-critical visibility.
- **Deliverables:** evaluation toolkit, schemas/templates, reports/export-import, tests.
- **Gate:** metric/unit tests PASS; benchmark runs recorded with dataset/model provenance and NOT_RUN/BLOCKED where applicable.

### P10 — Optional CIS and crosswalk integration
- **Scope:** optional CIS Controls v8.1 via **officially sourced mappings to NIST CSF 2.0** (identifier relationships only, never embedding-fabricated). Preserve relationship types (Equivalent/Subset/Superset), license-aware storage (no restricted files committed), graceful without optional data. Separately versioned crosswalk for NIST CSF 2.0 / CIS v8.1 / ISO/IEC 27001:2022 refs.
- **CIS inconsistency resolution (must implement consistently):** treat CIS as corroborating/contradicting evidence, **not automatic proof**. On contradiction prefer **NEEDS_REVIEW** unless reviewed evidence and documented rules justify otherwise. **Never auto-upgrade** a NIST outcome solely due to CIS match. Record in ADR and regression tests.
- **Deliverables:** optional importer/module, ADR, regression tests, UI flags.

### P11 — Security, performance, reliability hardening
- **Scope:** provider diagnostics (rate limit/daily quota/auth/unsupported structured output/transient), bounded exponential backoff, configurable fallback, model availability state, no fabricated fallback results; security/perf/reindex/backup guidance.
- **Gate:** targeted tests and failure-injection checks PASS; docs updated.

### P12 — End-to-end acceptance and documentation
- **Scope:** required E2E scenarios (real DOCX/PDF parsing, provenance, real embeddings, vector+BM25 retrieval, RRF, real structured LLM generation, valid/invalid quote detection, out-of-document abstention, NIST category assessment, deterministic status rules, assessment persistence, provider failure recovery, exceptions/contradictory evidence review). Provider-dependent cases PASS or BLOCKED with evidence.
- **Deliverables:** acceptance report, final docs/runbook, updated `PROJECT_STATE.md` and phase reports, explicit limitations and **overall status (expected PARTIALLY COMPLETE if any critical gate blocked/unverified)**.

## 4. ADRs to add/update (minimum)
- **ADR-0007 (proposed):** Storage interfaces and index manifests (Postgres/Qdrant + lightweight local adapter `chunks.jsonl`/`vectors.npy`/BM25 state; version/compatibility/reindex).
- **ADR-0008 (proposed):** Policy evidence contract — exact quote verification, allowed normalization, paraphrase rejection, chunk/location resolvability.
- **ADR-0009 (proposed):** Deterministic NIST assessment scoring and LLM-as-analysis-only boundary (statuses, conflicts, exceptions, retrieval-failure semantics).
- **ADR-0010 (proposed):** CIS/NIST inconsistency resolution and optional crosswalk licensing/provenance rules.

## 5. Verification and reporting protocol (every phase)
1. Implement real code (no mocks presented as production evidence).
2. Add/update unit and relevant integration tests.
3. Execute verification commands and record outputs.
4. Mark **PASS / FAIL / BLOCKED / NOT_RUN** per case.
5. Fix failures before progressing.
6. Preserve previous working functionality (regression suite).
7. Update `docs/PROJECT_STATE.md` and phase/report docs.
8. Record architecture decisions and unresolved limitations.

## 6. Immediate next implementation target (after P0)
- **Start P1:** audit and harden `apps/api/app/parsing/*` and provenance model against DOCX/PDF policy-evidence requirements (source locators, heading recovery, no unrelated heading merges), add/adjust tests non-destructively, then proceed to **P2** chunking alignment (~600/80).  
- **Parallel prep (non-blocking):** design storage interfaces/manifest schema (ADR-0007 draft) and verify official NIST CSF 2.0 machine-readable source/licensing for P6 without importing unverified/restricted data.