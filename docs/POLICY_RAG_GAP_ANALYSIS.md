# Policy RAG / NIST CSF Evidence Platform — Gap Analysis

**Date:** 2026-11-16  
**Status:** Workspace audit completed (non-destructive). This analysis compares the **current workspace implementation** against the **corrected Policy RAG / NIST CSF Evidence Platform specification supplied by the user**. The supplied README/spec is treated as a *functional reference description only*; behavior of the original reference repository has **not** been verified by executing it.

## 1. Current implementation status (verified)

### 1.1 Baseline verification (executed now)
| Check | Command | Result |
|---|---|---|
| Lint | `uv run ruff check apps tests` | **PASS** — All checks passed |
| Types | `PYTHONPATH=apps/api:apps/worker uv run mypy apps/api/app apps/worker/worker` | **PASS** — no issues in 86 source files |
| Tests | `PYTHONPATH=apps/api:apps/worker uv run pytest -q` | **PASS** — **77 passed in 60.67s** |

Environment/tooling observed: Python 3.12 via `uv 0.12.13`, Node `v24.20.0`/npm `11.19.0`, Docker compose services configured (`postgres`, `redis`, `qdrant`, `ollama`; `api/worker/web` under `full` profile). No `AGENTS.md` present. No git repository initialized.

### 1.2 What is actually implemented
- **Product shape:** general-purpose Vietnamese-first multilingual RAG/chat platform (KB/document/job/chat/trace/settings/models/stats APIs), **not** a Policy RAG / NIST CSF evidence-assessment platform.
- **Parsing/provenance (`apps/api/app/parsing/*`):** DOCX/PDF/TXT/MD/HTML parsers, NFC normalization, page numbers for PDF text extraction, basic DOCX heading styles/tables/paragraph iteration, char spans per block, parser versioning. **Gaps:** PDF heading recovery is limited; DOCX/PDF provenance is block/page-level but not yet the exact policy-evidence model required (e.g. robust section/page+heading locators per evidence record); scanned/OCR path explicitly rejected (`OcrRequiredError`).
- **Chunking (`apps/api/app/chunking/chunker.py`):** structure-aware heading/section chunking with parent/child chunks, configurable tokens, overlap from sentence tail. **Defaults/behavior differ:** target 384 / max 512 / overlap 48 (spec target ~600, overlap ~80). Prevents oversized chunks and keeps table/header context, but does not explicitly implement “never merge unrelated headings” as a named invariant with tests matching the new spec.
- **Embeddings (`apps/api/app/providers/*`, `embeddings/service.py`):** providers `local` (sentence-transformers, lazy CPU, L2-normalized output), `openai`-compatible, `ollama`; content-addressed cache; dimension checks; no mixing within configured model/dim paths. **No Gemini embeddings, no document/query task-type separation, no explicit task-type-aware normalization policy for Gemini.**
- **Indexes/storage:**
  - **Vector:** Qdrant (`indexing/vector_store.py`, cosine distance, mandatory workspace/KB/model/active filters, revision activation, deterministic point IDs, schema/dimension checks on existing collection).
  - **Sparse/BM25:** explicit BM25 over PostgreSQL `chunk_terms` (`retrieval/sparse.py`, configurable `k1/b`, corpus stats, active-revision scoped). No `ts_rank` dependency for BM25 scoring.
  - **Persistence:** PostgreSQL metadata/chunks/terms/jobs/traces/chat/evals (`persistence/models.py`, Alembic migrations). File storage for uploads in `data/`.
  - **Missing required lightweight parity:** no `chunks.jsonl` + `vectors.npy` + BM25-state local index adapter; no storage interfaces abstracting document/chunk/vector/sparse/assessment/framework repositories; indexes do not record all required versions together in one manifest (chunking/source doc/index schema/versioned config) as specified.
- **Retrieval (`apps/api/app/retrieval/*`):** independent dense (Qdrant, `min_dense_score` applied as cosine `score_threshold` in Qdrant query — **correctly on cosine, not RRF**) and sparse BM25 lanes, RRF fusion, optional reranker, context assembly, per-stage timings/candidates in traces. Retrieval inspector UI and explicit exposure of per-lane rankings/RRF/cosine/selected chunks/filtering explanations as end-user features are **not fully implemented** (API/trace data exists partially; no frontend).
- **Generation/citations (`apps/api/app/generation/*`, `citations/validator.py`):** Ollama and OpenAI-compatible chat providers, grounded prompt, refusal/insufficient-context behavior, marker citations `[S1]` validation/stripping, rewrite on missing citations. **Not aligned with spec:** no structured Pydantic output schema for policy answers, no exact source-quotation verification against chunks, no quote normalization rules (Unicode quotes/controlled ellipsis) with paraphrase rejection, no evidence records resolving to verifiable chunk+location with exact quote checks, no Gemini/OpenRouter adapters, no provider-specific quota/error classification as required.
- **NIST CSF / CIS / crosswalk:** **none implemented**. `GET /api/v1/graph` is an unrelated honest gated placeholder. No official NIST CSF 2.0 machine-readable data, loader, subcategory/category/function model, evidence-per-subcategory retrieval, cached subcategory query embeddings, structured assessment, deterministic status computation, persistent assessment results, human review workflow, CIS module, or crosswalk layer.
- **APIs/UI:** FastAPI routes for health, KBs, documents/jobs, search, chat/stream/sessions, traces, models, settings, stats, evaluations. **No frontend** (`apps/web` absent). No policy Q&A UI, NIST assessment UI, retrieval inspector UI, evaluation labeling/review UI, or CIS UI.
- **Evaluation (`apps/api/app/evaluation/*`, `evaluation/*`):** retrieval-focused harness, 74-question AI-authored Vietnamese GRC-style dataset (`NOT_VERIFIED`, synthetic corpus), metrics runner/APIs/persistence. **Missing spec requirements:** labeling template generation/validation, reviewer agreement/Cohen’s kappa, confusion matrix, evidence citation accuracy, exact quote validity, false COVERED/MISSING tracking, abstention/NEEDS_REVIEW behavior, held-out evaluation docs, model comparison.

## 2. Features that can be reused (REUSE / with care)

| Area | Classification | Rationale / caveats |
|---|---|---|
| FastAPI app shell, config, logging, request IDs, error handling, middleware | **REUSE** | Stable, tested foundation; extend config for Gemini/OpenRouter/NIST/CIS/storage modes. |
| Auth/API-key, workspace isolation patterns, rate limiting | **REUSE (MODIFY scope)** | Useful for multi-user operation; product may simplify but isolation logic is correct and tested. |
| PostgreSQL + Alembic + SQLAlchemy models for documents/revisions/chunks/jobs/traces | **REUSE** | Proven provenance/revision/publication model; extend for policy evidence, framework, assessments, review records. |
| Qdrant vector store + active-revision filtering | **REUSE** | Correct cosine filtering semantics, dimension mismatch protection, activation/deactivation flow; adapt for Gemini dims/models/task types and optional local adapter parity. |
| BM25 over `chunk_terms` + tokenizer | **REUSE/MODIFY** | Explicit, auditable sparse lane and RRF-compatible; verify tokenizer suitability for English policy text and required behavior, keep independent lane. |
| RRF fusion core | **REUSE** | Correctly fuses ranks (not score scales); expose component ranks/scores per spec. |
| Parsing primitives (DOCX/PDF/HTML/MD/TXT), char spans, table handling | **REUSE/MODIFY** | Good starting point; must harden heading recovery, source locators, and exact-quote resolvability for policy evidence. |
| Heading/section-aware chunker with parent-child | **REUSE/MODIFY** | Re-tune defaults (~600/80), enforce non-merging invariants, and record chunking/index versions in manifests. |
| Ingestion/job runtime, revision publication ordering, cancellation/retry cleanup | **REUSE** | Valuable reliability behavior; adapt to new embedding/index/assessment pipelines. |
| Provider abstractions, retry helpers, embedding cache | **REUSE/MODIFY** | Extend for Gemini task types, OpenRouter, structured output, bounded backoff, quota/auth/unsupported/transient classification. |
| Test harness (pytest + real Postgres/Qdrant/Redis, fixtures) | **REUSE** | Strong base for new non-destructive, provider-gated integration tests. |
| Evaluation metrics primitives (`metrics.py`) | **REUSE** | Recall@k/MRR/nDCG/percentiles reusable; expand to required safety/evidence/reviewer metrics. |
| Docker/infra for Postgres/Redis/Qdrant/Ollama | **REUSE** | Keep for existing path; ensure lightweight local adapter works without forcing DB complexity. |

## 3. Incorrect architectural assumptions (must be corrected)

1. **Wrong product objective:** built as broad Advanced RAG/chat/graph-ready platform instead of **Policy RAG + NIST CSF 2.0 design-coverage evidence** with mandatory preliminary/human-review framing.
2. **Wrong model stack assumption:** defaults to E5/local/Ollama/OpenAI-compatible; spec requires **Gemini embeddings + Gemini structured generation**, **OpenRouter-compatible generation adapter**, task-type separation, L2 normalization consistency, and no incompatible embedding mixing.
3. **Answer/evidence contract mismatch:** marker citations `[S1]` and “quote” snippets from context are **not** exact-source-quote verification or structured evidence records; spec requires structured answers, exact quotation validation, quote normalization limits (Unicode quotes/controlled ellipsis only), paraphrase rejection, and chunk/location-resolvable evidence.
4. **Retrieval productization gap:** MIN_SIM (`min_dense_score`) is applied in dense lane (cosine threshold) which is directionally right, but system does not yet provide required **Retrieval Inspector** outputs (independent vector/BM25 rankings, RRF ranks, cosine scores, selected chunks, filtering explanations) as first-class verified capability/UI.
5. **Storage rigidity:** reliance on Postgres+Qdrant is preserved but spec prioritizes functional parity and requires **storage interfaces** plus **lightweight local dev adapter** (`chunks.jsonl`, `vectors.npy`, BM25 state) and index manifests/version rejection/reindex procedure.
6. **Assessment/compliance framing risk:** no NIST logic today, but future implementation must not claim operational effectiveness/compliance/certification/audit completion; LLM must not be authoritative for final status; retrieval failure must not prove absence.
7. **Overbuilt/deferred areas emphasized:** graph endpoints, broad chat/settings/models surfaces, general eval corpus — useful in places but secondary to P0–P8 policy/NIST-critical path; must not delay functional parity.
8. **CIS/crosswalk logic hazard (per spec):** README inconsistency (CIS “can only lower” vs example upgrading MISSING→PARTIAL) must be resolved explicitly (treat CIS as corroborating/contradicting evidence, prefer NEEDS_REVIEW on contradiction, never auto-upgrade on CIS match alone) and codified in ADR/tests — currently no implementation or ADR.

## 4. Missing Policy RAG functionality (by spec area)

- **P1 Parsing/provenance:** robust DOCX heading/paragraph/table provenance and PDF page+recovered-heading provenance; exact source-location records suitable for quote verification; no unrelated heading merges (verified by tests).
- **P2 Chunking:** ~600 token target, ~80 overlap (configurable), heading-aware boundaries, recorded chunking version.
- **P3 Embeddings/indexing:** Gemini document/query embeddings with task types, consistent L2 normalization, cosine similarity correctness, incompatible-config rejection, persistent index metadata (embedding model/dims/chunking version/source doc version/index schema version), reindex procedure.
- **P4 Retrieval:** independent vector top-20 and BM25 top-20, RRF fusion, top-K evidence selection, MIN_SIM on cosine only (not RRF), exposed individual rankings/scores/filters.
- **P5 Q&A/verification:** structured grounded answers, exact quote verification with allowed Unicode-quote/controlled-ellipsis normalization only, paraphrase rejection, insufficient-evidence answers, evidence records resolving to actual chunk+location.
- **P6 NIST CSF 2.0:** official machine-readable data import with provenance/version/source/licensing verification; 6 Functions / 22 Categories / 106 Subcategories; loader/lookups/grouping; evidence retrieval per subcategory; cached subcategory query embeddings; structured assessment; resumable independently persisted results; failed category must not destroy successes.
- **P7 Deterministic scoring:** app-computed status (COVERED/PARTIAL/NEEDS_REVIEW/MISSING/ERROR) from verified evidence + LLM analysis fields (coverage, language strength, exception_present, citations, supporting quotes, rationale, uncertainty); confidence/limitations separate; explicit unit tests for every rule/conflict.
- **P8 UI:** Document Management, Policy Q&A with quote/source inspection, NIST assessment/review, Retrieval Inspector, Evaluation, optional CIS views — no fabricated stats/mock outputs.
- **P9 Evaluation:** labeling templates/validation, reviewer agreement, Cohen’s kappa, confusion matrix, evidence citation accuracy, exact quote validity, Recall@K, false COVERED/MISSING, abstention/NEEDS_REVIEW, held-out docs, model comparison, dataset limitations.
- **P10 CIS/crosswalk:** optional CIS v8.1 from officially sourced mappings to NIST CSF 2.0 (identifier relationships only, no embedding-fabricated mappings), relationship types preserved, license-aware storage, graceful without optional files; separately versioned crosswalk for NIST CSF 2.0/CIS v8.1/ISO/IEC 27001:2022 refs; ADR + regression tests for inconsistency resolution.
- **P11 Hardening:** provider diagnostics/quota classification/fallback/bounded backoff, structured-output validation/failure reporting, security/performance/reliability hardening, backup/reindex guidance.

## 5. Migration risks

- **Scope/regression risk:** large existing codebase; corrective refactor can break 77 passing tests or ingestion/retrieval behavior — **mitigate with non-destructive changes, adapter layers, and continuous targeted/full verification**.
- **Provider/credential risk:** Gemini/OpenRouter/NIST official data access/licensing and quotas may be unavailable — **mark BLOCKED, never PASS without execution; keep provider-independent validation and local adapters**.
- **Legal/licensing risk:** NIST data provenance/licenses and CIS mapping restrictions must be verified before import/commit; restricted derived datasets must not be committed.
- **Assessment integrity risk:** conflating document vs operational evidence or auto-upgrading/downgrading status incorrectly; requires strict deterministic rules, ADR, and exhaustive unit tests.
- **Storage divergence risk:** supporting both Postgres/Qdrant and lightweight local index can create inconsistency — **define interfaces + manifests + compatibility checks**.
- **Performance/context risk:** “one assessment LLM call per Category when provider context limits allow” needs chunk budgeting, fallback splitting, and resumability.

## 6. Revised task sequence (P0–P12)

1. **P0 (now):** workspace audit, `POLICY_RAG_GAP_ANALYSIS.md`, `POLICY_RAG_MIGRATION_PLAN.md`, update `PROJECT_STATE.md`, record ADRs needed (incl. CIS inconsistency, storage/index strategy). **Do not restart/delete.**
2. **P1:** verified policy parsing/provenance hardening (DOCX/PDF locators, heading recovery, no unrelated merge) + tests.
3. **P2:** heading-aware chunking aligned to ~600/80, invariants, chunking versioning + tests.
4. **P3:** Gemini embedding provider (doc/query task types, consistent L2/cosine), index manifests/version checks, reindex flow; lightweight local index adapter (`chunks.jsonl`, `vectors.npy`, BM25 state) behind interfaces.
5. **P4:** retrieval alignment (vector top-20/BM25 top-20, RRF, MIN_SIM on cosine only), exposed rankings/scores/selection/filtering data (retrieval inspector backend).
6. **P5:** structured policy Q&A, exact quote verification (allowed normalization only), paraphrase rejection, insufficient-evidence handling, evidence records resolved to chunk+location.
7. **P6:** official NIST CSF 2.0 loader/lookups with provenance/licensing verification, cached subcategory query embeddings, evidence retrieval per subcategory, structured/resumable per-category assessment persistence.
8. **P7:** deterministic status computation, LLM-as-analysis-only contract, confidence/limitations separation, conflict/exception handling, exhaustive unit tests.
9. **P8:** usable application UI (docs, Q&A, NIST assessment/review, retrieval inspector, evaluation) with real data only.
10. **P9:** human-review evaluation toolkit (labels, agreement/kappa, confusion matrix, citation/quote validity, Recall@K, false COVERED/MISSING, abstention/NEEDS_REVIEW), held-out docs, model comparison.
11. **P10:** optional CIS v8.1 + separately versioned crosswalk (official identifier mappings only, license-aware, graceful absence), ADR-compliant CIS evidence handling.
12. **P11:** security/performance/reliability hardening and diagnostics.
13. **P12:** end-to-end acceptance and documentation with explicit PASS/FAIL/BLOCKED/NOT_RUN.

## 7. Files/components requiring changes (initial)

- `docs/PROJECT_STATE.md`, `docs/POLICY_RAG_GAP_ANALYSIS.md`, `docs/POLICY_RAG_MIGRATION_PLAN.md`, new ADRs under `docs/adr/` (storage/index-manifest strategy, CIS/NIST-CIS conflict resolution, assessment determinism/evidence contract).
- `apps/api/app/core/config.py`, `.env.example`, `README.md`: new provider/index/assessment settings, Gemini/OpenRouter config, local-adapter/index-manifest settings, reindex controls.
- Parsing/chunking: `apps/api/app/parsing/{parsers.py,types.py,detect.py}`, `apps/api/app/chunking/chunker.py`, related ingestion integration.
- Embeddings/providers: `apps/api/app/providers/{embedding_base.py,factory.py,remote_embedding.py,local_embedding.py}`, new `gemini_embedding.py` (task types), `embeddings/service.py`, normalization/model-id/index metadata handling.
- Index/storage: `apps/api/app/indexing/vector_store.py`, `apps/api/app/retrieval/sparse.py`, new storage interfaces and local adapters (e.g. `apps/api/app/storage/*` or equivalent), ingestion/runtime wiring, manifest/version checks.
- Retrieval: `apps/api/app/retrieval/{dense.py,sparse.py,fusion.py,service.py,context.py,types.py,chunks.py}`, API/trace exposure for inspector.
- Generation/citations: `apps/api/app/generation/{base.py,factory.py,service.py,ollama.py,openai.py}`, new Gemini/OpenRouter providers and structured output validators, `apps/api/app/citations/validator.py` (exact quote verification, normalization, evidence records), schemas.
- NIST/CIS/crosswalk/assessment: new modules (e.g. `apps/api/app/frameworks/*`, `apps/api/app/assessments/*`), data ingestion scripts/loaders, persistence models/migrations, APIs.
- API/UI: `apps/api/app/api/v1/*`, schemas/deps, new frontend `apps/web/*` (P8+).
- Evaluation: `apps/api/app/evaluation/*`, `evaluation/*`, labeling/review artifacts and reports.
- Tests: extend `tests/unit`, `tests/integration`, provider-gated e2e; add deterministic scoring, quote verification, NIST/CIS/crosswalk, index compatibility tests.

## 8. Immediate conclusion

- **Baseline is healthy** (77/77 tests, ruff, mypy pass) and contains substantial **reusable infrastructure** (ingestion, Postgres/Qdrant, BM25/RRF, auth, traces, eval harness), but it is **not functionally aligned** with the corrected Policy RAG / NIST CSF evidence platform.
- **P0 complete for audit artifacts in this step** once migration plan/state are written. Next implementation focus must be **P1–P4** (provenance, chunking, Gemini/indexing, retrieval parity) before UI/assessment breadth, reusing existing working components and preserving verified behavior. All provider/data/licensing-dependent steps will be recorded as PASS/FAIL/BLOCKED/NOT_RUN with evidence.