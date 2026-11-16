# Phase Status

Status legend: **DONE** (implemented and verified by executed tests), **PARTIAL** (some parts implemented/verified), **BLOCKED** (external dependency unavailable), **TODO** (not started).

| Phase | Scope | Status | Evidence / notes |
| --- | --- | --- | --- |
| 0 | Reference analysis, parity matrix, ADRs | DONE | `REFERENCE_UNAVAILABLE`; evidence recorded, no parity claimed. |
| 1 | Foundation: config, auth, DB, health, KB, CLI, infra | DONE | Included in 77-test suite; Ruff/mypy clean. |
| 2/3 | Parsing, Vietnamese chunking, embeddings, ingestion, jobs | DONE | Real Postgres/Qdrant ingestion tests passed. |
| 4 | Retrieval baseline: dense search, context, generation, citations, traces, chat/stream | PARTIAL | Implemented and tested with fake embeddings/LLM/reranker; **real LLM/model E2E not run**. |
| 5 | Sparse BM25, hybrid RRF, reranker, context optimization | PARTIAL | BM25+RRF+optional reranker implemented and tested; real semantic/reranker weights not benchmarked. |
| 6 | GraphRAG (gated) | PARTIAL | `/graph` reports honest gated/unavailable status; no extraction/build/query pipeline. |
| 7 | Frontend: 10 dashboard pages | TODO | `apps/web` absent. |
| 8 | Evaluation: 50+ questions, retrieval/generation metrics | PARTIAL | 74-question AI-authored dataset (`NOT_VERIFIED`), retrieval metrics/runner/APIs/persistence and tests done; no human review, no faithfulness/citation-support LLM-judge runs. |
| 9 | Security hardening and abuse cases | PARTIAL | Auth, workspace isolation, rate limiting, SSRF/ingestion controls exist; adversarial/redirect/DNS-rebinding tests and broader review pending. |
| 10 | Docs, ADRs, runbooks, final report | PARTIAL | Docs/state/reports updated through this session; final report must remain `PARTIALLY COMPLETE`. |

## Gate rules

- A phase is only DONE when its required tests actually execute successfully.
- Real-model claims require a real model probe/real run; configured-only capability must not be reported as available.
- AI-authored evaluation labels must be disclosed; human review remains `NOT_VERIFIED` until performed.
- Reference-parity claims remain prohibited while the reference is inaccessible.
