# Feature Parity Matrix

Reference parity status: **UNVERIFIED** — `NguynHung-VN/GRC-RAG` returned 404.
Nothing in the "Reference" column is a verified reference feature. It records
what a GRC-oriented RAG system would plausibly include, and is marked
`UNKNOWN` rather than inferred.

Status legend:
- Implementation: `DONE`, `PARTIAL`, `PLANNED`, `DEFERRED`, `NOT_APPLICABLE`
- Verification: `TESTED` (executed evidence), `PARTIAL`, `NOT_RUN`, `UNVERIFIED`

| Capability | Reference evidence | Reference classification | Independent enhancement | Implementation | Verification |
|---|---|---|---|---|---|
| Repository purpose | none | UNKNOWN | Vietnamese-first GRC-ready RAG platform | DONE | TESTED |
| Backend framework | none | UNKNOWN | FastAPI + Pydantic v2 + SQLAlchemy 2 async | DONE | TESTED |
| Relational metadata store | none | UNKNOWN | PostgreSQL 16 + Alembic | DONE | TESTED |
| Vector store | none | UNKNOWN | Qdrant with workspace/KB payload filters | DONE | TESTED |
| Lexical retrieval | none | UNKNOWN | Explicit BM25 over PostgreSQL term statistics | DONE | TESTED |
| Hybrid fusion | none | UNKNOWN | Reciprocal Rank Fusion, configurable depth | DONE | TESTED |
| Reranking | none | UNKNOWN | Optional multilingual cross-encoder, bounded fallback | PARTIAL | TESTED |
| PDF/DOCX/TXT/MD/HTML ingestion | none | UNKNOWN | Signature validation, provenance, page/section spans | DONE | TESTED |
| URL ingestion | none | UNKNOWN | Opt-in, SSRF guarded, redirect-validated | PARTIAL | TESTED |
| OCR | none | UNKNOWN | Explicit optional pipeline; no silent empty index | DEFERRED | NOT_RUN |
| Structure-aware chunking | none | UNKNOWN | Heading/paragraph/table aware, parent-child | DONE | TESTED |
| Vietnamese normalization | none | UNKNOWN | NFC, diacritic preserving, syllable+bigram tokenizer | DONE | TESTED |
| Embedding providers | none | UNKNOWN | Local E5, OpenAI-compatible, Ollama adapters | PARTIAL | TESTED |
| Embedding model versioning | none | UNKNOWN | Revision-scoped vectors, dimension checks, no mixing | PARTIAL | TESTED |
| Citation validation | none | UNKNOWN | Marker resolution + abstention, claim support in eval | PARTIAL | TESTED |
| Query rewriting | none | UNKNOWN | Optional, bounded, trace-visible | DEFERRED | NOT_RUN |
| GraphRAG | none | UNKNOWN | Gated: provenance-bound entities/relations, local/global | DEFERRED | NOT_RUN |
| Deterministic routing | none | UNKNOWN | AUTO/hybrid/basic modes with explicit fallback/warnings | DONE | TESTED |
| Evaluation | none | UNKNOWN | Versioned dataset, retrieval metrics, run APIs; 74 AI-authored questions (human review NOT_VERIFIED) | PARTIAL | TESTED |
| Web dashboard | none | UNKNOWN | Next.js Vietnamese-first, real API contracts | TODO | NOT_RUN |
| Authentication | none | UNKNOWN | Hashed workspace API keys + scopes | DONE | TESTED |
| Tenant isolation | none | UNKNOWN | Server-side workspace scoping on retrieval/chat/traces/evals/settings | DONE | TESTED |
| Rate limiting | none | UNKNOWN | Redis sliding window with explicit local fallback | DONE | PARTIAL |
| Streaming answers | none | UNKNOWN | SSE with typed events and citation event | DONE | TESTED |
| Observability | none | UNKNOWN | JSON logs, request IDs, persisted retrieval traces | PARTIAL | TESTED |
| Docker deployment | none | UNKNOWN | Compose for infra + app profile (web image pending) | PARTIAL | TESTED |
| CI | none | UNKNOWN | Ruff, mypy, pytest, web build | PARTIAL | NOT_RUN |
| Graph database | none | UNKNOWN | Intentionally omitted until demonstrated need | NOT_APPLICABLE | — |
| Reference parity | none | UNKNOWN | Cannot be claimed | — | UNVERIFIED |

## Verified engineering improvements (independent of reference)

1. Explicit `REFERENCE_UNAVAILABLE` handling instead of invented parity.
2. Readiness distinguishes *configured* from *available* for LLM and embeddings.
3. Document revisions make reindexing and atomic publication auditable.
4. Hybrid retrieval stages are individually switchable and traced.
5. BM25 is implemented explicitly and documented as non-identical to
   PostgreSQL `ts_rank`.
6. Retrieved text is framed as untrusted data in generation prompts.
7. Every phase report contains executed test evidence, not declarations.
