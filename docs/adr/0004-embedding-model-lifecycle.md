# ADR 0004 — Embedding model lifecycle

**Status:** Accepted
**Date:** 2026-11-16

## Context

Embedding models have different dimensions and vector spaces. Mixing vectors
from different model versions silently corrupts retrieval.

## Decision

- Default local model: `intfloat/multilingual-e5-small` (MIT, 384 dimensions,
  multilingual, CPU-friendly). E5 query/passage prefixes are applied.
- `BAAI/bge-m3` is documented as an optional heavier candidate, not downloaded
  automatically.
- Every KB records `embedding_model` and `embedding_dimension`.
- Every chunk records `embedding_model` and `embedding_dimension`.
- Qdrant points carry `embedding_model`; queries filter on it.
- Changing the model requires a new Qdrant collection (name includes a version)
  and a full reindex; the readiness/CLI checks dimension mismatches.
- Embedding cache keys include model identifier and content hash.

## Consequences

- No incompatible-vector mixing by construction.
- Reindexing costs are explicit.
- Model downloads happen only when an operator enables/uses the provider,
  never during application startup.
