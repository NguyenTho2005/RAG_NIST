# ADR 0002 — Sparse retrieval: explicit BM25, not `ts_rank`

**Status:** Accepted
**Date:** 2026-11-16

## Context

PostgreSQL full-text search is available for free, but `ts_rank`/`ts_rank_cd`
are not BM25. Treating them as equivalent would make ablation reports
misleading and violate the requirement to select a sparse implementation
explicitly.

## Decision

Implement BM25 directly over PostgreSQL:

- `chunks.term_count` stores document length in sparse terms.
- `chunk_terms(chunk_id, term, tf)` stores postings.
- Query-time SQL computes IDF from active-corpus document frequency and applies
  `k1=1.2`, `b=0.75` (configurable).
- Vietnamese tokenization emits normalized syllables, syllable bigrams, and
  identifier-preserving alphanumeric tokens. Diacritics are preserved and are
  significant (`mã` ≠ `ma`).

## Consequences

- Scoring is genuinely BM25 and comparable across ablations.
- Postings add storage and write cost; very large corpora may justify a
  dedicated search engine later.
- Tokenization is deterministic and dependency-free, but is not full Vietnamese
  word segmentation. This limitation is recorded and tested with terminology
  and identifier cases.
