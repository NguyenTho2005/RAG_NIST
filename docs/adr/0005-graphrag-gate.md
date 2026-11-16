# ADR 0005 — GraphRAG gate

**Status:** Accepted
**Date:** 2026-11-16

## Context

Graph extraction is expensive, LLM-dependent, and can invent relationships.
It must not destabilise the baseline retrieval system.

## Decision

- GraphRAG is disabled by default (`GRAPH_ENABLED=false`).
- It is implemented only after baseline and hybrid retrieval gates pass.
- Extraction is schema-constrained JSON; malformed output is rejected with
  bounded retries.
- Every relationship stores the supporting chunk and confidence.
- Community summaries are derived artifacts, never primary evidence.
- Graph retrieval failure falls back to hybrid retrieval and records the reason.
- No Neo4j by default; PostgreSQL persists nodes/edges, NetworkX computes
  neighborhoods/communities.

## Consequences

- Basic search works with no graph data.
- Graph improvements must be benchmarked against hybrid retrieval before being
  described as improvements.
