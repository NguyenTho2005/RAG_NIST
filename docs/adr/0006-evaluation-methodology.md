# ADR 0006 — Evaluation methodology

**Status:** Accepted
**Date:** 2026-11-16

## Context

Retrieval quality, answer grounding, and operational cost must be measured, not
asserted. Citation validity and citation support are different properties.

## Decision

- Versioned datasets with answerable, unanswerable, multi-hop, conflicting,
  updated/deleted, Vietnamese-terminology, and prompt-injection cases.
- Retrieval metrics: Recall@k, Precision@k, Hit@k, MRR, nDCG@k.
- Answer metrics: answer relevance, citation validity (markers resolve),
  citation support (cited source supports the claim), faithfulness, abstention
  accuracy, completeness.
- Operational metrics: p50/p95 latency, throughput, indexing time, tokens,
  estimated cost, memory, stage latency.
- Ablations: dense, sparse, hybrid, hybrid+rerank, hybrid+rerank+context.
- The seed set is AI-authored and explicitly labelled; human review is recorded
  as `NOT_VERIFIED` until performed. Synthetic labels are never presented as
  human-reviewed gold data.
- Reports include environment, dataset hash, model identifiers, and commands.

## Consequences

- LLM-judge metrics require a configured provider; otherwise they are BLOCKED,
  not PASS.
- Results are reproducible only with the recorded model/version combination.
