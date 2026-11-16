"""Retrieval metrics for source-level qrels.

Relevance judgments are (filename, evidence span) pairs. A retrieved chunk is
relevant when its filename matches and its text contains the normalized
evidence span. Retrieved chunks are deduplicated to the first occurrence of
each source so one long document cannot inflate nDCG by repeating a source.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(matched: int, expected: int) -> float:
    if expected <= 0:
        return 0.0
    return matched / expected


def precision_at_k(matched: int, k: int) -> float:
    if k <= 0:
        return 0.0
    return matched / k


def hit_at_k(matched: int) -> float:
    return 1.0 if matched > 0 else 0.0


def reciprocal_rank(first_relevant_rank: int | None) -> float:
    if first_relevant_rank is None:
        return 0.0
    return 1.0 / first_relevant_rank


def dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def ndcg_at_k(relevant_flags: Sequence[bool], expected: int, k: int) -> float:
    if expected <= 0 or k <= 0:
        return 0.0
    observed = dcg([1.0 if flag else 0.0 for flag in relevant_flags[:k]])
    ideal = dcg([1.0] * min(expected, k))
    if ideal == 0:
        return 0.0
    return min(observed / ideal, 1.0)


def ndcg_from_ranks(ranks: Sequence[int], expected: int, k: int) -> float:
    """Binary nDCG where each relevant source contributes at its first chunk rank."""
    if expected <= 0 or k <= 0:
        return 0.0
    observed = sum(1.0 / math.log2(rank + 1) for rank in ranks if 1 <= rank <= k)
    ideal = dcg([1.0] * min(expected, k))
    if ideal == 0:
        return 0.0
    return min(observed / ideal, 1.0)


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return round(sum(items) / len(items), 4)
