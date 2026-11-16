"""Deterministic test-only embedding provider.

This class is intentionally NOT registered in the provider factory and cannot be
selected through environment configuration. It exists so unit/integration tests
can run without downloading weights while still exercising dimension checks,
batching, caching, and persistence.
"""

from __future__ import annotations

import asyncio
import hashlib
import math

from app.providers.embedding_base import InputType


class DeterministicFakeEmbeddingProvider:
    def __init__(
        self,
        dimension: int = 8,
        delay_seconds: float = 0.0,
        *,
        model_id: str | None = None,
    ) -> None:
        self.dimension = dimension
        self.model_id = model_id or f"test-deterministic-{dimension}"
        self.calls = 0
        self.delay_seconds = delay_seconds

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        self.calls += 1
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        vectors: list[list[float]] = []
        for text in texts:
            seed = hashlib.sha256(f"{input_type}:{text}".encode()).digest()
            values: list[float] = []
            block = seed
            while len(values) < self.dimension:
                # Map bytes deterministically into a finite [-1, 1) range.
                for byte in block:
                    values.append((byte / 127.5) - 1.0)
                    if len(values) == self.dimension:
                        break
                block = hashlib.sha256(block).digest()
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            vectors.append([value / norm for value in values])
        return vectors

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None
