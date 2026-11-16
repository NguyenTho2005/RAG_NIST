"""Content-addressed embedding cache and batched embedding generation."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Awaitable, Callable

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ProviderError
from app.observability.logging import get_logger
from app.persistence.models import EmbeddingCache
from app.providers.embedding_base import EmbeddingProvider, InputType
from app.providers.retry import with_retries

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int], Awaitable[None]]


def _cache_key(model_id: str, text: str) -> str:
    return hashlib.sha256(f"{model_id}\x00{text}".encode()).hexdigest()


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class EmbeddingService:
    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        cache_enabled: bool = True,
        batch_size: int = 16,
        max_attempts: int = 3,
    ) -> None:
        self.provider = provider
        self.cache_enabled = cache_enabled
        self.batch_size = max(1, batch_size)
        self.max_attempts = max(1, max_attempts)

    @property
    def model_id(self) -> str:
        return self.provider.model_id

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    async def _lookup_cache(
        self, session: AsyncSession, keys: list[str], dimension: int
    ) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        if not self.cache_enabled or not keys:
            return found
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            rows = (
                await session.execute(
                    EmbeddingCache.__table__.select()
                    .where(EmbeddingCache.key.in_(batch))
                    .where(EmbeddingCache.model == self.model_id)
                    .where(EmbeddingCache.dimension == dimension)
                )
            ).all()
            for row in rows:
                found[row.key] = _unpack(row.vector)
        return found

    async def _store_cache(
        self, session: AsyncSession, items: list[tuple[str, list[float]]]
    ) -> None:
        if not self.cache_enabled or not items:
            return
        for start in range(0, len(items), 200):
            chunk = items[start : start + 200]
            statement = insert(EmbeddingCache).values(
                [
                    {
                        "key": key,
                        "model": self.model_id,
                        "dimension": len(vector),
                        "vector": _pack(vector),
                    }
                    for key, vector in chunk
                ]
            )
            await session.execute(statement.on_conflict_do_nothing(index_elements=["key"]))
        await session.commit()

    async def _embed_batch(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        vectors = await with_retries(
            lambda: self.provider.embed(texts, input_type),
            attempts=self.max_attempts,
            label="embedding_batch",
            retry_if=lambda exc: isinstance(exc, ProviderError) and exc.retryable,
        )
        if len(vectors) != len(texts):
            raise ProviderError(
                "Số lượng vector không khớp số lượng văn bản.",
                code="embedding_count_mismatch",
                retryable=True,
            )
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ProviderError(
                    f"Vector dimension {len(vector)} != configured {self.dimension}.",
                    code="embedding_dimension_mismatch",
                )
        return vectors

    async def embed_documents(
        self,
        session: AsyncSession,
        texts: list[str],
        *,
        progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        keys = [_cache_key(self.model_id, text) for text in texts]
        cached = await self._lookup_cache(session, keys, self.dimension)

        unique_missing: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in cached and key not in unique_missing:
                unique_missing[key] = text
        missing_keys = list(unique_missing.keys())
        total = len(missing_keys)
        done = 0

        for start in range(0, total, self.batch_size):
            batch_keys = missing_keys[start : start + self.batch_size]
            batch_texts = [unique_missing[key] for key in batch_keys]
            vectors = await self._embed_batch(batch_texts, "passage")
            await self._store_cache(session, list(zip(batch_keys, vectors, strict=True)))
            for key, vector in zip(batch_keys, vectors, strict=True):
                cached[key] = vector
            done += len(batch_keys)
            if progress is not None:
                await progress(done, total)

        result: list[list[float]] = []
        for key in keys:
            cached_vector = cached.get(key)
            if cached_vector is None:  # pragma: no cover - defensive
                raise ProviderError(
                    "Embedding cache miss after generation.", code="embedding_missing"
                )
            result.append(cached_vector)
        return result

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed_batch([text], "query")
        return vectors[0]

    async def ping(self) -> bool:
        return await self.provider.ping()

    async def aclose(self) -> None:
        await self.provider.aclose()
