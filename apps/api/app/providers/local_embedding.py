"""Local sentence-transformers embeddings, loaded lazily on CPU by default."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, cast

import anyio

from app.core.errors import ProviderError
from app.observability.logging import get_logger
from app.providers.embedding_base import InputType

logger = get_logger(__name__)


class LocalEmbeddingProvider:
    def __init__(
        self,
        model_name: str,
        dimension: int,
        *,
        device: str = "cpu",
        revision: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.model_id = f"{model_name}@{revision}" if revision else model_name
        self.dimension = dimension
        self.device = device
        self.revision = revision
        self.batch_size = batch_size
        self._model = None
        self._load_lock = threading.Lock()
        self._encode_semaphore = asyncio.Semaphore(2)

    def _uses_e5_prefixes(self) -> bool:
        return "e5" in self.model_name.lower()

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dependency is installed
                raise ProviderError(
                    "Local embedding extra is not installed; install sentence-transformers.",
                    code="embedding_dependency_missing",
                ) from exc
            logger.info(
                "embedding_model_loading",
                extra={"event": "embedding_model_loading", "model": self.model_id},
            )
            kwargs = {"device": self.device, "trust_remote_code": False}
            if self.revision:
                kwargs["revision"] = self.revision
            try:
                model = SentenceTransformer(self.model_name, **kwargs)
            except Exception as exc:
                raise ProviderError(
                    f"Không thể tải embedding model '{self.model_id}'. "
                    "Kiểm tra kết nối Hugging Face hoặc cấu hình model cục bộ.",
                    code="embedding_model_load_failed",
                    retryable=True,
                ) from exc
            actual = model.get_sentence_embedding_dimension()
            if actual is not None and int(actual) != self.dimension:
                raise ProviderError(
                    f"Embedding dimension mismatch: model={actual}, configured={self.dimension}.",
                    code="embedding_dimension_mismatch",
                )
            self._model = model
            return model

    def _encode(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        model = self._load()
        prefix = "query: " if input_type == "query" else "passage: "
        prepared = [f"{prefix}{text}" for text in texts] if self._uses_e5_prefixes() else texts
        vectors = model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        result = cast(list[list[float]], vectors.tolist())
        for vector in result:
            if len(vector) != self.dimension:
                raise ProviderError(
                    f"Embedding dimension mismatch: got {len(vector)}, expected {self.dimension}.",
                    code="embedding_dimension_mismatch",
                )
        return result

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > 512:
            raise ProviderError(
                "Embedding batch exceeds 512 texts.", code="embedding_batch_too_large"
            )
        async with self._encode_semaphore:
            try:
                return await anyio.to_thread.run_sync(self._encode, texts, input_type)
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(
                    "Lỗi khi tạo embedding cục bộ.", code="embedding_failed", retryable=True
                ) from exc

    async def ping(self) -> bool:
        # Do not trigger a model download during readiness checks.
        return self._model is not None

    async def aclose(self) -> None:
        self._model = None
