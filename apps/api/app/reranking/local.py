"""Local cross-encoder reranker loaded lazily on CPU.

The default candidate is ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``
(Apache-2.0). Loading is never triggered by health probes; the first real search
with reranking enabled performs the download unless weights are cached locally.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, cast

import anyio

from app.core.errors import ProviderError
from app.observability.logging import get_logger

logger = get_logger(__name__)


class LocalCrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.model_id = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(1)

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - installed in dev extra
                raise ProviderError(
                    "Chưa cài sentence-transformers để rerank.",
                    code="reranker_dependency_missing",
                ) from exc
            logger.info(
                "reranker_model_loading",
                extra={"event": "reranker_model_loading", "model": self.model_id},
            )
            try:
                model = CrossEncoder(
                    self.model_id,
                    device=self.device,
                    trust_remote_code=False,
                )
            except Exception as exc:
                raise ProviderError(
                    f"Không thể tải reranker '{self.model_id}'.",
                    code="reranker_model_load_failed",
                    retryable=True,
                ) from exc
            self._model = model
            return model

    def _rerank_sync(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        model = self._load()
        pairs = [(query, document) for document in documents]
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        values = cast(list[float], [float(value) for value in scores])
        ranked = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
        return ranked[: max(1, top_n)]

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        if len(documents) > 512:
            raise ProviderError(
                "Reranker nhận quá nhiều ứng viên.", code="reranker_batch_too_large"
            )
        async with self._semaphore:
            try:
                return await anyio.to_thread.run_sync(self._rerank_sync, query, documents, top_n)
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(
                    "Lỗi khi rerank ứng viên.", code="reranker_failed", retryable=True
                ) from exc

    async def ping(self) -> bool:
        return self._model is not None

    async def aclose(self) -> None:
        self._model = None
