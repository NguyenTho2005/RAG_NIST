"""OpenAI-compatible and Ollama embedding adapters."""

from __future__ import annotations

import httpx

from app.core.errors import ProviderError
from app.observability.logging import get_logger
from app.providers.embedding_base import InputType

logger = get_logger(__name__)


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimension: int,
        timeout: float = 60.0,
    ) -> None:
        self.model_id = model
        self.dimension = dimension
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.post(
                "/embeddings", json={"model": self.model_id, "input": texts}
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                "Không kết nối được embedding provider.",
                code="embedding_unreachable",
                retryable=True,
            ) from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise ProviderError(
                f"Embedding provider trả về HTTP {response.status_code}.",
                code="embedding_upstream_error",
                retryable=True,
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"Embedding provider từ chối yêu cầu (HTTP {response.status_code}).",
                code="embedding_rejected",
                details={"status": response.status_code},
            )
        payload = response.json()
        data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in data]
        if len(vectors) != len(texts) or any(not isinstance(vector, list) for vector in vectors):
            raise ProviderError(
                "Embedding provider trả về số lượng vector không khớp.",
                code="embedding_invalid_response",
                retryable=True,
            )
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ProviderError(
                    f"Embedding dimension mismatch: got {len(vector)}, expected {self.dimension}.",
                    code="embedding_dimension_mismatch",
                )
        return vectors

    async def ping(self) -> bool:
        try:
            response = await self._client.get("/models", timeout=3)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


class OllamaEmbeddingProvider:
    def __init__(self, *, base_url: str, model: str, dimension: int, timeout: float = 60.0) -> None:
        self.model_id = model
        self.dimension = dimension
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.post(
                f"{self._base_url}/api/embed",
                json={"model": self.model_id, "input": texts},
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                "Không kết nối được Ollama.",
                code="embedding_unreachable",
                retryable=True,
            ) from exc
        if response.status_code == 404:
            raise ProviderError(
                f"Ollama chưa có model '{self.model_id}'. Chạy: ollama pull {self.model_id}",
                code="embedding_model_missing",
            )
        if response.status_code >= 500 or response.status_code == 429:
            raise ProviderError(
                f"Ollama embedding lỗi HTTP {response.status_code}.",
                code="embedding_upstream_error",
                retryable=True,
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"Ollama từ chối embedding request (HTTP {response.status_code}).",
                code="embedding_rejected",
            )
        vectors = response.json().get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ProviderError(
                "Ollama trả về số lượng vector không khớp.",
                code="embedding_invalid_response",
                retryable=True,
            )
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ProviderError(
                    f"Embedding dimension mismatch: got {len(vector)}, expected {self.dimension}.",
                    code="embedding_dimension_mismatch",
                )
        return vectors

    async def ping(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
