"""Embedding provider factory."""

from __future__ import annotations

from app.core.config import Settings
from app.core.errors import ProviderError
from app.providers.embedding_base import EmbeddingProvider
from app.providers.local_embedding import LocalEmbeddingProvider
from app.providers.remote_embedding import (
    OllamaEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider(
            settings.embedding_model,
            settings.embedding_dimension,
            device=settings.embedding_device,
            revision=settings.embedding_model_revision,
            batch_size=settings.embedding_batch_size,
        )
    if settings.embedding_provider == "openai":
        if not settings.embedding_api_key:
            raise ProviderError(
                "EMBEDDING_API_KEY chưa được cấu hình.",
                code="embedding_not_configured",
                status_code=503,
            )
        return OpenAICompatibleEmbeddingProvider(
            base_url=settings.embedding_api_base,
            api_key=settings.embedding_api_key.get_secret_value(),
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            timeout=settings.embedding_timeout_seconds,
        )
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            timeout=settings.embedding_timeout_seconds,
        )
    raise ProviderError(
        f"Embedding provider '{settings.embedding_provider}' không được hỗ trợ.",
        code="embedding_provider_unknown",
    )
