"""LLM provider factory. ``none`` is an explicit, non-fabricating capability."""

from __future__ import annotations

from app.core.config import Settings
from app.core.errors import ProviderError
from app.generation.base import LLMProvider
from app.generation.ollama import OllamaChatProvider
from app.generation.openai import OpenAICompatibleChatProvider


def build_llm_provider(settings: Settings) -> LLMProvider | None:
    if settings.llm_provider == "none":
        return None
    if settings.llm_provider == "ollama":
        return OllamaChatProvider(
            base_url=settings.llm_api_base,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.llm_provider == "openai":
        if not settings.llm_api_key:
            raise ProviderError(
                "LLM_API_KEY chưa được cấu hình.",
                code="llm_not_configured",
                status_code=503,
            )
        return OpenAICompatibleChatProvider(
            base_url=settings.llm_api_base,
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )
    raise ProviderError(
        f"LLM provider '{settings.llm_provider}' không được hỗ trợ.",
        code="llm_provider_unknown",
    )
