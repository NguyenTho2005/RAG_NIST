"""Unit tests for configuration and security primitives."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import RateLimitedError
from app.security.auth import generate_api_key, hash_api_key
from app.security.ratelimit import InMemoryRateLimiter


def test_config_rejects_excessive_overlap() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        Settings(_env_file=None, chunk_target_tokens=100, chunk_overlap_tokens=100)


def test_config_requires_key_for_openai_embedding() -> None:
    with pytest.raises(ValidationError, match="EMBEDDING_API_KEY"):
        Settings(_env_file=None, embedding_provider="openai", embedding_api_key=None)


def test_config_lists_parse_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    monkeypatch.setenv("ALLOWED_URL_HOSTS", "example.com")
    settings = Settings(_env_file=None)
    assert settings.cors_origins == ["http://a.test", "http://b.test"]
    assert settings.allowed_url_hosts == ["example.com"]


def test_api_key_is_not_stored_in_plaintext() -> None:
    key, prefix, digest = generate_api_key()
    assert key.startswith("rag_")
    assert prefix == key[:14]
    assert digest == hash_api_key(key)
    assert key not in digest
    assert len(digest) == 64


async def test_in_memory_rate_limiter_enforces_limit() -> None:
    limiter = InMemoryRateLimiter(limit_per_minute=2)
    await limiter.check("key")
    await limiter.check("key")
    with pytest.raises(RateLimitedError):
        await limiter.check("key")
