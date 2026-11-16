"""Bounded retry with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable

from app.core.errors import ProviderError
from app.observability.logging import get_logger

logger = get_logger(__name__)


async def with_retries[T](
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    label: str,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_if: Callable[[Exception], bool] | None = None,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            last_error = exc
            should_retry = (
                retry_if(exc) if retry_if else isinstance(exc, ProviderError) and exc.retryable
            )
            if not should_retry or attempt == attempts:
                raise
            jitter = 0.5 + secrets.randbelow(1000) / 1000
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) * jitter
            logger.warning(
                "provider_retry",
                extra={"event": label, "attempt": attempt, "delay_ms": int(delay * 1000)},
            )
            await asyncio.sleep(delay)
    assert last_error is not None
    raise last_error
