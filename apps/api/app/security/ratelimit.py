"""Rate limiting with Redis, plus an explicit in-process fallback."""

from __future__ import annotations

import asyncio
import time
from collections import deque

from redis.asyncio import Redis

from app.core.errors import RateLimitedError


class InMemoryRateLimiter:
    """Per-process sliding window. Used only when Redis is unavailable."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._events: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, scope: str) -> None:
        now = time.monotonic()
        async with self._lock:
            bucket = self._events.setdefault(scope, deque())
            while bucket and now - bucket[0] > 60:
                bucket.popleft()
            if len(bucket) >= self.limit:
                raise RateLimitedError("Quá nhiều yêu cầu. Vui lòng thử lại sau.")
            bucket.append(now)


class RedisRateLimiter:
    def __init__(self, redis: Redis, limit_per_minute: int) -> None:
        self.redis = redis
        self.limit = limit_per_minute

    async def check(self, scope: str) -> None:
        bucket = int(time.time() // 60)
        key = f"ratelimit:{scope}:{bucket}"
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, 70)
        if count > self.limit:
            raise RateLimitedError("Quá nhiều yêu cầu. Vui lòng thử lại sau.")
