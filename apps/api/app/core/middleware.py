"""Request correlation, security headers, and timing middleware."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.observability.logging import get_logger, request_id_ctx

logger = get_logger("app.http")
_SAFE_REQUEST_ID = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = (
            incoming[:64] if incoming and set(incoming) <= _SAFE_REQUEST_ID else uuid.uuid4().hex
        )
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "request_completed",
                extra={
                    "event": "request_completed",
                    "path": request.url.path,
                    "status_code": locals().get("response", None)
                    and getattr(locals()["response"], "status_code", 500),
                    "duration_ms": duration_ms,
                },
            )
            request_id_ctx.reset(token)
        response.headers["x-request-id"] = request_id
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["x-frame-options"] = "DENY"
        response.headers["referrer-policy"] = "no-referrer"
        response.headers["permissions-policy"] = "geolocation=(), microphone=(), camera=()"
        return response
