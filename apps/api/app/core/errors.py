"""Consistent, typed error semantics for the HTTP API."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.observability.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for expected application failures."""

    status_code = 400
    code = "bad_request"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.details = details or {}
        self.retryable = retryable


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class UnauthorizedError(AppError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(AppError):
    status_code = 403
    code = "forbidden"


class ValidationFailedError(AppError):
    status_code = 422
    code = "validation_failed"


class UnsupportedMediaError(AppError):
    status_code = 415
    code = "unsupported_media_type"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"


class ProviderError(AppError):
    status_code = 502
    code = "provider_error"
    retryable = True


class ProviderUnavailableError(AppError):
    status_code = 503
    code = "provider_unavailable"


class GenerationUnavailableError(AppError):
    status_code = 503
    code = "generation_unavailable"


class IngestionError(AppError):
    status_code = 422
    code = "ingestion_failed"


class SSRFBlockedError(AppError):
    status_code = 400
    code = "url_not_allowed"


def error_body(
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "retryable": retryable,
        },
        "request_id": request_id,
    }


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            error_body(
                code=exc.code,
                message=exc.message,
                details=exc.details,
                request_id=request_id_ctx.get(),
                retryable=exc.retryable,
            ),
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            error_body(
                code="validation_failed",
                message="Dữ liệu gửi lên không hợp lệ.",
                details={"errors": exc.errors()},
                request_id=request_id_ctx.get(),
            ),
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "http_error"
        if exc.status_code == 401:
            code = "unauthorized"
        elif exc.status_code == 403:
            code = "forbidden"
        elif exc.status_code == 404:
            code = "not_found"
        return JSONResponse(
            error_body(
                code=code,
                message=str(exc.detail),
                request_id=request_id_ctx.get(),
            ),
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled_exception",
            extra={"path": request.url.path, "event": "unhandled_exception"},
        )
        return JSONResponse(
            error_body(
                code="internal_error",
                message="Đã xảy ra lỗi hệ thống. Vui lòng thử lại sau.",
                request_id=request_id_ctx.get(),
            ),
            status_code=500,
        )
