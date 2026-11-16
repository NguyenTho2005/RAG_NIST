"""Server-side URL fetching with SSRF controls and bounded reads."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import anyio
import httpx

from app.core.config import Settings
from app.core.errors import ForbiddenError, SSRFBlockedError

_REDIRECTS = {301, 302, 303, 307, 308}
_ALLOWED_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
}
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("198.18.0.0/15"),
]


@dataclass(slots=True)
class FetchedPage:
    content: bytes
    final_url: str
    content_type: str
    filename: str


def _is_public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    return not any(ip in network for network in _BLOCKED_NETWORKS)


async def validate_public_url(url: str, allowed_hosts: list[str]) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise SSRFBlockedError("Chỉ cho phép URL http/https.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise SSRFBlockedError("URL không hợp lệ hoặc chứa thông tin đăng nhập.")
    if parsed.port not in (None, 80, 443):
        raise SSRFBlockedError("Chỉ cho phép cổng 80 hoặc 443.")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname in {"localhost", "metadata.google.internal"} or hostname.endswith(".local"):
        raise SSRFBlockedError("Host nội bộ bị chặn.")
    if allowed_hosts and not any(
        hostname == allowed or hostname.endswith(f".{allowed.lstrip('.')}")
        for allowed in allowed_hosts
    ):
        raise ForbiddenError(
            "Host không nằm trong ALLOWED_URL_HOSTS.",
            code="url_host_not_allowed",
        )

    try:
        await anyio.to_thread.run_sync(
            lambda: socket.getaddrinfo(
                hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
            )
        )
    except socket.gaierror as exc:
        raise SSRFBlockedError("Không thể phân giải tên miền.") from exc

    addresses: set[str] = await anyio.to_thread.run_sync(
        lambda: {
            str(info[4][0])
            for info in socket.getaddrinfo(
                hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
            )
        }
    )
    if not addresses or not all(_is_public_ip(address) for address in addresses):
        raise SSRFBlockedError("URL trỏ tới địa chỉ mạng nội bộ và đã bị chặn.")
    return url


def _filename_from_url(url: str, content_type: str) -> str:
    path = urlparse(url).path
    name = path.rsplit("/", 1)[-1] or "page"
    if "." not in name:
        name = f"{name}.html" if "html" in content_type else f"{name}.txt"
    return name[:200]


async def fetch_url(url: str, settings: Settings, *, max_bytes: int | None = None) -> FetchedPage:
    """Fetch a public URL, validating every redirect and enforcing a byte cap."""
    if not settings.allowed_url_hosts:
        raise ForbiddenError(
            "URL ingestion chưa được bật. Cấu hình ALLOWED_URL_HOSTS để bật.",
            code="url_ingestion_disabled",
        )
    limit = max_bytes or settings.max_upload_bytes
    current = url
    headers = {"User-Agent": "AdvancedRAG/0.1 (+server-side ingestion)"}
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=settings.url_fetch_timeout_seconds,
        headers=headers,
    ) as client:
        for _ in range(6):
            await validate_public_url(current, settings.allowed_url_hosts)
            try:
                async with client.stream("GET", current) as response:
                    if response.status_code in _REDIRECTS:
                        location = response.headers.get("location")
                        if not location:
                            raise ForbiddenError(
                                "Redirect thiếu Location header.", code="url_redirect_invalid"
                            )
                        current = urljoin(current, location)
                        continue
                    if response.status_code >= 400:
                        raise ForbiddenError(
                            f"URL trả về HTTP {response.status_code}.",
                            code="url_fetch_failed",
                        )
                    content_type = response.headers.get("content-type", "").split(";")[0].lower()
                    if content_type not in _ALLOWED_CONTENT_TYPES:
                        raise ForbiddenError(
                            "Content-Type "
                            f"'{content_type or 'unknown'}' không được hỗ trợ cho URL.",
                            code="url_content_type_not_allowed",
                        )
                    declared_length = response.headers.get("content-length")
                    if declared_length and int(declared_length) > limit:
                        raise ForbiddenError(
                            "Nội dung URL vượt quá giới hạn.",
                            code="url_too_large",
                            status_code=413,
                        )
                    buffer = bytearray()
                    async for chunk in response.aiter_bytes():
                        buffer.extend(chunk)
                        if len(buffer) > limit:
                            raise ForbiddenError(
                                "Nội dung URL vượt quá giới hạn.",
                                code="url_too_large",
                                status_code=413,
                            )
                    return FetchedPage(
                        content=bytes(buffer),
                        final_url=str(response.url),
                        content_type=content_type,
                        filename=_filename_from_url(str(response.url), content_type),
                    )
            except httpx.HTTPError as exc:
                raise ForbiddenError(
                    "Không thể tải URL.", code="url_unreachable", status_code=502
                ) from exc
    raise ForbiddenError("Quá nhiều redirect.", code="url_too_many_redirects")
