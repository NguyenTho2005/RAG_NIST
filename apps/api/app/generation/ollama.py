"""Ollama chat provider (local or remote Ollama server)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.core.errors import ProviderError
from app.generation.base import LLMResult, Message


class OllamaChatProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def _payload(
        self, messages: list[Message], *, temperature: float, max_tokens: int, stream: bool
    ) -> dict[str, object]:
        return {
            "model": self.model_id,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> LLMResult:
        try:
            response = await self._http().post(
                "/api/chat",
                json=self._payload(
                    messages, temperature=temperature, max_tokens=max_tokens, stream=False
                ),
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(
                "Không thể gọi Ollama. Kiểm tra LLM_API_BASE và model đã được tải chưa.",
                code="llm_request_failed",
                retryable=True,
            ) from exc
        message = data.get("message") or {}
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Ollama trả về câu trả lời rỗng.", code="llm_empty_response")
        return LLMResult(
            text=text,
            model=str(data.get("model") or self.model_id),
            usage={
                "input_tokens": int(data.get("prompt_eval_count") or 0),
                "output_tokens": int(data.get("eval_count") or 0),
            },
            finish_reason="stop" if data.get("done") else None,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        try:
            async with self._http().stream(
                "POST",
                "/api/chat",
                json=self._payload(
                    messages, temperature=temperature, max_tokens=max_tokens, stream=True
                ),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        raise ProviderError(
                            f"Ollama báo lỗi: {data['error']}", code="llm_stream_failed"
                        )
                    text = (data.get("message") or {}).get("content")
                    if text:
                        yield text
                    if data.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise ProviderError(
                "Không thể stream câu trả lời từ Ollama.",
                code="llm_stream_failed",
                retryable=True,
            ) from exc

    async def ping(self) -> bool:
        try:
            response = await self._http().get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models") or []
        except Exception:
            return False
        names = {str(item.get("name")) for item in models if isinstance(item, dict)}
        if self.model_id in names:
            return True
        return any(name.startswith(f"{self.model_id}:") for name in names)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
