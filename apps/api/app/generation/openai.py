"""OpenAI-compatible chat completions provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.core.errors import ProviderError
from app.generation.base import LLMResult, Message


class OpenAICompatibleChatProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        return self._client

    def _payload(
        self, messages: list[Message], *, temperature: float, max_tokens: int, stream: bool
    ) -> dict[str, object]:
        return {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
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
                "/chat/completions",
                json=self._payload(
                    messages, temperature=temperature, max_tokens=max_tokens, stream=False
                ),
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(
                "Không thể gọi LLM tương thích OpenAI.",
                code="llm_request_failed",
                retryable=True,
            ) from exc
        choices = data.get("choices") or []
        content = ((choices[0].get("message") or {}).get("content")) if choices else None
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("LLM trả về câu trả lời rỗng.", code="llm_empty_response")
        usage = data.get("usage") or {}
        return LLMResult(
            text=content,
            model=str(data.get("model") or self.model_id),
            usage={
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
            },
            finish_reason=str(choices[0].get("finish_reason") or "") or None,
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
                "/chat/completions",
                json=self._payload(
                    messages, temperature=temperature, max_tokens=max_tokens, stream=True
                ),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices") or []
                    text = ((choices[0].get("delta") or {}).get("content")) if choices else None
                    if text:
                        yield text
        except httpx.HTTPError as exc:
            raise ProviderError(
                "Không thể stream câu trả lời từ LLM.",
                code="llm_stream_failed",
                retryable=True,
            ) from exc

    async def ping(self) -> bool:
        try:
            response = await self._http().get("/models")
            response.raise_for_status()
            models = response.json().get("data") or []
        except Exception:
            return False
        return any(item.get("id") == self.model_id for item in models if isinstance(item, dict))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
