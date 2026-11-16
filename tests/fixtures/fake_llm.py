"""Deterministic test-only LLM provider; never registered in production factory."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.generation.base import LLMResult, Message


class DeterministicFakeLLM:
    def __init__(
        self,
        *,
        answer: str = "Theo tài liệu, nhân viên được nghỉ 12 ngày phép năm [S1].",
        stream_chunks: list[str] | None = None,
    ) -> None:
        self.model_id = "test-fake-llm"
        self.answer = answer
        self.stream_chunks = stream_chunks or [
            "Theo tài liệu, ",
            "nhân viên ",
            "được nghỉ ",
            "12 ngày phép năm ",
            "[S1].",
        ]
        self.calls = 0
        self.stream_calls = 0
        self.last_messages: list[Message] = []

    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> LLMResult:
        self.calls += 1
        self.last_messages = messages
        return LLMResult(
            text=self.answer,
            model=self.model_id,
            usage={"input_tokens": 100, "output_tokens": 20},
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        self.stream_calls += 1
        self.last_messages = messages
        for chunk in self.stream_chunks:
            yield chunk

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None
