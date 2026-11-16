"""Context assembly with an explicit token budget.

Retrieved text is untrusted data. It is wrapped in numbered source blocks and
never inserted into the system instruction itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.retrieval.text import count_tokens, normalize_text
from app.retrieval.types import RetrievedChunk

_NO_CONTEXT_TEXT = "Tôi không tìm thấy thông tin liên quan trong các tài liệu đã được lập chỉ mục."


@dataclass(slots=True)
class ContextSource:
    index: int
    marker: str
    chunk: RetrievedChunk
    text: str
    truncated: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


def source_label(source: ContextSource) -> str:
    chunk = source.chunk
    parts = [f"[{source.marker}] Tệp: {chunk.filename}"]
    if chunk.heading_path:
        parts.append(f"Mục: {' > '.join(chunk.heading_path)}")
    if chunk.page_start is not None:
        page = f"Trang: {chunk.page_start}"
        if chunk.page_end and chunk.page_end != chunk.page_start:
            page += f"-{chunk.page_end}"
        parts.append(page)
    return " | ".join(parts)


def build_context(
    chunks: list[RetrievedChunk],
    *,
    token_budget: int,
    max_sources: int | None = None,
) -> list[ContextSource]:
    """Select chunks in rank order until the formatted context budget is full."""
    if token_budget <= 0:
        return []
    sources: list[ContextSource] = []
    used = 0
    seen: set[str] = set()
    for chunk in chunks:
        if max_sources is not None and len(sources) >= max_sources:
            break
        text = normalize_text(chunk.text)
        if not text or text in seen:
            continue
        index = len(sources) + 1
        source = ContextSource(index=index, marker=f"S{index}", chunk=chunk, text=text)
        block = f"{source_label(source)}\n{text}\n"
        block_tokens = count_tokens(block)
        remaining = token_budget - used
        if block_tokens > remaining:
            if not sources or remaining < 80:
                continue
            # Keep a partial source only when it still contributes meaningfully.
            available = max(20, remaining - 20)
            words = text.split()
            truncated = ""
            for word in words:
                candidate = f"{truncated} {word}".strip()
                if count_tokens(candidate) > available:
                    break
                truncated = candidate
            if not truncated:
                continue
            source.text = truncated
            source.truncated = True
            block_tokens = count_tokens(f"{source_label(source)}\n{truncated}\n")
        sources.append(source)
        seen.add(text)
        used += block_tokens
    return sources


def format_context(sources: list[ContextSource]) -> str:
    if not sources:
        return ""
    blocks = []
    for source in sources:
        blocks.append(f"{source_label(source)}\n{source.text}")
    return "\n\n".join(blocks)


def source_metadata(source: ContextSource) -> dict[str, object]:
    chunk = source.chunk
    return {
        "marker": f"[{source.marker}]",
        "source_index": source.index,
        "chunk_id": str(chunk.chunk_id),
        "document_id": str(chunk.document_id),
        "revision_id": str(chunk.revision_id),
        "filename": chunk.filename,
        "heading_path": list(chunk.heading_path),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "score": round(chunk.score, 6),
        "source": chunk.source,
    }


__all__ = [
    "_NO_CONTEXT_TEXT",
    "ContextSource",
    "build_context",
    "format_context",
    "source_label",
    "source_metadata",
]
