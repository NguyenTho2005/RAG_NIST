"""Structure-aware, token-bounded chunking with parent/child hierarchy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from app.parsing.types import ParsedBlock, ParsedDocument
from app.retrieval.text import count_tokens, normalize_text, split_long_text

CHUNKER_VERSION = "2026.11.1"
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")


@dataclass(slots=True)
class SectionDraft:
    ordinal: int
    title: str
    level: int
    path: list[str]
    parent_ordinal: int | None
    page_start: int | None
    page_end: int | None
    blocks: list[ParsedBlock]


@dataclass(slots=True)
class ChunkDraft:
    kind: Literal["child", "parent"]
    ordinal: int
    text: str
    token_count: int
    heading_path: list[str]
    section_ordinal: int
    page_start: int | None
    page_end: int | None
    source_spans: list[dict[str, Any]]
    parent_index: int | None = None
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkingResult:
    sections: list[SectionDraft]
    chunks: list[ChunkDraft]
    chunker_version: str = CHUNKER_VERSION


def build_sections(document: ParsedDocument) -> list[SectionDraft]:
    sections: list[SectionDraft] = []
    stack: list[tuple[int, str, int]] = []  # level, title, section ordinal
    current: SectionDraft | None = None

    def start_section(title: str, level: int, page: int | None, block: ParsedBlock) -> None:
        nonlocal current
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][2] if stack else None
        current = SectionDraft(
            ordinal=len(sections),
            title=title,
            level=level,
            path=[entry[1] for entry in stack] + [title] if title else [],
            parent_ordinal=parent,
            page_start=page,
            page_end=page,
            blocks=[block],
        )
        sections.append(current)
        stack.append((level, title, current.ordinal))

    for block in document.blocks:
        if block.is_heading:
            if current is not None and len(current.blocks) == 1 and current.blocks[0].is_heading:
                # Consecutive heading: replace title while keeping hierarchy.
                title = block.text
                level = block.heading_level or 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                parent = stack[-1][2] if stack else None
                current = SectionDraft(
                    ordinal=len(sections),
                    title=title,
                    level=level,
                    path=[entry[1] for entry in stack] + [title],
                    parent_ordinal=parent,
                    page_start=block.page,
                    page_end=block.page,
                    blocks=[block],
                )
                sections.append(current)
                stack.append((level, title, current.ordinal))
                continue
            start_section(block.text, block.heading_level or 1, block.page, block)
            continue
        if current is None:
            current = SectionDraft(
                ordinal=len(sections),
                title=document.title,
                level=1,
                path=[document.title] if document.title else [],
                parent_ordinal=None,
                page_start=block.page,
                page_end=block.page,
                blocks=[],
            )
            sections.append(current)
        current.blocks.append(block)
        current.page_end = block.page or current.page_end

    if not sections and document.blocks:
        sections.append(
            SectionDraft(
                ordinal=0,
                title=document.title,
                level=1,
                path=[document.title],
                parent_ordinal=None,
                page_start=document.blocks[0].page,
                page_end=document.blocks[-1].page,
                blocks=list(document.blocks),
            )
        )
    return sections


def _span(block: ParsedBlock, reason: str = "content") -> dict[str, Any]:
    span: dict[str, Any] = {
        "block_ordinal": block.ordinal,
        "char_start": block.char_start,
        "char_end": block.char_end,
        "kind": block.kind,
        "reason": reason,
    }
    if block.page is not None:
        span["page"] = block.page
    if block.table_id:
        span["table_id"] = block.table_id
        span["table_row"] = block.table_row
    return span


def _sentence_tail(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    sentences = [part for part in _SENTENCE_RE.split(text.strip()) if part]
    tail: list[str] = []
    used = 0
    for sentence in reversed(sentences):
        tokens = count_tokens(sentence)
        if used + tokens > max_tokens:
            break
        tail.insert(0, sentence)
        used += tokens
    return " ".join(tail)


def _header_prefix(block: ParsedBlock) -> tuple[str, int]:
    header = block.table_header
    if not header:
        return "", 0
    line = " | ".join(cell for cell in header if cell)
    if not line:
        return "", 0
    return f"{line}\n", count_tokens(line) + 1


class StructureAwareChunker:
    def __init__(
        self,
        *,
        target_tokens: int = 384,
        max_tokens: int = 512,
        overlap_tokens: int = 48,
        parent_max_tokens: int = 1200,
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be < target_tokens")
        if target_tokens > max_tokens:
            raise ValueError("target_tokens must be <= max_tokens")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.parent_max_tokens = parent_max_tokens

    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        sections = build_sections(document)
        drafts: list[ChunkDraft] = []
        child_drafts: list[ChunkDraft] = []

        for section in sections:
            section_chunks = self._section_children(section)
            if not section_chunks:
                continue
            # Every retrievable child keeps a section-level parent so context
            # expansion works uniformly, including for short sections.
            self._attach_parents(section, section_chunks, drafts, child_drafts)

        drafts.extend(child_drafts)
        for ordinal, draft in enumerate(drafts):
            draft.ordinal = ordinal
            draft.content_hash = (
                __import__("hashlib").sha256(draft.text.encode("utf-8")).hexdigest()
            )
        return ChunkingResult(sections=sections, chunks=drafts)

    def _section_children(self, section: SectionDraft) -> list[ChunkDraft]:
        prefix = " > ".join(section.path) if section.path else ""
        prefix_text = f"{prefix}\n" if prefix else ""
        prefix_tokens = count_tokens(prefix_text)
        heading_spans = [
            _span(block, "heading_context") for block in section.blocks if block.is_heading
        ]

        blocks = [block for block in section.blocks if not block.is_heading]
        chunks: list[ChunkDraft] = []
        current_blocks: list[ParsedBlock] = []
        current_spans: list[dict[str, Any]] = []
        current_tokens = prefix_tokens
        current_parts: list[str] = []
        overlap_text = ""
        overlap_tokens = 0
        page_start: int | None = None
        page_end: int | None = None
        last_table_id: str | None = None

        def reset(keep_overlap: bool) -> None:
            nonlocal current_blocks, current_spans, current_tokens, current_parts
            nonlocal page_start, page_end, overlap_text, overlap_tokens, last_table_id
            tail = (
                _sentence_tail(" ".join(current_parts), self.overlap_tokens) if keep_overlap else ""
            )
            current_blocks = []
            current_spans = []
            current_parts = []
            current_tokens = prefix_tokens
            page_start = None
            page_end = None
            last_table_id = None
            overlap_text = tail
            overlap_tokens = count_tokens(tail) if tail else 0
            if tail:
                current_parts.append(tail)
                current_tokens += overlap_tokens

        def emit() -> None:
            nonlocal page_start, page_end
            if not current_blocks:
                return
            body = "\n\n".join(current_parts).strip()
            text = normalize_text(f"{prefix_text}{body}")
            if not text:
                return
            token_count = count_tokens(text)
            if token_count > self.max_tokens:
                # Defensive: never publish an oversized chunk.
                for piece in split_long_text(text, self.max_tokens):
                    if not piece.strip():
                        continue
                    chunks.append(
                        ChunkDraft(
                            kind="child",
                            ordinal=0,
                            text=normalize_text(piece),
                            token_count=count_tokens(piece),
                            heading_path=list(section.path),
                            section_ordinal=section.ordinal,
                            page_start=page_start,
                            page_end=page_end,
                            source_spans=[*current_spans, *heading_spans],
                        )
                    )
            else:
                chunks.append(
                    ChunkDraft(
                        kind="child",
                        ordinal=0,
                        text=text,
                        token_count=token_count,
                        heading_path=list(section.path),
                        section_ordinal=section.ordinal,
                        page_start=page_start,
                        page_end=page_end,
                        source_spans=[*current_spans, *heading_spans],
                    )
                )

        for block in blocks:
            block_tokens = count_tokens(block.text)
            header_text, header_tokens = ("", 0)
            if (
                block.kind == "table"
                and block.table_id != last_table_id
                and block.table_row not in (0, None)
            ):
                header_text, header_tokens = _header_prefix(block)
            needed = block_tokens + header_tokens + (2 if current_parts else 0)

            exceeds = current_tokens + needed > self.target_tokens
            hard_exceeds = current_tokens + needed > self.max_tokens
            if current_blocks and (exceeds or hard_exceeds):
                emit()
                reset(keep_overlap=True)
                if block.kind == "table":
                    header_text, header_tokens = _header_prefix(block)
                    needed = block_tokens + header_tokens + (2 if current_parts else 0)
                    last_table_id = None

            if overlap_text and current_tokens + needed > self.max_tokens:
                current_parts = [part for part in current_parts if part != overlap_text]
                current_tokens -= overlap_tokens
                overlap_text = ""
                overlap_tokens = 0
                needed = block_tokens + header_tokens + (2 if current_parts else 0)

            # A single pathological block: split, preserving provenance.
            if prefix_tokens + header_tokens + block_tokens > self.max_tokens:
                body = f"{header_text}{block.text}"
                available = max(32, self.max_tokens - prefix_tokens)
                for piece in split_long_text(body, available):
                    piece = normalize_text(piece)
                    if not piece:
                        continue
                    start_index = block.text.find(piece)
                    span = _span(block)
                    if start_index >= 0:
                        span["char_start"] = block.char_start + start_index
                        span["char_end"] = block.char_start + start_index + len(piece)
                    chunks.append(
                        ChunkDraft(
                            kind="child",
                            ordinal=0,
                            text=normalize_text(f"{prefix_text}{piece}"),
                            token_count=count_tokens(f"{prefix_text}{piece}"),
                            heading_path=list(section.path),
                            section_ordinal=section.ordinal,
                            page_start=block.page,
                            page_end=block.page,
                            source_spans=[span, *heading_spans],
                        )
                    )
                current_blocks = []
                current_spans = []
                current_parts = []
                current_tokens = prefix_tokens
                overlap_text = ""
                overlap_tokens = 0
                continue

            if header_text:
                current_parts.append(header_text.rstrip("\n"))
                current_tokens += header_tokens
            current_parts.append(block.text)
            current_blocks.append(block)
            current_spans.append(_span(block))
            current_tokens += block_tokens
            page_start = page_start or block.page
            page_end = block.page or page_end
            if block.table_id:
                last_table_id = block.table_id

        emit()
        return chunks

    def _attach_parents(
        self,
        section: SectionDraft,
        children: list[ChunkDraft],
        drafts: list[ChunkDraft],
        child_drafts: list[ChunkDraft],
    ) -> None:
        groups: list[list[ChunkDraft]] = []
        current: list[ChunkDraft] = []
        tokens = 0
        for child in children:
            if current and tokens + child.token_count > self.parent_max_tokens:
                groups.append(current)
                current = []
                tokens = 0
            current.append(child)
            tokens += child.token_count
        if current:
            groups.append(current)

        for group in groups:
            first = group[0]
            last = group[-1]
            seen_spans: list[dict[str, Any]] = []
            seen_keys: set[tuple[int | None, int | None, str]] = set()
            for child in group:
                for span in child.source_spans:
                    key = (
                        span.get("block_ordinal"),
                        span.get("char_start"),
                        span.get("reason", ""),
                    )
                    if key not in seen_keys:
                        seen_keys.add(key)
                        seen_spans.append(span)
            parent_text = "\n\n".join(child.text for child in group)
            parent = ChunkDraft(
                kind="parent",
                ordinal=0,
                text=parent_text,
                token_count=count_tokens(parent_text),
                heading_path=list(section.path),
                section_ordinal=section.ordinal,
                page_start=first.page_start,
                page_end=last.page_end,
                source_spans=seen_spans,
                metadata={"group_size": len(group)},
            )
            parent_ref = len(drafts)
            drafts.append(parent)
            for child in group:
                child.parent_index = parent_ref
            child_drafts.extend(group)
