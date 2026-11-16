"""Retrieval orchestration: dense + BM25, RRF fusion, optional reranking."""

from __future__ import annotations

import time
import uuid

from app.core.config import Settings
from app.core.errors import ProviderError
from app.embeddings.service import EmbeddingService
from app.indexing.vector_store import VectorStore
from app.persistence.db import Database
from app.reranking.base import Reranker
from app.retrieval.context import ContextSource, build_context
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.sparse import SparseRetriever
from app.retrieval.types import RetrievalOutcome


class RetrievalService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        embeddings: EmbeddingService,
        vectors: VectorStore,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.embeddings = embeddings
        self.vectors = vectors
        self.reranker = reranker
        self.dense = DenseRetriever(
            settings=settings,
            database=database,
            embeddings=embeddings,
            vectors=vectors,
        )
        self.sparse = SparseRetriever(
            settings=settings,
            database=database,
            embedding_model=embeddings.model_id,
        )

    async def search(
        self,
        *,
        workspace_id: uuid.UUID,
        kb_id: uuid.UUID,
        query: str,
        top_k: int,
        mode: str = "auto",
        document_ids: list[uuid.UUID] | None = None,
    ) -> tuple[RetrievalOutcome, list[ContextSource]]:
        requested = mode.lower()
        warnings: list[str] = []
        if requested in {"graph_local", "graph_global"}:
            warnings.append(
                f"Chế độ '{requested}' chưa được bật; dùng truy hồi hybrid có kiểm soát."
            )
            requested = "hybrid"
        if requested not in {"basic", "auto", "hybrid"}:
            requested = "auto"
        use_hybrid = requested != "basic"

        candidate_pool = max(self.settings.fusion_top_k, top_k)
        candidate_limit = max(self.settings.dense_top_k, candidate_pool)
        dense = await self.dense.search(
            workspace_id=workspace_id,
            kb_id=kb_id,
            query=query,
            limit=candidate_limit,
            document_ids=document_ids,
        )
        timings = dict(dense.timings)
        candidates = dict(dense.candidates)
        chunks = dense.chunks
        mode_used = "dense"

        if use_hybrid:
            sparse = await self.sparse.search(
                workspace_id=workspace_id,
                kb_id=kb_id,
                query=query,
                limit=candidate_limit,
                document_ids=document_ids,
            )
            timings.update(sparse.timings)
            candidates.update(sparse.candidates)
            fused = reciprocal_rank_fusion(
                [dense.chunks, sparse.chunks],
                k=self.settings.rrf_k,
                limit=candidate_pool,
            )
            candidates["fused"] = len(fused)
            chunks = fused
            mode_used = "hybrid"
            if not sparse.chunks:
                warnings.append(
                    "Không có ứng viên BM25; kết quả hybrid hiện dựa trên dense và RRF."
                )

        if self.reranker is not None and self.settings.rerank_enabled and chunks:
            rerank_started = time.perf_counter()
            try:
                top_n = min(len(chunks), max(top_k, self.settings.rerank_top_n))
                ranked = await self.reranker.rerank(
                    query,
                    [chunk.text for chunk in chunks],
                    top_n=top_n,
                )
                reordered = []
                for index, score in ranked:
                    if 0 <= index < len(chunks):
                        chunk = chunks[index]
                        chunk.score = float(score)
                        chunk.scores["rerank"] = float(score)
                        reordered.append(chunk)
                if reordered:
                    chunks = reordered
                    mode_used = f"{mode_used}+rerank"
                    candidates["reranked"] = len(reordered)
            except ProviderError as exc:
                warnings.append(f"Reranker không khả dụng ({exc.code}); đã dùng RRF.")
            timings["rerank_ms"] = round((time.perf_counter() - rerank_started) * 1000, 2)

        sources = build_context(
            chunks,
            token_budget=self.settings.context_token_budget,
            max_sources=top_k,
        )
        return (
            RetrievalOutcome(
                chunks=chunks,
                timings=timings,
                candidates=candidates,
                mode=mode,
                mode_used=mode_used,
                embedding_model=self.embeddings.model_id,
                warnings=[*dense.warnings, *warnings],
            ),
            sources,
        )

    def trace_config(self, *, top_k: int, mode: str) -> dict[str, object]:
        return {
            "mode_requested": mode,
            "top_k": top_k,
            "dense_top_k": self.settings.dense_top_k,
            "sparse_top_k": self.settings.sparse_top_k,
            "fusion_top_k": self.settings.fusion_top_k,
            "min_dense_score": self.settings.min_dense_score,
            "embedding_model": self.embeddings.model_id,
            "embedding_dimension": self.embeddings.dimension,
            "context_token_budget": self.settings.context_token_budget,
            "bm25_k1": self.settings.bm25_k1,
            "bm25_b": self.settings.bm25_b,
            "rrf_k": self.settings.rrf_k,
            "sparse_enabled": mode != "basic",
            "rerank_enabled": self.reranker is not None and self.settings.rerank_enabled,
            "reranker_model": self.settings.reranker_model,
            "graph_enabled": False,
        }
