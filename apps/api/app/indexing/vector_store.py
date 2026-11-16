"""Qdrant collection management and filtered vector operations."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from app.core.errors import ProviderError
from app.observability.logging import get_logger

logger = get_logger(__name__)
_NAMESPACE = uuid.UUID("4f4f6c7e-6f5a-4a1d-9a12-2c1b7d3a9e10")


def point_id(model_id: str, chunk_id: uuid.UUID | str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{model_id}:{chunk_id}"))


class VectorStore:
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        dimension: int,
        model_id: str,
    ) -> None:
        self.client = client
        self.collection = collection
        self.dimension = dimension
        self.model_id = model_id

    async def ensure_collection(self) -> None:
        collections = {item.name for item in (await self.client.get_collections()).collections}
        if self.collection not in collections:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
            for field, schema in (
                ("workspace_id", PayloadSchemaType.KEYWORD),
                ("knowledge_base_id", PayloadSchemaType.KEYWORD),
                ("document_id", PayloadSchemaType.KEYWORD),
                ("revision_id", PayloadSchemaType.KEYWORD),
                ("embedding_model", PayloadSchemaType.KEYWORD),
                ("kind", PayloadSchemaType.KEYWORD),
                ("active", PayloadSchemaType.BOOL),
            ):
                await self.client.create_payload_index(
                    collection_name=self.collection, field_name=field, field_schema=schema
                )
            logger.info(
                "qdrant_collection_created",
                extra={"event": "qdrant_collection_created"},
            )
            return
        info = await self.client.get_collection(self.collection)
        params = info.config.params.vectors
        existing_dim = getattr(params, "size", None)
        if existing_dim is not None and int(existing_dim) != self.dimension:
            raise ProviderError(
                f"Qdrant collection '{self.collection}' có dimension {existing_dim}, "
                f"cấu hình yêu cầu {self.dimension}. Dùng collection mới và reindex.",
                code="vector_dimension_mismatch",
            )

    def _base_filter(self, workspace_id: uuid.UUID, kb_id: uuid.UUID) -> list[Any]:
        return [
            FieldCondition(key="workspace_id", match=MatchValue(value=str(workspace_id))),
            FieldCondition(key="knowledge_base_id", match=MatchValue(value=str(kb_id))),
            FieldCondition(key="embedding_model", match=MatchValue(value=self.model_id)),
            FieldCondition(key="kind", match=MatchValue(value="child")),
        ]

    def search_filter(
        self,
        workspace_id: uuid.UUID,
        kb_id: uuid.UUID,
        *,
        active: bool = True,
        document_ids: list[uuid.UUID] | None = None,
    ) -> Filter:
        must = self._base_filter(workspace_id, kb_id)
        must.append(FieldCondition(key="active", match=MatchValue(value=active)))
        if document_ids:
            must.append(
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=[str(value) for value in document_ids]),
                )
            )
        return Filter(must=must)

    async def upsert(
        self,
        *,
        workspace_id: uuid.UUID,
        kb_id: uuid.UUID,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        filename: str,
        items: list[dict[str, Any]],
    ) -> list[str]:
        """Upsert embeddings; items require chunk_id, vector, token_count, etc."""
        ids: list[str] = []
        points: list[PointStruct] = []
        for item in items:
            chunk_id = item["chunk_id"]
            pid = point_id(self.model_id, chunk_id)
            ids.append(pid)
            points.append(
                PointStruct(
                    id=pid,
                    vector=item["vector"],
                    payload={
                        "workspace_id": str(workspace_id),
                        "knowledge_base_id": str(kb_id),
                        "document_id": str(document_id),
                        "revision_id": str(revision_id),
                        "chunk_id": str(chunk_id),
                        "kind": "child",
                        "active": False,
                        "embedding_model": self.model_id,
                        "content_hash": item.get("content_hash"),
                        "token_count": item.get("token_count"),
                        "ordinal": item.get("ordinal"),
                        "page_start": item.get("page_start"),
                        "page_end": item.get("page_end"),
                        "filename": filename,
                    },
                )
            )
        for start in range(0, len(points), 64):
            batch = points[start : start + 64]
            try:
                await self.client.upsert(collection_name=self.collection, points=batch, wait=True)
            except Exception as exc:
                raise ProviderError(
                    "Không thể ghi vector vào Qdrant.",
                    code="qdrant_upsert_failed",
                    retryable=True,
                    details={"error": f"{type(exc).__name__}: {exc}"},
                ) from exc
        return ids

    async def set_active(self, ids: list[str], *, active: bool) -> None:
        if not ids:
            return
        for start in range(0, len(ids), 256):
            batch = ids[start : start + 256]
            await self.client.set_payload(
                collection_name=self.collection,
                payload={"active": active},
                points=PointIdsList(points=batch),
                wait=True,
            )

    async def _set_active_filter(self, condition: Filter, *, active: bool) -> None:
        from qdrant_client.models import FilterSelector

        try:
            await self.client.set_payload(
                collection_name=self.collection,
                payload={"active": active},
                points=FilterSelector(filter=condition),
                wait=True,
            )
        except Exception as exc:
            raise ProviderError(
                "Không thể cập nhật trạng thái vector trong Qdrant.",
                code="qdrant_payload_update_failed",
                retryable=True,
            ) from exc

    async def activate_revision(
        self, workspace_id: uuid.UUID, kb_id: uuid.UUID, revision_id: uuid.UUID
    ) -> None:
        await self._set_active_filter(
            Filter(
                must=[
                    FieldCondition(key="workspace_id", match=MatchValue(value=str(workspace_id))),
                    FieldCondition(key="knowledge_base_id", match=MatchValue(value=str(kb_id))),
                    FieldCondition(key="revision_id", match=MatchValue(value=str(revision_id))),
                    FieldCondition(key="embedding_model", match=MatchValue(value=self.model_id)),
                ]
            ),
            active=True,
        )

    async def deactivate_revision(self, revision_id: uuid.UUID) -> None:
        await self._set_active_filter(
            Filter(
                must=[
                    FieldCondition(key="revision_id", match=MatchValue(value=str(revision_id))),
                    FieldCondition(key="embedding_model", match=MatchValue(value=self.model_id)),
                ]
            ),
            active=False,
        )

    async def deactivate_document(self, workspace_id: uuid.UUID, document_id: uuid.UUID) -> None:
        await self._set_active_filter(
            Filter(
                must=[
                    FieldCondition(key="workspace_id", match=MatchValue(value=str(workspace_id))),
                    FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
                ]
            ),
            active=False,
        )

    async def delete_by_filter(self, condition: Filter) -> None:
        try:
            await self.client.delete(
                collection_name=self.collection,
                points_selector=condition,
                wait=True,
            )
        except Exception as exc:
            raise ProviderError(
                "Không thể xoá vector khỏi Qdrant.",
                code="qdrant_delete_failed",
                retryable=True,
            ) from exc

    async def delete_document(self, workspace_id: uuid.UUID, document_id: uuid.UUID) -> None:
        await self.delete_by_filter(
            Filter(
                must=[
                    FieldCondition(key="workspace_id", match=MatchValue(value=str(workspace_id))),
                    FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
                ]
            )
        )

    async def delete_revision(self, revision_id: uuid.UUID) -> None:
        await self.delete_by_filter(
            Filter(
                must=[FieldCondition(key="revision_id", match=MatchValue(value=str(revision_id)))]
            )
        )

    async def search(
        self,
        *,
        workspace_id: uuid.UUID,
        kb_id: uuid.UUID,
        vector: list[float],
        limit: int,
        score_threshold: float | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[Any]:
        try:
            response = await self.client.query_points(
                collection_name=self.collection,
                query=vector,
                query_filter=self.search_filter(
                    workspace_id, kb_id, active=True, document_ids=document_ids
                ),
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold,
            )
            return list(response.points)
        except Exception as exc:
            raise ProviderError(
                "Truy vấn Qdrant thất bại.",
                code="qdrant_query_failed",
                retryable=True,
            ) from exc

    async def exists(self) -> bool:
        try:
            return bool(await self.client.collection_exists(self.collection))
        except Exception as exc:
            raise ProviderError(
                "Không thể kiểm tra Qdrant collection.",
                code="qdrant_collection_check_failed",
                retryable=True,
            ) from exc

    async def count(self) -> int:
        result = await self.client.count(collection_name=self.collection, exact=False)
        return int(result.count)
