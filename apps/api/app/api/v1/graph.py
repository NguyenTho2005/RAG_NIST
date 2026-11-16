"""Graph endpoint. GraphRAG is gated and reports unavailability honestly."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.security.auth import Principal, require_read

router = APIRouter()


class GraphStatus(BaseModel):
    enabled: bool
    available: bool
    reason: str
    entities: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)


@router.get("/graph", response_model=GraphStatus, summary="GraphRAG capability status")
async def graph_status(
    request: Request, principal: Principal = Depends(require_read)
) -> GraphStatus:
    settings = request.app.state.settings
    if not settings.graph_enabled:
        return GraphStatus(
            enabled=False,
            available=False,
            reason=(
                "GraphRAG đang tắt (GRAPH_ENABLED=false). Bật có kiểm soát sau khi có pipeline "
                "trích xuất thực thể/quan hệ và đánh giá chất lượng."
            ),
        )
    return GraphStatus(
        enabled=True,
        available=False,
        reason=(
            "GRAPH_ENABLED=true nhưng pipeline GraphRAG chưa được triển khai trong bản này; "
            "không trả dữ liệu giả."
        ),
    )
