"""Versioned API router registration."""

from fastapi import APIRouter

from app.api.v1 import (
    chat,
    documents,
    evaluations,
    graph,
    health,
    jobs,
    knowledge_bases,
    models,
    search,
    settings,
    stats,
    traces,
)

api_router = APIRouter()
api_router.include_router(health.router, tags=["system"])
api_router.include_router(knowledge_bases.router, tags=["knowledge-bases"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(jobs.router, tags=["jobs"])
api_router.include_router(search.router, tags=["retrieval"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(traces.router, tags=["traces"])
api_router.include_router(models.router, tags=["models"])
api_router.include_router(settings.router, tags=["settings"])
api_router.include_router(graph.router, tags=["graph"])
api_router.include_router(evaluations.router, tags=["evaluation"])
api_router.include_router(stats.router, tags=["stats"])
