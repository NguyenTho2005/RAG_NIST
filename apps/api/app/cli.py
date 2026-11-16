"""Operational CLI: workspace bootstrap, Qdrant initialization, demo seeding."""

from __future__ import annotations

import argparse
import asyncio
import re
import sys

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams
from sqlalchemy import select

from app.core.config import get_settings
from app.observability.logging import configure_logging
from app.persistence.db import Database
from app.persistence.models import ApiKey, KnowledgeBase, Workspace
from app.security.auth import generate_api_key


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:80] or "workspace"


async def create_workspace(name: str, slug: str | None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    db = Database(settings.database_url)
    key, prefix, key_hash = generate_api_key()
    async with db.sessionmaker() as session:
        existing = (
            await session.execute(
                select(Workspace).where(Workspace.slug == (slug or _slugify(name)))
            )
        ).scalar_one_or_none()
        if existing is not None:
            print(f"Workspace slug already exists: {existing.slug}", file=sys.stderr)
            await db.dispose()
            raise SystemExit(1)
        workspace = Workspace(name=name, slug=slug or _slugify(name))
        session.add(workspace)
        await session.flush()
        session.add(
            ApiKey(
                workspace_id=workspace.id,
                name="bootstrap",
                prefix=prefix,
                key_hash=key_hash,
                scopes=["read", "write", "admin"],
            )
        )
        await session.commit()
        print("Workspace created.")
        print(f"  id:   {workspace.id}")
        print(f"  slug: {workspace.slug}")
        print(f"  key:  {key}")
        print("Store this key now; it is not recoverable.")
    await db.dispose()


async def init_qdrant() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
    )
    collections = {c.name for c in (await client.get_collections()).collections}
    if settings.qdrant_collection in collections:
        info = await client.get_collection(settings.qdrant_collection)
        dims = info.config.params.vectors.size  # type: ignore[union-attr]
        print(
            f"Collection {settings.qdrant_collection} exists (dim={dims}). "
            f"Configured dim={settings.embedding_dimension}."
        )
        if dims != settings.embedding_dimension:
            print("ERROR: dimension mismatch; use a new QDRANT_COLLECTION name.", file=sys.stderr)
            await client.close()
            raise SystemExit(2)
    else:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimension, distance=Distance.COSINE
            ),
        )
        for field, schema in (
            ("workspace_id", PayloadSchemaType.KEYWORD),
            ("knowledge_base_id", PayloadSchemaType.KEYWORD),
            ("document_id", PayloadSchemaType.KEYWORD),
            ("revision_id", PayloadSchemaType.KEYWORD),
            ("embedding_model", PayloadSchemaType.KEYWORD),
            ("active", PayloadSchemaType.BOOL),
        ):
            await client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=schema,
            )
        print(f"Collection {settings.qdrant_collection} created.")
    await client.close()


async def seed_demo() -> None:
    """Create a demo KB inside a bootstrap workspace (idempotent)."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    db = Database(settings.database_url)
    async with db.sessionmaker() as session:
        workspace = (
            await session.execute(select(Workspace).order_by(Workspace.created_at).limit(1))
        ).scalar_one_or_none()
        if workspace is None:
            print("No workspace found. Run create-workspace first.", file=sys.stderr)
            await db.dispose()
            raise SystemExit(1)
        kb = (
            await session.execute(
                select(KnowledgeBase).where(
                    KnowledgeBase.workspace_id == workspace.id, KnowledgeBase.name == "Demo"
                )
            )
        ).scalar_one_or_none()
        if kb is None:
            kb = KnowledgeBase(
                workspace_id=workspace.id,
                name="Demo",
                description="Knowledge base mẫu cho đánh giá và thử nghiệm.",
                embedding_model=settings.embedding_model,
                embedding_dimension=settings.embedding_dimension,
                chunk_target_tokens=settings.chunk_target_tokens,
            )
            session.add(kb)
            await session.commit()
            await session.refresh(kb)
        print(f"Demo KB id: {kb.id}")
    await db.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ws = sub.add_parser("create-workspace")
    p_ws.add_argument("--name", required=True)
    p_ws.add_argument("--slug")

    sub.add_parser("init-qdrant")
    sub.add_parser("seed-demo")

    args = parser.parse_args()
    if args.command == "create-workspace":
        asyncio.run(create_workspace(args.name, args.slug))
    elif args.command == "init-qdrant":
        asyncio.run(init_qdrant())
    elif args.command == "seed-demo":
        asyncio.run(seed_demo())


if __name__ == "__main__":
    main()
