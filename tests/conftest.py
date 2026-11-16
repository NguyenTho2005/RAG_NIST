"""Shared test fixtures.

Integration tests run against real PostgreSQL, Redis, and Qdrant from
docker-compose. A dedicated ``rag_test`` database is created and its public
schema rebuilt for every test session; the development database is untouched.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest

TEST_DB = "rag_test"
PG_ADMIN_URL = os.environ.get(
    "TEST_PG_ADMIN_URL", "postgresql://rag:rag_dev_password@localhost:5432/postgres"
)
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    f"postgresql+asyncpg://rag:rag_dev_password@localhost:5432/{TEST_DB}",
)
TEST_QDRANT_COLLECTION = f"rag_test_{uuid.uuid4().hex[:10]}"
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["ENVIRONMENT"] = "test"
os.environ["LLM_PROVIDER"] = "none"
os.environ["INLINE_WORKER"] = "true"
os.environ["LOG_FORMAT"] = "console"
os.environ["EMBEDDING_MODEL"] = "test-embedding"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["EMBEDDING_DIMENSION"] = "8"
os.environ["QDRANT_COLLECTION"] = TEST_QDRANT_COLLECTION
os.environ["CHUNK_TARGET_TOKENS"] = "128"
os.environ["CHUNK_MAX_TOKENS"] = "160"
os.environ["CHUNK_OVERLAP_TOKENS"] = "16"
os.environ["RERANK_ENABLED"] = "false"
os.environ["MIN_DENSE_SCORE"] = "-1.0"

from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.persistence.db import Base, Database  # noqa: E402
from app.persistence.models import ApiKey, KnowledgeBase, Workspace  # noqa: E402
from app.security.auth import generate_api_key  # noqa: E402


async def _ensure_database() -> None:
    for attempt in range(20):
        try:
            conn = await asyncpg.connect(PG_ADMIN_URL)
            break
        except Exception:
            if attempt == 19:
                raise
            await asyncio.sleep(0.5)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session", autouse=True)
def prepare_database() -> None:
    asyncio.run(_ensure_database())
    settings = Settings()
    db = Database(settings.database_url)
    assert settings.database_url.endswith(TEST_DB)

    async def _build() -> None:
        async with db.engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
        async with db.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await db.dispose()

    asyncio.run(_build())


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    get_settings.cache_clear()
    return Settings()


@pytest.fixture(scope="session", autouse=True)
def cleanup_vector_collection(test_settings: Settings) -> AsyncIterator[None]:
    """Remove the dedicated test collection so repeated runs do not leak vectors."""
    yield
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse

    client = QdrantClient(url=test_settings.qdrant_url)
    try:
        client.delete_collection(TEST_QDRANT_COLLECTION)
    except UnexpectedResponse:
        pass
    finally:
        client.close()


@pytest.fixture
async def db(test_settings: Settings) -> AsyncIterator[Database]:
    database = Database(test_settings.database_url)
    yield database
    await database.dispose()


@pytest.fixture(autouse=True)
async def clean_tables(db: Database) -> AsyncIterator[None]:
    table_names = [table.name for table in reversed(Base.metadata.sorted_tables)]
    async with db.engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {', '.join(table_names)} RESTART IDENTITY CASCADE"))
    yield


@pytest.fixture
async def workspace(db: Database) -> tuple[Workspace, str, ApiKey]:
    key, prefix, key_hash = generate_api_key()
    async with db.sessionmaker() as session:
        workspace = Workspace(name="Test Workspace", slug=f"test-{uuid.uuid4().hex[:8]}")
        session.add(workspace)
        await session.flush()
        api_key = ApiKey(
            workspace_id=workspace.id,
            name="test",
            prefix=prefix,
            key_hash=key_hash,
            scopes=["read", "write", "admin"],
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(workspace)
        await session.refresh(api_key)
    return workspace, key, api_key


@pytest.fixture
async def second_workspace(db: Database) -> tuple[Workspace, str]:
    key, prefix, key_hash = generate_api_key()
    async with db.sessionmaker() as session:
        workspace = Workspace(name="Other Workspace", slug=f"other-{uuid.uuid4().hex[:8]}")
        session.add(workspace)
        await session.flush()
        session.add(
            ApiKey(
                workspace_id=workspace.id,
                name="other",
                prefix=prefix,
                key_hash=key_hash,
                scopes=["read", "write", "admin"],
            )
        )
        await session.commit()
        await session.refresh(workspace)
    return workspace, key


@pytest.fixture
async def app(test_settings: Settings):
    application = create_app(test_settings)
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest.fixture
async def fake_runtime(app, test_settings: Settings) -> AsyncIterator[object]:
    """Replace the provider with a deterministic test double (clearly labelled)."""
    from app.embeddings.service import EmbeddingService
    from app.ingestion.runtime import IngestionRuntime
    from app.providers.fake import DeterministicFakeEmbeddingProvider

    embeddings = EmbeddingService(
        DeterministicFakeEmbeddingProvider(
            test_settings.embedding_dimension, model_id=test_settings.embedding_model
        ),
        cache_enabled=True,
        batch_size=3,
        max_attempts=2,
    )
    runtime = IngestionRuntime(
        settings=test_settings,
        database=app.state.database,
        qdrant=app.state.qdrant,
        embeddings=embeddings,
    )
    await runtime.vectors.ensure_collection()
    previous = app.state.ingestion
    await previous.shutdown()
    app.state.ingestion = runtime
    yield runtime


@pytest.fixture
def auth_headers(workspace: tuple[Workspace, str, ApiKey]) -> dict[str, str]:
    return {"Authorization": f"Bearer {workspace[1]}"}


@pytest.fixture
async def kb(
    db: Database, workspace: tuple[Workspace, str, ApiKey], test_settings: Settings
) -> KnowledgeBase:
    ws, _, _ = workspace
    async with db.sessionmaker() as session:
        knowledge_base = KnowledgeBase(
            workspace_id=ws.id,
            name="KB thử nghiệm",
            description="fixture",
            embedding_model=test_settings.embedding_model,
            embedding_dimension=test_settings.embedding_dimension,
            chunk_target_tokens=test_settings.chunk_target_tokens,
        )
        session.add(knowledge_base)
        await session.commit()
        await session.refresh(knowledge_base)
    return knowledge_base
