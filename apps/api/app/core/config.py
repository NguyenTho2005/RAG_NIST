"""Typed application configuration with fail-fast validation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- runtime -----------------------------------------------------------
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    api_prefix: str = "/api/v1"
    request_timeout_seconds: float = 120.0

    # --- infrastructure ----------------------------------------------------
    database_url: str = "postgresql+asyncpg://rag:rag_dev_password@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "rag_chunks_v1"

    # --- storage -----------------------------------------------------------
    data_dir: Path = Path("./data")
    evaluation_dir: Path = Path("./evaluation")
    max_upload_bytes: int = 25 * 1024 * 1024
    allowed_url_hosts: Annotated[list[str], NoDecode] = Field(default_factory=list)
    url_fetch_timeout_seconds: float = 15.0

    # --- embeddings --------------------------------------------------------
    embedding_provider: Literal["local", "openai", "ollama"] = "local"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dimension: int = 384
    embedding_batch_size: int = 16
    embedding_device: str = "cpu"
    embedding_model_revision: str | None = None
    embedding_cache_enabled: bool = True
    embedding_api_base: str = "https://api.openai.com/v1"
    embedding_api_key: SecretStr | None = None
    embedding_max_retries: int = 3
    embedding_timeout_seconds: float = 60.0
    ollama_base_url: str = "http://localhost:11434"

    # --- generation --------------------------------------------------------
    llm_provider: Literal["ollama", "openai", "none"] = "ollama"
    llm_model: str = "qwen2.5:1.5b-instruct"
    llm_api_base: str = "http://localhost:11434"
    llm_api_key: SecretStr | None = None
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 1024
    llm_timeout_seconds: float = 120.0
    llm_streaming: bool = True
    llm_max_retries: int = 2
    generation_max_rewrites: int = 1

    # --- reranking ---------------------------------------------------------
    reranker_provider: Literal["none", "local"] = "local"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_timeout_seconds: float = 15.0

    # --- retrieval ---------------------------------------------------------
    retrieval_mode: Literal["basic", "hybrid", "graph_local", "graph_global", "auto"] = "hybrid"
    dense_top_k: int = 30
    sparse_top_k: int = 30
    fusion_top_k: int = 20
    rerank_top_n: int = 8
    rrf_k: int = 60
    min_dense_score: float = 0.20
    context_token_budget: int = 3000
    chunk_target_tokens: int = 384
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 48
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    query_rewrite_enabled: bool = False
    rerank_enabled: bool = True
    context_expansion_enabled: bool = True

    # --- graph -------------------------------------------------------------
    graph_enabled: bool = False
    graph_max_chunks: int = 400
    graph_max_extraction_attempts: int = 2
    graph_min_confidence: float = 0.5

    # --- jobs / security ---------------------------------------------------
    max_concurrent_jobs: int = 2
    inline_worker: bool = True
    job_max_attempts: int = 3
    auth_rate_limit_per_minute: int = 120
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )
    bootstrap_api_key: SecretStr | None = None

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}")
        return upper

    @field_validator("cors_origins", "allowed_url_hosts", mode="before")
    @classmethod
    def _split_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _validate_consistency(self) -> Settings:
        if self.embedding_dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be positive")
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TARGET_TOKENS")
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("CHUNK_TARGET_TOKENS must be <= CHUNK_MAX_TOKENS")
        if self.rerank_top_n > self.fusion_top_k:
            raise ValueError("RERANK_TOP_N must be <= FUSION_TOP_K")
        if self.embedding_provider == "openai" and not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER=openai")
        if self.llm_provider == "openai" and not self.llm_api_key:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=openai")
        if not self.database_url.startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_URL must use the postgresql+asyncpg:// scheme")
        return self

    @property
    def llm_configured(self) -> bool:
        return self.llm_provider != "none"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
