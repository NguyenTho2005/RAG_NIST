"""Domain enums shared by API, worker, and persistence layers."""

from __future__ import annotations

from enum import StrEnum


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REPROCESSING = "reprocessing"

    @classmethod
    def terminal(cls) -> set[DocumentStatus]:
        return {cls.READY, cls.FAILED, cls.CANCELLED}

    @classmethod
    def pipeline(cls) -> list[DocumentStatus]:
        return [
            cls.VALIDATING,
            cls.PARSING,
            cls.CHUNKING,
            cls.EMBEDDING,
            cls.INDEXING,
        ]


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStage(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    PUBLISHING = "publishing"
    GRAPH_EXTRACTION = "graph_extraction"
    DONE = "done"


class RevisionStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"


class JobType(StrEnum):
    INGEST = "ingest"
    REPROCESS = "reprocess"
    DELETE = "delete"
    GRAPH_BUILD = "graph_build"
    EVALUATION = "evaluation"


class RetrievalMode(StrEnum):
    BASIC = "basic"
    HYBRID = "hybrid"
    GRAPH_LOCAL = "graph_local"
    GRAPH_GLOBAL = "graph_global"
    AUTO = "auto"


class ChunkKind(StrEnum):
    CHILD = "child"
    PARENT = "parent"


class SourceType(StrEnum):
    UPLOAD = "upload"
    URL = "url"


class EntityType(StrEnum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    REGULATION = "regulation"
    POLICY = "policy"
    PROCESS = "process"
    ROLE = "role"
    SYSTEM = "system"
    DATE = "date"
    OTHER = "other"
