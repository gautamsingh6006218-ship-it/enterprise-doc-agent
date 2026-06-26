"""
api/models.py

What problem does this solve?
- Route handlers need typed request bodies and response shapes that are
  serialisable to JSON. Using raw dicts would lose validation and documentation.
- Pydantic models here are the API contract — they define exactly what the
  caller sends and what they receive back, independent of internal dataclasses.

Why separate API models from internal dataclasses?
- Internal dataclasses (PipelineResult, DocumentRecord, RetrievedContext) can
  change without breaking the API contract. The API models are the stable surface.
- Pydantic validates input and serialises output — internal dataclasses don't.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Ingest ─────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    success: bool
    document_id: str | None = None
    total_chunks: int = 0
    is_duplicate: bool = False
    duplicate_of: str | None = None
    similarity_score: float = 0.0
    total_duration_ms: float = 0.0
    failed_stage: str | None = None
    error: str | None = None


# ── Query ──────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = Field(default=False, description="Generate an LLM answer from retrieved chunks")


class ChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    query: str
    chunks: list[ChunkResult]
    window_texts: list[str] = []
    retrieval_stats: dict[str, Any] = {}
    answer: str | None = None
    answer_model: str | None = None


# ── Documents ──────────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    file_path: str
    status: str
    tenant_id: str
    owner_id: str
    total_chunks: int = 0
    failed_stage: str | None = None
    error: str | None = None
    is_duplicate: bool = False
    duplicate_of: str | None = None
    similarity_score: float = 0.0
    total_duration_ms: float = 0.0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class DeleteResponse(BaseModel):
    deleted: bool
    document_id: str


# ── Sync ───────────────────────────────────────────────────────────────────────

class SyncRequest(BaseModel):
    since: datetime | None = None
    space_key: str | None = None      # Confluence
    project_key: str | None = None    # Jira
    library_name: str = "Documents"   # SharePoint
    namespace: int = 0                # Wiki
    include_closed: bool = False      # Jira


class SyncResponse(BaseModel):
    success: bool
    source: str
    fetched: int = 0
    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = []
    duration_ms: float = 0.0
    error: str | None = None


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
