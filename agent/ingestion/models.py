"""
ingestion/models.py
-------------------
Core data models for the document ingestion pipeline.

These dataclasses act as the canonical data contracts shared across all
ingestion components (loaders, chunker, embedding service, vector store).
Any new field added here is immediately available to every downstream service.

Design decisions:
- Plain dataclasses (not Pydantic) to keep ingestion layer dependency-free.
  Pydantic validation is applied at the API boundary (FastAPI layer), not here.
- `file_hash` uses MD5 — fast and sufficient for deduplication; not used for
  security, so collision risk is acceptable.
- `tenant_id` enables multi-tenant isolation at the data model level so every
  downstream store (vector DB, metadata DB) can filter by tenant without extra
  join tables.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Document:
    """
    Represents a fully loaded, raw document before chunking.

    Produced by a DocumentLoader and consumed by the ChunkingService.

    Attributes:
        id:           Unique document identifier (UUID4). Generated fresh on
                      each load — use file_hash for deduplication checks.
        source_type:  Origin format: 'pdf', 'docx', 'txt', 'markdown', 'html'.
                      Used downstream to apply format-specific chunking strategies.
        source_path:  Absolute path to the source file on disk (or remote URI
                      for future cloud connectors).
        title:        Human-readable title. Defaults to the file stem; can be
                      overridden by loaders that parse a title from content
                      (e.g. HtmlLoader reads the <title> tag).
        text:         Full extracted plain text of the document. All loaders
                      normalise to plain UTF-8 text before returning.
        tenant_id:    Identifies which tenant this document belongs to.
                      Defaults to "default" for single-tenant deployments.
        file_hash:    MD5 hex digest of the raw file bytes. Used to detect
                      duplicate ingestion without re-processing the file.
        created_at:   UTC timestamp of when this Document object was created.
                      Stored in vector DB metadata for time-based filtering.
        metadata:     Loader-specific key/value pairs (page_count, file_size,
                      etc.). Passed through to chunk metadata for traceability.
    """

    id: str
    source_type: str
    source_path: str
    title: str
    text: str
    tenant_id: str = "default"
    file_hash: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """
    Represents a single chunk of a Document after text splitting.

    Produced by the ChunkingService and consumed by the EmbeddingService.
    Each chunk is independently embedded and stored in the vector database.

    Attributes:
        id:            Unique chunk identifier (UUID4).
        document_id:   References the parent Document.id — used to reconstruct
                       full document context when a chunk is retrieved.
        text:          The actual text content of this chunk.
        chunk_index:   Zero-based position of this chunk within the document.
                       Used to maintain reading order during context assembly.
        metadata:      Inherits all parent Document metadata plus chunking
                       parameters (chunk_size, chunk_overlap). This makes every
                       chunk self-describing when retrieved from the vector store.
    """

    id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
