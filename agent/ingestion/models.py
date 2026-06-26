"""
ingestion/models.py

What problem does this solve?
- Every service (loader, chunker, embedder, vector store) needs a shared
  data contract. Without it, each service defines its own dict structure
  and the pipeline breaks when field names diverge.

Why does this file exist?
- Single source of truth for all data shapes in the ingestion pipeline.
- Changing a field here propagates to every service automatically.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Document:
    """
    What problem does this solve?
    - Represents a fully loaded document before any chunking or embedding.
    - Carries RBAC metadata so every downstream service (vector store,
      retrieval, API) can enforce access control without re-fetching.

    Why does this class exist?
    - Loaders produce it. ChunkingService consumes it.
    - Decouples format-specific loading from all downstream processing.

    Field rationale:
    - id:           New UUID on every load. Use file_hash for dedup, not id.
    - source_type:  Drives format-aware chunking strategies downstream
                    (e.g. markdown splits on ## headers, PDF on page breaks).
    - source_path:  Absolute path stored for audit trail and re-ingestion.
    - title:        Human-readable label for UI display and metadata filtering.
    - text:         Always plain UTF-8. All loaders normalise to this.
    - tenant_id:    Isolates data between enterprise clients in a shared cluster.
                    Every vector store query filters on this first.
    - owner_id:     Who uploaded this document. Required for row-level RBAC
                    and audit logging ("who ingested what, when").
    - access_roles: Which roles can read this document.
                    Empty list = no restrictions within the tenant.
                    Example: ["hr", "legal"] restricts to those roles only.
    - visibility:   Coarse-grained access level.
                    "public"     → anyone in the tenant can read.
                    "restricted" → only roles listed in access_roles can read.
                    "private"    → only owner_id can read.
    - file_hash:    MD5 of raw bytes. Used to skip re-ingestion of unchanged files.
                    Not a security hash — collision risk is acceptable here.
    - created_at:   UTC timestamp. Used for time-range filtering in vector queries.
    - metadata:     Loader-specific key/value pairs (page_count, file_size, etc).
                    Passed to chunk metadata so every chunk is self-describing.

    Why dataclass and not Pydantic?
    - Ingestion layer is internal — no HTTP boundary here.
    - Pydantic validation runs at the API layer (FastAPI schemas).
    - Dataclasses are lighter and have no external dependency.
    """

    id: str
    source_type: str
    source_path: str
    title: str
    text: str

    # --- Multi-tenancy ---
    tenant_id: str = "default"

    # --- RBAC ---
    owner_id: str = "system"
    access_roles: list[str] = field(default_factory=list)
    visibility: str = "public"          # "public" | "restricted" | "private"

    # --- Deduplication & audit ---
    file_hash: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Loader-specific metadata (page_count, file_size, etc.) ---
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """
    What problem does this solve?
    - LLMs have context limits. A Document must be split into smaller pieces
      that can be independently embedded and retrieved.
    - Each chunk needs enough metadata to trace back to its source document
      and enforce the same RBAC rules as the parent.

    Why does this class exist?
    - ChunkingService produces it. EmbeddingService consumes it.
    - EmbeddingService needs only text + identifiers — this is exactly that.

    Field rationale:
    - id:            Unique per chunk. Used as the vector store point ID.
    - document_id:   Links chunk back to parent Document for context assembly.
                     When a chunk is retrieved, the full document can be fetched.
    - text:          The actual text to be embedded. Sized by chunking strategy.
    - chunk_index:   Zero-based position. Preserved for ordered context windows
                     (show surrounding chunks to the LLM for better answers).
    - metadata:      Inherits parent Document metadata + chunking parameters.
                     Includes tenant_id, owner_id, access_roles, visibility
                     so the vector store can apply RBAC filters at query time
                     without a separate metadata DB lookup.

    Why not embed RBAC fields directly (like Document does)?
    - They live in metadata dict so the vector store payload schema stays flat
      and filterable without changing the DocumentChunk dataclass every time
      a new RBAC field is added.
    """

    id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
