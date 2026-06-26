"""
registry/models.py

What problem does this solve?
- After DocumentPipeline.run() completes, there is no persistent record of
  what happened: which files were ingested, how many chunks they produced,
  whether they failed and at which stage. Without a registry, operators
  cannot answer "has this file been ingested?" or "why did doc-X fail?"

Why a separate DocumentRecord model?
- Document (from ingestion) is an in-memory extraction result. DocumentRecord
  is the persistent audit trail written to PostgreSQL after every pipeline run.
  Mixing them would couple the loader layer to the registry schema.

Why status as a string enum instead of a bool?
- "completed", "failed_ingestion", "failed_preprocessing", "failed_chunking",
  "failed_embedding", "duplicate" — operators filter by status to find
  documents that need reprocessing at a specific stage. A bool cannot do that.
"""

from dataclasses import dataclass, field
from datetime import datetime


# Valid status values for DocumentRecord.status
STATUSES = {
    "pending",           # created, not yet processed
    "completed",         # all 4 stages succeeded
    "duplicate",         # near-duplicate of an existing document
    "failed_ingestion",
    "failed_preprocessing",
    "failed_chunking",
    "failed_embedding",
}


@dataclass
class DocumentRecord:
    """
    What problem does this solve?
    - Persistent representation of every pipeline run result. Stored in the
      `documents` PostgreSQL table. Enables: retry logic, audit trails,
      duplicate detection at the API layer, and operator dashboards.

    Why include file_hash?
    - Exact deduplication: if a file_hash already exists in the registry,
      the API can skip re-ingestion without running the full pipeline.
      Complements MinHash near-dedup (which catches paraphrased duplicates).

    Why include failed_stage?
    - Operators filter documents by failed_stage to find all docs that failed
      at embedding (e.g., GPU OOM) and resubmit just the embedding stage.

    Fields:
    - id:               document_id assigned at ingestion (UUID).
    - file_path:        Absolute path to the source file (for audit/retry).
    - file_hash:        SHA-256 of the source file (for exact dedup).
    - status:           One of STATUSES (see above).
    - tenant_id:        Tenant partition key.
    - owner_id:         Who ingested this document.
    - total_chunks:     Number of chunks written to PgVector (0 if not completed).
    - failed_stage:     Stage name where pipeline failed. None if completed.
    - error:            Error message from failed stage. None if completed.
    - is_duplicate:     True if MinHash near-dedup flagged this document.
    - duplicate_of:     document_id of the matching near-duplicate.
    - similarity_score: Jaccard similarity to the near-duplicate (0–1).
    - total_duration_ms: Wall-clock time for the full pipeline run.
    - created_at:       When this record was first inserted.
    - updated_at:       When this record was last updated.
    """

    id: str
    file_path: str
    status: str
    tenant_id: str
    owner_id: str
    file_hash: str = ""
    total_chunks: int = 0
    failed_stage: str | None = None
    error: str | None = None
    is_duplicate: bool = False
    duplicate_of: str | None = None
    similarity_score: float = 0.0
    total_duration_ms: float = 0.0
    created_at: datetime | None = None
    updated_at: datetime | None = None
