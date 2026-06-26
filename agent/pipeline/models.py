"""
pipeline/models.py

What problem does this solve?
- The orchestrator runs 4 independent services sequentially. Without shared
  result models, each stage's output would be a raw dict or tuple — no
  type safety, no consistent shape for callers to inspect.

Why StageResult + PipelineResult instead of one flat result?
- StageResult captures per-stage success, duration, and stats — needed for
  monitoring (which stage is the bottleneck?) and retry logic (which stage
  failed so we can resume from there?).
- PipelineResult is the aggregate view callers care about: did it succeed,
  how many chunks were stored, was it a duplicate?
"""

from dataclasses import dataclass, field


@dataclass
class StageResult:
    """
    What problem does this solve?
    - Per-stage observability: operators see exactly where a document failed
      and how long each stage took, without reading logs.

    Fields:
    - stage:       One of: "ingestion", "preprocessing", "chunking", "embedding".
    - success:     True = stage completed without error.
    - duration_ms: Wall-clock time for this stage in milliseconds.
    - error:       Failure reason from the service Result. None on success.
    - stats:       Stage-specific metrics (chunk count, avg tokens, etc.).
    """

    stage: str
    success: bool
    duration_ms: float
    error: str | None = None
    stats: dict | None = None


@dataclass
class PipelineResult:
    """
    What problem does this solve?
    - Single return value from DocumentPipeline.run() that callers can
      inspect without understanding the internals of all 4 services.

    Why include both success and failed_stage?
    - success=False tells the caller something went wrong.
    - failed_stage tells the DocumentRegistry which status to record
      ("failed_chunking" vs "failed_embedding") so operators can filter
      by failure type and re-run only the affected stage.

    Why is_duplicate a separate flag instead of success=False?
    - A duplicate is not a failure — the document exists and is already
      indexed. Callers should treat it as a valid outcome, not an error.
      is_duplicate=True + success=True means "no new work needed".

    Fields:
    - success:          True = all stages completed and chunks embedded.
    - file_path:        Source file — for audit trail and retry.
    - document_id:      Document.id assigned at ingestion. None if ingestion failed.
    - is_duplicate:     True if MinHash LSH detected a near-duplicate.
    - duplicate_of:     document_id of the existing near-duplicate.
    - similarity_score: Jaccard similarity to the duplicate (0–1).
    - total_chunks:     Number of chunks written to PgVector.
    - total_duration_ms: Wall-clock time for the full pipeline.
    - failed_stage:     Which stage failed. None on success.
    - error:            Human-readable error from the failed stage.
    - stages:           Per-stage StageResult list — for detailed monitoring.
    """

    success: bool
    file_path: str
    document_id: str | None = None
    is_duplicate: bool = False
    duplicate_of: str | None = None
    similarity_score: float = 0.0
    total_chunks: int = 0
    total_duration_ms: float = 0.0
    failed_stage: str | None = None
    error: str | None = None
    stages: list[StageResult] = field(default_factory=list)
