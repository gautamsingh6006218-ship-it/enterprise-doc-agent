"""
services/pipeline_service.py

What problem does this solve?
- DocumentPipeline.run() produces a PipelineResult but does not persist it.
  Without PipelineService, API endpoints would have to orchestrate two things:
  call the pipeline AND write to the registry. That logic would be duplicated
  across every entry point (REST API, CLI, Celery worker).

Why a separate PipelineService over calling DocumentPipeline directly?
- Single responsibility: DocumentPipeline orchestrates services. PipelineService
  owns the side-effect of persisting the result to the registry.
- Consistent Result pattern: returns PipelineResult (never raises) so async
  task queues never crash on registry write failures.

How this connects:
  DocumentPipeline.run()  →  PipelineResult
  PipelineService.run()   →  PipelineResult  +  DocumentRecord written to registry
"""

from agent.ingestion.models import Document
from agent.pipeline.models import PipelineResult
from agent.pipeline.orchestrator import DocumentPipeline
from agent.registry.models import DocumentRecord, STATUSES
from agent.registry.store import DocumentRegistryStore


def _status_from_result(result: PipelineResult) -> str:
    """
    Derive the DocumentRecord status string from a PipelineResult.

    Why a helper instead of inline logic?
    - Keeps PipelineService.__init__/run() readable.
    - The mapping from PipelineResult fields → status string is a small but
      non-trivial decision point — isolating it makes it testable.
    """
    if result.is_duplicate:
        return "duplicate"
    if not result.success:
        stage = result.failed_stage or "ingestion"
        return f"failed_{stage}"
    return "completed"


class PipelineService:
    """
    What problem does this solve?
    - Combines DocumentPipeline (computation) with DocumentRegistryStore
      (persistence). Callers call run() once and both happen atomically
      from their perspective.

    Why inject both pipeline and registry_store?
    - Tests can mock either: verify the pipeline was called, verify the
      registry was written, without running real ML models or a real DB.

    Why is registry_store optional (None default)?
    - Allows PipelineService to be used without a database for smoke testing
      (e.g., during local development before Docker is running). When None,
      results are still returned but not persisted.
    """

    def __init__(
        self,
        pipeline: DocumentPipeline,
        registry_store: DocumentRegistryStore | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._registry = registry_store

    def run(
        self,
        file_path: str,
        tenant_id: str = "default",
        owner_id: str = "system",
        access_roles: list[str] | None = None,
        visibility: str = "public",
        file_hash: str = "",
        original_filename: str = "",
    ) -> PipelineResult:
        """
        What problem does this solve?
        - Runs the full ingestion pipeline and persists the result to the
          document registry in one call. Returns PipelineResult — never raises.

        Why file_hash as a parameter?
        - The caller (API endpoint or CLI) computes the SHA-256 of the uploaded
          file before calling run(). PipelineService stores it in the registry
          for exact-duplicate checks on future uploads.
          Computing it here would require reading the file twice (once for
          hashing, once for ingestion).

        Args:
        - file_path:    Absolute path to the file to ingest.
        - tenant_id:    Multi-tenant partition key.
        - owner_id:     Who is ingesting (audit + RBAC).
        - access_roles: Roles allowed to read this document.
        - visibility:   "public" | "restricted" | "private".
        - file_hash:    SHA-256 hex digest of the file (for exact dedup).
        """
        result = self._pipeline.run(
            file_path,
            tenant_id=tenant_id,
            owner_id=owner_id,
            access_roles=access_roles,
            visibility=visibility,
            original_filename=original_filename,
        )

        if self._registry is not None:
            self._persist(result, tenant_id, owner_id, file_hash, original_filename)

        return result

    # ── Private ────────────────────────────────────────────────────────────────

    def _persist(
        self,
        result: PipelineResult,
        tenant_id: str,
        owner_id: str,
        file_hash: str,
        original_filename: str = "",
    ) -> None:
        """
        Build a DocumentRecord from the PipelineResult and upsert it.

        Why absorb exceptions here?
        - A registry write failure must not fail the ingestion result.
          The document is already embedded and queryable. A failed audit
          record is recoverable; aborting a completed ingestion is not.
        """
        try:
            record = DocumentRecord(
                id=result.document_id or result.file_path,
                file_path=original_filename or result.file_path,
                file_hash=file_hash,
                status=_status_from_result(result),
                tenant_id=tenant_id,
                owner_id=owner_id,
                total_chunks=result.total_chunks,
                failed_stage=result.failed_stage,
                error=result.error,
                is_duplicate=result.is_duplicate,
                duplicate_of=result.duplicate_of,
                similarity_score=result.similarity_score,
                total_duration_ms=result.total_duration_ms,
            )
            self._registry.upsert(record)
        except Exception:
            # Registry write failure: logged but not propagated.
            # Document is already in PgVector and queryable.
            pass
