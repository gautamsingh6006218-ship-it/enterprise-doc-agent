"""
documents/service.py

What problem does this solve?
- A document lives in two tables owned by two modules: its pipeline record in
  `documents` (registry) and its chunks in `document_chunks` (embeddings).
  Removing or replacing a document must clear both. That two-step delete was
  copied into the ingest route, the documents route, and the file watcher,
  each reaching into PgVectorStore directly.

Why a separate module instead of a delete on RegistryService?
- The registry module must not depend on embeddings. DocumentService is the
  one place that coordinates both stores.

Why does ingest() live here and not in the upload routes?
- Both upload routes and the file watcher hashed the bytes, checked for an
  existing copy, removed a stale version, and ran the pipeline on a temp
  file. That is RAG logic, so the backend now only hands over the bytes.
"""

import hashlib
import logging
import os
import tempfile

from agent.embeddings import PgVectorStore
from agent.pipeline import PipelineResult, PipelineService
from agent.registry import DocumentRecord, RegistryResult, RegistryService

logger = logging.getLogger(__name__)


class DocumentService:
    """
    What problem does this solve?
    - Callers add, replace or delete a document with one call and never touch
      the chunk store, the registry store or the pipeline directly.

    Why inject registry_service, chunk_store and pipeline_service?
    - Tests pass mocks and assert the order of calls without a database.
    """

    def __init__(
        self,
        registry_service: RegistryService,
        chunk_store: PgVectorStore,
        pipeline_service: PipelineService,
    ) -> None:
        self._registry = registry_service
        self._chunks = chunk_store
        self._pipeline = pipeline_service

    def ingest(
        self,
        file_bytes: bytes,
        original_filename: str,
        tenant_id: str,
        owner_id: str,
        access_roles: list[str] | None = None,
        visibility: str = "public",
        replace: bool = False,
    ) -> PipelineResult:
        """
        Index a file under its original filename.

        - Same content already indexed for this tenant → a duplicate result for
          that record, and the pipeline does not run. replace=True skips this.
        - Same filename with new content → the old version is deleted first.
        - Otherwise → the pipeline runs on a temp copy of the bytes.
        """
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if not replace:
            match = self._registry.get_by_file_hash(file_hash, tenant_id)
            if match.success and match.record is not None:
                return _duplicate_of(match.record)

        unchanged = self.remove_stale_version(original_filename, file_hash, tenant_id)
        if unchanged is not None and not replace:
            return _duplicate_of(unchanged)

        suffix = os.path.splitext(original_filename)[1] or ".bin"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="ingest_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(file_bytes)

            return self._pipeline.run(
                file_path=tmp_path,
                tenant_id=tenant_id,
                owner_id=owner_id,
                access_roles=access_roles,
                visibility=visibility,
                file_hash=file_hash,
                original_filename=original_filename,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def delete(self, document_id: str) -> RegistryResult:
        """
        Delete a document's chunks, then its registry record.

        Why chunks first?
        - If the record went first and the chunk delete then failed, the
          chunks would stay searchable with no record left to delete them by.
        """
        self._chunks.delete_by_document_id(document_id)
        return self._registry.delete(document_id)

    def remove_stale_version(
        self,
        original_filename: str,
        file_hash: str,
        tenant_id: str,
    ) -> DocumentRecord | None:
        """
        Prepare to ingest a file whose name may already be indexed.

        Returns:
        - The existing record when a document with this filename has the same
          content hash (nothing changed). Nothing is deleted.
        - None after deleting the old version when the content changed.
        - None when no document has this filename or the lookup fails.
        """
        match = self._registry.get_by_original_filename(original_filename, tenant_id)
        if not match.success or match.record is None:
            return None

        existing = match.record
        if existing.file_hash == file_hash:
            return existing

        logger.info(
            "Replacing document %s: '%s' content changed", existing.id, original_filename
        )
        self.delete(existing.id)
        return None


# ── Private ────────────────────────────────────────────────────────────────────

def _duplicate_of(record: DocumentRecord) -> PipelineResult:
    """Result for content that is already indexed as `record`."""
    return PipelineResult(
        success=True,
        file_path=record.file_path,
        document_id=record.id,
        total_chunks=record.total_chunks,
        is_duplicate=True,
        duplicate_of=record.id,
        similarity_score=1.0,
    )
