"""
services/registry_service.py

What problem does this solve?
- API endpoints (GET /documents, GET /documents/{id}, DELETE /documents/{id})
  need access to DocumentRecord data but should not call DocumentRegistryStore
  directly — that would scatter raw store calls across the API layer.

Why a service over the store directly?
- Validation: list_by_tenant enforces a max limit so API consumers cannot
  request unbounded result sets.
- Result pattern: returns RegistryResult (never raises) consistent with every
  other service in the pipeline.
- When registry becomes its own microservice, only this file's interface matters.
"""

from dataclasses import dataclass

from agent.registry.models import DocumentRecord
from agent.registry.store import DocumentRegistryStore


@dataclass
class RegistryResult:
    """
    What problem does this solve?
    - Uniform result type across all registry operations.
      Callers always check success — no try/except at API layer.

    Fields:
    - success:  True = operation completed without error.
    - record:   Single DocumentRecord (get operations). None otherwise.
    - records:  List of DocumentRecords (list operations). Empty if none.
    - deleted:  True if delete found and removed the record.
    - error:    Failure reason. None on success.
    """

    success: bool
    record: DocumentRecord | None = None
    records: list[DocumentRecord] = None
    deleted: bool = False
    error: str | None = None

    def __post_init__(self):
        if self.records is None:
            self.records = []


_MAX_LIST_LIMIT = 500


class RegistryService:
    """
    What problem does this solve?
    - Single entry point for all DocumentRecord queries. API routes call
      get/list/delete here — not the store directly.

    Why require store at construction?
    - No valid default exists. An unconfigured registry would silently
      return empty results, masking bugs. Fail fast at construction.
    """

    def __init__(self, store: DocumentRegistryStore) -> None:
        self._store = store

    def get(self, document_id: str) -> RegistryResult:
        """
        What problem does this solve?
        - API GET /documents/{id}: fetch a single document's pipeline status.

        Returns RegistryResult(success=True, record=None) if not found
        (not an error — callers check record is not None separately).
        """
        try:
            record = self._store.get(document_id)
            return RegistryResult(success=True, record=record)
        except Exception as e:
            return RegistryResult(success=False, error=f"Registry lookup failed: {e}")

    def get_by_file_hash(self, file_hash: str, tenant_id: str) -> RegistryResult:
        """
        What problem does this solve?
        - Exact-duplicate check at the API layer before running the pipeline.
          If a file was already ingested (same SHA-256, same tenant),
          return the existing record without running ingestion again.

        Args:
        - file_hash: SHA-256 hex digest of the uploaded file.
        - tenant_id: Only check within this tenant's records.
        """
        try:
            record = self._store.get_by_file_hash(file_hash, tenant_id)
            return RegistryResult(success=True, record=record)
        except Exception as e:
            return RegistryResult(success=False, error=f"Hash lookup failed: {e}")

    def list_by_tenant(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> RegistryResult:
        """
        What problem does this solve?
        - API GET /documents: list all documents for a tenant, optionally
          filtered by status for operator dashboards.

        Why cap limit at _MAX_LIST_LIMIT?
        - Prevents a single query from returning 500K rows and OOMing the API.
          Callers paginate using offset.

        Args:
        - tenant_id: Required — never list across tenants.
        - status:    Optional filter (e.g., "failed_embedding").
        - limit:     Page size. Capped at 500 regardless of caller value.
        - offset:    Pagination offset.
        """
        try:
            capped = min(limit, _MAX_LIST_LIMIT)
            records = self._store.list_by_tenant(
                tenant_id, status=status, limit=capped, offset=offset
            )
            return RegistryResult(success=True, records=records)
        except Exception as e:
            return RegistryResult(success=False, error=f"Registry list failed: {e}")

    def delete(self, document_id: str) -> RegistryResult:
        """
        What problem does this solve?
        - API DELETE /documents/{id}: remove the registry record for a document.
          Note: chunk deletion from PgVector must be handled separately
          (call EmbeddingService.delete_document(document_id) first).

        Returns RegistryResult(success=True, deleted=True) if found and removed,
        RegistryResult(success=True, deleted=False) if the ID did not exist.
        """
        try:
            deleted = self._store.delete(document_id)
            return RegistryResult(success=True, deleted=deleted)
        except Exception as e:
            return RegistryResult(success=False, error=f"Registry delete failed: {e}")
