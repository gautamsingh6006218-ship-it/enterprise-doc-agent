"""
registry/store.py

What problem does this solve?
- PipelineService needs to persist a DocumentRecord after every pipeline run.
- RegistryService needs to query records (by ID, tenant, status) for the API layer.
- Without this layer, every caller would write raw SQL — no type safety, no
  central place to update the schema mapping.

Why psycopg2 directly instead of an ORM (SQLAlchemy)?
- Same reason as PgVectorStore: the queries are simple CRUD with one table.
  An ORM adds a dependency and a learning curve for minimal benefit here.
  When the registry grows to need complex joins, migrating is straightforward.

Why accept connection as a constructor arg?
- Tests inject a mock connection — no PostgreSQL instance required.
- Production wires a real psycopg2 connection (or connection pool) at startup.

Why upsert instead of insert-only?
- Pipeline retries: if a document is reingested after a failure, the existing
  record should be updated (status promoted from failed_* to completed) rather
  than creating a duplicate row.
"""

from datetime import datetime, timezone
from typing import Any

from agent.registry.models import DocumentRecord


class DocumentRegistryStore:
    """
    What problem does this solve?
    - All SQL for the `documents` table lives here. Callers use
      upsert/get/list/delete — no raw SQL at call sites.

    Why is connection required (not optional)?
    - There is no valid default connection. Requiring it at construction
      forces callers to decide explicitly where data is stored.
      A None default would silently produce NullPointerErrors at runtime.
    """

    def __init__(self, connection: Any) -> None:
        self._conn = connection

    def upsert(self, record: DocumentRecord) -> None:
        """
        What problem does this solve?
        - Writes or updates a DocumentRecord after every pipeline run.

        Why ON CONFLICT DO UPDATE?
        - If the same document_id is resubmitted (retry after failure),
          update status, error, total_chunks, and timing — don't duplicate.

        Args:
        - record: DocumentRecord with all fields populated.
        """
        sql = """
            INSERT INTO documents (
                id, file_path, file_hash, status,
                tenant_id, owner_id,
                total_chunks, failed_stage, error,
                is_duplicate, duplicate_of, similarity_score,
                total_duration_ms, created_at, updated_at
            ) VALUES (
                %(id)s, %(file_path)s, %(file_hash)s, %(status)s,
                %(tenant_id)s, %(owner_id)s,
                %(total_chunks)s, %(failed_stage)s, %(error)s,
                %(is_duplicate)s, %(duplicate_of)s, %(similarity_score)s,
                %(total_duration_ms)s, %(created_at)s, %(updated_at)s
            )
            ON CONFLICT (id) DO UPDATE SET
                status            = EXCLUDED.status,
                total_chunks      = EXCLUDED.total_chunks,
                failed_stage      = EXCLUDED.failed_stage,
                error             = EXCLUDED.error,
                is_duplicate      = EXCLUDED.is_duplicate,
                duplicate_of      = EXCLUDED.duplicate_of,
                similarity_score  = EXCLUDED.similarity_score,
                total_duration_ms = EXCLUDED.total_duration_ms,
                updated_at        = EXCLUDED.updated_at
        """
        now = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            cur.execute(sql, {
                "id":                record.id,
                "file_path":         record.file_path,
                "file_hash":         record.file_hash,
                "status":            record.status,
                "tenant_id":         record.tenant_id,
                "owner_id":          record.owner_id,
                "total_chunks":      record.total_chunks,
                "failed_stage":      record.failed_stage,
                "error":             record.error,
                "is_duplicate":      record.is_duplicate,
                "duplicate_of":      record.duplicate_of,
                "similarity_score":  record.similarity_score,
                "total_duration_ms": record.total_duration_ms,
                "created_at":        record.created_at or now,
                "updated_at":        now,
            })
        self._conn.commit()

    def get(self, document_id: str) -> DocumentRecord | None:
        """
        What problem does this solve?
        - API GET /documents/{id} needs to fetch a single record.
        - PipelineService checks for existing records before re-running (exact dedup).

        Returns None if no record exists for document_id.
        """
        sql = "SELECT * FROM documents WHERE id = %s"
        with self._conn.cursor() as cur:
            cur.execute(sql, (document_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row, cur.description)

    def get_by_file_hash(self, file_hash: str, tenant_id: str) -> DocumentRecord | None:
        """
        What problem does this solve?
        - Exact-duplicate check at the API layer: if a file with this SHA-256
          hash was already ingested by this tenant, skip the pipeline entirely.
        - Faster than MinHash LSH for identical files (byte-for-byte equality).

        Args:
        - file_hash: SHA-256 hex digest of the source file.
        - tenant_id: Only check within this tenant's records.
        """
        sql = """
            SELECT * FROM documents
            WHERE file_hash = %s AND tenant_id = %s AND status = 'completed'
            LIMIT 1
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (file_hash, tenant_id))
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row, cur.description)

    def list_by_tenant(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        """
        What problem does this solve?
        - API GET /documents lists all documents for a tenant, with optional
          status filter (e.g., "failed_embedding") for operator dashboards.

        Args:
        - tenant_id: Required — RBAC partition, never list across tenants.
        - status:    Optional filter. None = all statuses.
        - limit:     Page size (default 100, max enforced by caller).
        - offset:    Pagination offset.
        """
        if status is not None:
            sql = """
                SELECT * FROM documents
                WHERE tenant_id = %s AND status = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            params = (tenant_id, status, limit, offset)
        else:
            sql = """
                SELECT * FROM documents
                WHERE tenant_id = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            params = (tenant_id, limit, offset)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            description = cur.description

        return [self._row_to_record(row, description) for row in rows]

    def delete(self, document_id: str) -> bool:
        """
        What problem does this solve?
        - API DELETE /documents/{id} removes the registry record.
          (Chunk deletion from PgVector is handled separately by EmbeddingService.)

        Returns True if a row was deleted, False if it did not exist.
        """
        sql = "DELETE FROM documents WHERE id = %s"
        with self._conn.cursor() as cur:
            cur.execute(sql, (document_id,))
            deleted = cur.rowcount > 0
        self._conn.commit()
        return deleted

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_record(row: tuple, description: Any) -> DocumentRecord:
        """Map a psycopg2 cursor row to a DocumentRecord."""
        col_names = [d.name for d in description]
        data = dict(zip(col_names, row))
        return DocumentRecord(
            id=data["id"],
            file_path=data["file_path"],
            file_hash=data.get("file_hash", ""),
            status=data["status"],
            tenant_id=data["tenant_id"],
            owner_id=data["owner_id"],
            total_chunks=data.get("total_chunks", 0),
            failed_stage=data.get("failed_stage"),
            error=data.get("error"),
            is_duplicate=data.get("is_duplicate", False),
            duplicate_of=data.get("duplicate_of"),
            similarity_score=data.get("similarity_score", 0.0),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
