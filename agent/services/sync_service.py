"""
services/sync_service.py

What problem does this solve?
- Connectors know how to fetch documents from remote sources.
- PipelineService knows how to ingest a file path into PgVector.
- Neither knows about the other. SyncService bridges them:
  fetch ConnectorDocs → write to temp files → call PipelineService for each.

Why SyncService instead of calling PipelineService directly in each connector?
- Connectors should not know about the pipeline. Connector = data fetching.
  Pipeline = processing. Mixing them would make connectors untestable in
  isolation and couple them to the pipeline's internal API.
- SyncService owns the cross-cutting concerns: temp file lifecycle,
  incremental sync (skip unchanged docs), per-doc error isolation,
  and SyncResult stats aggregation.

Why write ConnectorDoc.content to a temp file?
- PipelineService.run() takes a file path. It never accepts raw text.
  The existing LoaderRegistry routes by file extension — writing HTML to
  a .html temp file means HtmlLoader handles it automatically.
  No new loader code needed.

Why is_up_to_date check before pipeline?
- Re-ingesting an unchanged Confluence page is expensive: chunking + BGE-M3
  encoding + PgVector upsert for content that hasn't changed. The registry
  stores file_hash (SHA-256 of content). If the hash matches, skip.

Why per-doc try/except instead of stopping on first failure?
- A single corrupted Jira issue should not abort the sync of 10,000 issues.
  Each document is isolated. SyncResult.errors captures failures for retry.
"""

import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path

from agent.connectors.base import BaseConnector
from agent.connectors.models import ConnectorDoc, SyncResult
from agent.services.pipeline_service import PipelineService
from agent.services.registry_service import RegistryService

logger = logging.getLogger(__name__)

# Map ConnectorDoc.content_type → temp file extension
_CONTENT_TYPE_EXT = {
    "html":     ".html",
    "text":     ".txt",
    "markdown": ".md",
}


class SyncService:
    """
    What problem does this solve?
    - Single entry point for syncing any enterprise source into PgVector.
      Callers call sync(connector, tenant_id, ...) — they never manage
      temp files, dedup checks, or per-document error handling.

    Why require both pipeline_service and registry_service?
    - pipeline_service: ingests each document (writes chunks to PgVector).
    - registry_service: checks whether a document is already up-to-date
      (get_by_file_hash) so we skip unchanged docs.
    - Both are required — no valid default exists.
    """

    def __init__(
        self,
        pipeline_service: PipelineService,
        registry_service: RegistryService,
    ) -> None:
        self._pipeline = pipeline_service
        self._registry = registry_service

    def sync(
        self,
        connector: BaseConnector,
        tenant_id: str,
        owner_id: str = "sync",
        access_roles: list[str] | None = None,
        visibility: str = "public",
        since=None,
        **connector_kwargs,
    ) -> SyncResult:
        """
        What problem does this solve?
        - Fetches all (or incrementally updated) documents from a connector
          and ingests each one through PipelineService.

        Args:
        - connector:         Any BaseConnector subclass (Confluence, SharePoint, etc.)
        - tenant_id:         Multi-tenant partition for all ingested chunks/records.
        - owner_id:          Audit owner (default "sync" for scheduled jobs).
        - access_roles:      RBAC roles applied to all ingested documents.
        - visibility:        "public" | "restricted" | "private".
        - since:             datetime — only sync documents modified after this.
                             None = full sync.
        - **connector_kwargs: Source-specific filters forwarded to
                              connector.fetch_documents() (space_key, project_key, etc.)

        Returns SyncResult with per-source stats.
        """
        start = time.perf_counter()
        result = SyncResult(success=True, source=connector.source_name)

        # ── Fetch ──────────────────────────────────────────────────────────────
        try:
            docs = connector.fetch_documents(since=since, **connector_kwargs)
        except Exception as e:
            return SyncResult(
                success=False,
                source=connector.source_name,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
                error=f"Connector fetch failed: {e}",
            )

        result.fetched = len(docs)

        # ── Ingest each doc ────────────────────────────────────────────────────
        for doc in docs:
            try:
                self._process_doc(
                    doc, tenant_id, owner_id, access_roles, visibility, result
                )
            except Exception as e:
                result.failed += 1
                result.errors.append(f"{doc.source_url}: {e}")
                logger.warning("Sync doc failed %s: %s", doc.source_url, e)

        result.duration_ms = round((time.perf_counter() - start) * 1000, 1)
        return result

    # ── Private ────────────────────────────────────────────────────────────────

    def _process_doc(
        self,
        doc: ConnectorDoc,
        tenant_id: str,
        owner_id: str,
        access_roles: list[str] | None,
        visibility: str,
        result: SyncResult,
    ) -> None:
        content_hash = _content_hash(doc)

        # Skip if already ingested and unchanged
        existing = self._registry.get_by_file_hash(content_hash, tenant_id)
        if existing.success and existing.record is not None:
            result.skipped += 1
            return

        tmp_path = _write_temp_file(doc)
        try:
            extra_meta = {
                "source_name": doc.source_name,
                "source_url": doc.source_url,
                **doc.metadata,
            }
            pipeline_result = self._pipeline.run(
                file_path=tmp_path,
                tenant_id=tenant_id,
                owner_id=owner_id,
                access_roles=access_roles,
                visibility=visibility,
                file_hash=content_hash,
            )
            if pipeline_result.success:
                result.ingested += 1
            else:
                result.failed += 1
                result.errors.append(
                    f"{doc.source_url}: {pipeline_result.error}"
                )
        finally:
            # Always clean up temp files — pipeline has already read and
            # processed the content; keeping it on disk is a security risk.
            _cleanup_temp(tmp_path)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _content_hash(doc: ConnectorDoc) -> str:
    """
    SHA-256 of the document content or file bytes.

    Why hash file bytes for file_path docs?
    - SharePoint files (DOCX, PDF) are already on disk in doc.file_path.
      Hashing doc.content (which is "") would produce the same hash for
      every binary file — defeating exact dedup.
    """
    if doc.file_path and os.path.exists(doc.file_path):
        h = hashlib.sha256()
        with open(doc.file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    return hashlib.sha256(doc.content.encode("utf-8", errors="replace")).hexdigest()


def _write_temp_file(doc: ConnectorDoc) -> str:
    """
    Returns a file path ready for PipelineService.run().

    - Binary docs (SharePoint downloads): file_path already set, return as-is.
    - Text/HTML docs: write content to a temp file with the right extension.
    """
    if doc.file_path:
        return doc.file_path

    ext = _CONTENT_TYPE_EXT.get(doc.content_type, ".txt")
    fd, path = tempfile.mkstemp(suffix=ext, prefix="sync_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(doc.content)
    except Exception:
        os.close(fd)
        raise
    return path


def _cleanup_temp(path: str) -> None:
    """Delete temp file if it exists. Silent on error."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception:
        pass
