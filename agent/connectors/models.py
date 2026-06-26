"""
connectors/models.py

What problem does this solve?
- Confluence, SharePoint, Jira, and Wiki all return content in different shapes
  (HTML pages, downloaded files, issue text). Without a shared model, SyncService
  would need to know each source's response format — tight coupling.
- ConnectorDoc is the normalised output every connector produces. SyncService
  only knows about ConnectorDoc; it never knows which source it came from.

Why content vs file_path as separate fields?
- Text-based sources (Confluence, Jira, Wiki) produce text/HTML inline.
  SyncService writes this to a temp file with the right extension.
- File-based sources (SharePoint DOCX/PDF downloads) produce a file already
  on disk. SyncService passes that path directly to PipelineService.
  Having both fields on one model covers both cases without subclassing.

Why source_url as identifier?
- It is the canonical, stable identity for a remote document.
  Used as the file_hash key in the registry so SyncService can detect
  "this Confluence page was already ingested and hasn't changed".
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConnectorDoc:
    """
    What problem does this solve?
    - Normalised document representation produced by every source connector.
      SyncService works exclusively with ConnectorDoc — never with source-
      specific API response objects.

    Fields:
    - source_url:    Canonical URL or ID for this document. Used as the
                     stable identifier for dedup and incremental sync.
    - title:         Document title (for Document.title in the pipeline).
    - source_name:   "confluence" | "sharepoint" | "jira" | "wiki" — stored
                     in chunk metadata so retrieval results cite their source.
    - content:       Text or HTML content. Non-empty for text-based sources.
                     Empty string when file_path is set (binary files).
    - content_type:  "html" | "text" | "markdown" — determines which temp
                     file extension SyncService writes (`.html`, `.txt`, `.md`).
    - file_path:     Pre-downloaded file path (SharePoint binary files).
                     None for text-based sources — SyncService creates the
                     temp file itself.
    - author:        Author/creator of the document (for audit metadata).
    - last_modified: When the document was last updated at the source.
                     Used by incremental sync: if unchanged since last ingest,
                     skip re-processing.
    - metadata:      Source-specific extras: space_key, project_key,
                     issue_type, site_url, etc. Stored in chunk metadata
                     for attribution in retrieval results.
    """

    source_url: str
    title: str
    source_name: str
    content: str = ""
    content_type: str = "html"
    file_path: str | None = None
    author: str = ""
    last_modified: datetime | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SyncResult:
    """
    What problem does this solve?
    - SyncService processes dozens to thousands of documents in one call.
      Without a result object, the caller (API endpoint or scheduled job)
      would have no way to know how many succeeded, how many were skipped,
      or which ones failed.

    Fields:
    - success:      True = sync completed (even if some individual docs failed).
                    False = sync itself crashed (auth failure, network error).
    - source:       Connector name ("confluence", "sharepoint", etc.).
    - fetched:      Total documents retrieved from the source.
    - ingested:     Successfully processed by the pipeline.
    - skipped:      Already up-to-date in the registry (no re-processing needed).
    - failed:       Documents that caused pipeline errors.
    - errors:       Per-document error messages for failed docs.
    - duration_ms:  Total wall-clock time for the sync run.
    - error:        Top-level error if success=False.
    """

    success: bool
    source: str
    fetched: int = 0
    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None
