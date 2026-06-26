"""
connectors/sharepoint.py

What problem does this solve?
- SharePoint is the primary document store for most Microsoft-stack enterprises:
  Word docs, PowerPoint decks, Excel sheets, PDFs — all live in SharePoint
  document libraries. Without a connector, operators must manually download
  and re-upload files, making incremental sync impossible.

Why O365 (python-o365) over Microsoft Graph SDK directly?
- O365 wraps the Microsoft Graph API with a Pythonic interface and handles
  OAuth token refresh automatically. The raw Graph SDK requires manual token
  management and more boilerplate.
- O365's SharePoint interface gives direct access to DriveItem objects with
  last_modified timestamps, download URLs, and file extensions.

Authentication (Azure AD app registration required):
1. Register an app in Azure Portal → App registrations
2. Grant: Sites.Read.All (application permission)
3. Create a client secret
4. Pass client_id, client_secret, tenant_id here

Why download files to disk instead of loading content in memory?
- The existing LoaderRegistry already handles every file format (PDF, DOCX,
  XLSX, PPTX, etc.) by reading from a file path. Loading content into memory
  and writing it back would duplicate effort.
- Downloading to a named temp file with the correct extension lets
  LoaderRegistry auto-detect format via MIME + extension — zero new code.

Why filter by extension?
- SharePoint libraries contain non-document files (.aspx pages, .json config
  files, thumbnails) that should not be ingested. The extension allowlist
  ensures only meaningful document types enter the pipeline.
"""

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from agent.connectors.base import BaseConnector
from agent.connectors.models import ConnectorDoc

logger = logging.getLogger(__name__)

try:
    from O365 import Account
    from O365.sharepoint import SharepointSite
    _O365_AVAILABLE = True
except ImportError:
    _O365_AVAILABLE = False

# File extensions worth ingesting from SharePoint
_SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".csv", ".txt", ".md",
    ".html", ".htm",
}


class SharePointConnector(BaseConnector):
    """
    What problem does this solve?
    - Downloads files from a SharePoint document library to temp files,
      then returns ConnectorDoc objects with file_path set so SyncService
      can pass them directly to PipelineService without any format conversion.

    Why store site_url in metadata?
    - Retrieval results should cite the SharePoint site and library so users
      can navigate back to the source document. Without this, retrieved
      chunks are untraceable to their origin.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        site_url: str,
    ) -> None:
        """
        Args:
        - client_id:     Azure AD app registration client ID.
        - client_secret: Azure AD app registration client secret.
        - tenant_id:     Azure AD tenant ID (not the RAG tenant_id).
        - site_url:      SharePoint site URL,
                         e.g. "https://myorg.sharepoint.com/sites/Engineering"
        """
        if not _O365_AVAILABLE:
            raise ImportError(
                "O365 is required for SharePointConnector. "
                "Install it with: pip install O365"
            )
        credentials = (client_id, client_secret)
        self._account = Account(
            credentials,
            auth_flow_type="credentials",
            tenant_id=tenant_id,
        )
        if not self._account.is_authenticated:
            self._account.authenticate()
        self._site_url = site_url

    @property
    def source_name(self) -> str:
        return "sharepoint"

    def fetch_documents(
        self,
        since: datetime | None = None,
        library_name: str = "Documents",
        **kwargs,
    ) -> list[ConnectorDoc]:
        """
        What problem does this solve?
        - Downloads files from a SharePoint document library to local temp
          files and returns ConnectorDoc objects for SyncService.

        Args:
        - since:        Only fetch files modified after this datetime.
        - library_name: Name of the document library to sync
                        (default "Documents" = the default library).

        Why download to tempfile.mkstemp with suffix?
        - NamedTemporaryFile with delete=False lets the pipeline read the
          file after this method returns. SyncService is responsible for
          deleting temp files after pipeline processing completes.
        """
        try:
            sp = self._account.sharepoint()
            site = sp.get_site(self._site_url)
            drive = site.get_document_library(library_name)
            return self._fetch_drive_items(drive.get_root_folder(), since)
        except Exception as e:
            logger.error("SharePoint fetch failed: %s", e)
            return []

    # ── Private ────────────────────────────────────────────────────────────────

    def _fetch_drive_items(self, folder, since: datetime | None) -> list[ConnectorDoc]:
        """Recursively walk the folder tree, downloading supported file types."""
        docs = []
        try:
            for item in folder.get_items():
                if item.is_folder:
                    docs.extend(self._fetch_drive_items(item, since))
                elif self._should_ingest(item, since):
                    doc = self._download_item(item)
                    if doc:
                        docs.append(doc)
        except Exception as e:
            logger.warning("SharePoint folder walk error: %s", e)
        return docs

    def _should_ingest(self, item, since: datetime | None) -> bool:
        ext = Path(item.name).suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            return False
        if since and item.modified is not None:
            modified_utc = item.modified.astimezone(timezone.utc)
            since_utc = since.astimezone(timezone.utc) if since.tzinfo else since.replace(tzinfo=timezone.utc)
            if modified_utc <= since_utc:
                return False
        return True

    def _download_item(self, item) -> ConnectorDoc | None:
        try:
            suffix = Path(item.name).suffix.lower()
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="sp_")
            os.close(fd)
            item.download(to_path=tmp_path)

            return ConnectorDoc(
                source_url=f"sharepoint::{self._site_url}::{item.object_id}",
                title=Path(item.name).stem,
                source_name=self.source_name,
                content="",
                file_path=tmp_path,
                author=getattr(item, "created_by", {}).get("user", {}).get("displayName", ""),
                last_modified=getattr(item, "modified", None),
                metadata={
                    "site_url": self._site_url,
                    "file_name": item.name,
                    "item_id": item.object_id,
                },
            )
        except Exception as e:
            logger.warning("SharePoint download failed for %s: %s", item.name, e)
            return None
