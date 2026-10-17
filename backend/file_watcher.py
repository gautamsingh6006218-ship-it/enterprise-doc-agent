"""
backend/file_watcher.py

Watches a local directory for new or modified files and automatically
ingests them through the full pipeline.

Environment variables:
  WATCH_DIR          — absolute path to directory to watch (required to enable)
  WATCH_TENANT_ID    — tenant_id for ingested documents (default: "default")
  WATCH_OWNER_ID     — owner_id for ingested documents (default: "system")
  WATCH_VISIBILITY   — visibility for ingested documents (default: "public")
  WATCH_RECURSIVE    — "true" to watch subdirectories (default: "false")

How it works:
  - watchdog observes WATCH_DIR for file system events.
  - On file create or modify: debounced 3s to avoid re-triggering on partial writes.
  - Hands the bytes to DocumentService.ingest(), which skips unchanged files,
    replaces changed ones, and runs the full pipeline.
  - Files that fail ingestion are logged but do not crash the watcher.

Debounce:
  - Some editors write files in multiple flushes. A 3-second debounce
    ensures the file is fully written before ingestion starts.
  - A threading.Timer resets on every new event for the same path.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Callable

from agent.documents import DocumentService

logger = logging.getLogger(__name__)

try:
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
    from watchdog.observers import Observer
    _WATCHDOG_AVAILABLE = True
except ImportError:
    _WATCHDOG_AVAILABLE = False

_SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".markdown", ".html", ".htm",
    ".docx", ".pptx", ".xlsx", ".csv", ".odt", ".rtf", ".epub",
    ".eml", ".png", ".jpg", ".jpeg", ".tiff",
}

_DEBOUNCE_SECONDS = 3.0


class _IngestHandler(FileSystemEventHandler if _WATCHDOG_AVAILABLE else object):
    """
    Handles watchdog file-system events and triggers the ingestion pipeline.
    Debounces rapid writes by resetting a timer on every new event.
    """

    def __init__(
        self,
        tenant_id: str,
        owner_id: str,
        visibility: str,
        get_document_service: Callable[[], DocumentService],
    ) -> None:
        if _WATCHDOG_AVAILABLE:
            super().__init__()
        self._tenant_id = tenant_id
        self._owner_id = owner_id
        self._visibility = visibility
        self._get_document_service = get_document_service
        self._pending: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def _schedule(self, path: str) -> None:
        ext = Path(path).suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            return

        with self._lock:
            existing = self._pending.get(path)
            if existing:
                existing.cancel()
            timer = threading.Timer(
                _DEBOUNCE_SECONDS,
                self._ingest,
                args=[path],
            )
            self._pending[path] = timer
            timer.start()

    def _ingest(self, path: str) -> None:
        with self._lock:
            self._pending.pop(path, None)

        if not os.path.exists(path):
            return

        try:
            file_bytes = Path(path).read_bytes()
        except OSError as exc:
            logger.warning("File watcher: cannot read '%s': %s", path, exc)
            return

        original_name = Path(path).name

        try:
            result = self._get_document_service().ingest(
                file_bytes=file_bytes,
                original_filename=original_name,
                tenant_id=self._tenant_id,
                owner_id=self._owner_id,
                visibility=self._visibility,
            )

            if result.is_duplicate:
                logger.debug(
                    "File watcher: '%s' is a duplicate of %s, skipping",
                    original_name,
                    result.duplicate_of,
                )
            elif result.success:
                logger.info(
                    "File watcher: ingested '%s' → %d chunks (%.0fms)",
                    original_name,
                    result.total_chunks,
                    result.total_duration_ms,
                )
            else:
                logger.error(
                    "File watcher: ingestion failed for '%s' at stage '%s': %s",
                    original_name,
                    result.failed_stage,
                    result.error,
                )

        except Exception as exc:
            logger.exception("File watcher: unexpected error ingesting '%s': %s", path, exc)


_observer: "Observer | None" = None


def start_file_watcher(
    get_document_service: Callable[[], DocumentService],
) -> "Observer | None":
    """
    Start the watchdog observer if WATCH_DIR is configured.
    Returns the observer instance (or None if disabled).
    Called once at application startup.

    Why take a service factory instead of importing backend.dependencies?
    - Tests pass a fake service without patching module state. The factory
      runs per event, so services still load lazily on the first file.
    """
    global _observer

    if not _WATCHDOG_AVAILABLE:
        logger.warning("File watcher disabled: watchdog package not installed")
        return None

    watch_dir = os.getenv("WATCH_DIR", "").strip()
    if not watch_dir:
        logger.info("File watcher disabled (WATCH_DIR not set)")
        return None

    if not os.path.isdir(watch_dir):
        logger.error("File watcher: WATCH_DIR '%s' does not exist", watch_dir)
        return None

    tenant_id = os.getenv("WATCH_TENANT_ID", "default")
    owner_id = os.getenv("WATCH_OWNER_ID", "system")
    visibility = os.getenv("WATCH_VISIBILITY", "public")
    recursive = os.getenv("WATCH_RECURSIVE", "false").lower() == "true"

    handler = _IngestHandler(
        tenant_id=tenant_id,
        owner_id=owner_id,
        visibility=visibility,
        get_document_service=get_document_service,
    )
    _observer = Observer()
    _observer.schedule(handler, path=watch_dir, recursive=recursive)
    _observer.start()

    logger.info(
        "File watcher started: dir=%s recursive=%s tenant=%s",
        watch_dir,
        recursive,
        tenant_id,
    )
    return _observer


def stop_file_watcher() -> None:
    """Stop the watchdog observer at application shutdown."""
    global _observer
    if _observer and _observer.is_alive():
        _observer.stop()
        _observer.join(timeout=5)
        logger.info("File watcher stopped")
    _observer = None
