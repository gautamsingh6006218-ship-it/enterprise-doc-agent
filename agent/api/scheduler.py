"""
api/scheduler.py

Scheduled background sync for enterprise connectors (Confluence, Jira, SharePoint, Wiki).

Environment variables:
  SYNC_SOURCES          — comma-separated list: "confluence,jira,sharepoint,wiki"
                          Leave empty to disable scheduled sync.
  SYNC_INTERVAL_MINUTES — how often to run each source (default: 60)
  SYNC_TENANT_ID        — tenant_id used for synced documents (default: "default")
  SYNC_OWNER_ID         — owner_id used for synced documents (default: "system")

How it works:
  - APScheduler runs a background thread-pool job every SYNC_INTERVAL_MINUTES.
  - Each job calls SyncService.sync() for its configured source.
  - Connector credentials come from source-specific env vars (same as POST /sync).
  - Failures per-source are logged but do not stop other sources from syncing.
"""

import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _run_sync(source: str, tenant_id: str, owner_id: str) -> None:
    """Execute a sync for one source. Runs inside APScheduler's thread pool."""
    try:
        from agent.api.routes.sync import _build_connector
        from agent.api.dependencies import get_pipeline_service, get_registry_service
        from agent.services.sync_service import SyncService
        from fastapi import HTTPException

        try:
            connector = _build_connector(source)
        except HTTPException as exc:
            logger.warning("Scheduled sync skipped for '%s': %s", source, exc.detail)
            return

        sync_svc = SyncService(
            pipeline_service=get_pipeline_service(),
            registry_service=get_registry_service(),
        )
        result = sync_svc.sync(
            connector=connector,
            tenant_id=tenant_id,
            owner_id=owner_id,
        )
        if result.success:
            logger.info(
                "Scheduled sync '%s': fetched=%d ingested=%d skipped=%d failed=%d (%.0fms)",
                source,
                result.fetched,
                result.ingested,
                result.skipped,
                result.failed,
                result.duration_ms,
            )
        else:
            logger.error("Scheduled sync '%s' failed: %s", source, result.error)

    except Exception as exc:
        logger.exception("Unexpected error in scheduled sync for '%s': %s", source, exc)


def start_scheduler() -> BackgroundScheduler | None:
    """
    Start the APScheduler background scheduler if SYNC_SOURCES is configured.
    Returns the scheduler instance (or None if sync is disabled).
    Called once at application startup.
    """
    global _scheduler

    sources_raw = os.getenv("SYNC_SOURCES", "").strip()
    if not sources_raw:
        logger.info("Scheduled sync disabled (SYNC_SOURCES not set)")
        return None

    sources = [s.strip().lower() for s in sources_raw.split(",") if s.strip()]
    interval_minutes = int(os.getenv("SYNC_INTERVAL_MINUTES", "60"))
    tenant_id = os.getenv("SYNC_TENANT_ID", "default")
    owner_id = os.getenv("SYNC_OWNER_ID", "system")

    _scheduler = BackgroundScheduler(daemon=True)

    for source in sources:
        _scheduler.add_job(
            _run_sync,
            trigger="interval",
            minutes=interval_minutes,
            args=[source, tenant_id, owner_id],
            id=f"sync_{source}",
            name=f"Auto-sync {source}",
            replace_existing=True,
        )
        logger.info(
            "Registered scheduled sync: source=%s interval=%dm tenant=%s",
            source,
            interval_minutes,
            tenant_id,
        )

    _scheduler.start()
    logger.info("Scheduler started with %d job(s)", len(sources))
    return _scheduler


def stop_scheduler() -> None:
    """Stop the scheduler gracefully at application shutdown."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
    _scheduler = None
