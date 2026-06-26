"""
api/routes/sync.py

What problem does this solve?
- Confluence/SharePoint/Jira/Wiki connectors need to be triggered on a schedule
  or on-demand. Without an HTTP endpoint, operators have to SSH into a server
  and run a Python script manually — unscalable and error-prone.

POST /sync/{source}
- source: "confluence" | "sharepoint" | "jira" | "wiki"
- Triggers the pre-configured connector for that source
- Returns SyncResponse with ingested/skipped/failed counts

Why pre-configured connectors instead of accepting credentials in the request?
- Credentials (API tokens, client secrets) must never appear in HTTP request
  bodies — they'd be logged by load balancers and visible in browser history.
  Connectors are configured via environment variables at startup; the API
  only triggers them.

Environment variables for each source:
  Confluence:  CONFLUENCE_URL, CONFLUENCE_TOKEN, CONFLUENCE_USERNAME
  SharePoint:  SP_CLIENT_ID, SP_CLIENT_SECRET, SP_TENANT_ID, SP_SITE_URL
  Jira:        JIRA_URL, JIRA_TOKEN, JIRA_USERNAME
  Wiki:        WIKI_HOST, WIKI_BOT_USER, WIKI_BOT_PASSWORD

Why return 503 if a connector is not configured?
- A 503 (Service Unavailable) clearly signals "the connector exists but is
  not set up" — different from 404 (unknown source) or 500 (internal error).
"""

import os

from fastapi import APIRouter, Depends, HTTPException, status

from agent.api.auth import get_rbac_context
from agent.api.models import SyncRequest, SyncResponse
from agent.connectors.base import BaseConnector
from agent.retrieval.models import RBACContext
from agent.services.pipeline_service import PipelineService
from agent.services.registry_service import RegistryService
from agent.services.sync_service import SyncService
from agent.api.dependencies import get_pipeline_service, get_registry_service

router = APIRouter(prefix="/sync", tags=["sync"])

_KNOWN_SOURCES = {"confluence", "sharepoint", "jira", "wiki"}


@router.post("/{source}", response_model=SyncResponse)
def trigger_sync(
    source: str,
    request: SyncRequest = None,
    rbac: RBACContext = Depends(get_rbac_context),
    pipeline_svc: PipelineService = Depends(get_pipeline_service),
    registry_svc: RegistryService = Depends(get_registry_service),
) -> SyncResponse:
    """
    Trigger an incremental or full sync from an enterprise source.

    - source: confluence | sharepoint | jira | wiki
    - since: ISO-8601 datetime for incremental sync (optional)
    - space_key / project_key / library_name / namespace: source-specific filters
    """
    if request is None:
        request = SyncRequest()

    source = source.lower()
    if source not in _KNOWN_SOURCES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown source '{source}'. Known: {sorted(_KNOWN_SOURCES)}",
        )

    connector = _build_connector(source)

    sync_svc = SyncService(
        pipeline_service=pipeline_svc,
        registry_service=registry_svc,
    )

    # Build source-specific kwargs from request body
    kwargs = {}
    if request.space_key:
        kwargs["space_key"] = request.space_key
    if request.project_key:
        kwargs["project_key"] = request.project_key
    if request.library_name != "Documents":
        kwargs["library_name"] = request.library_name
    if request.namespace != 0:
        kwargs["namespace"] = request.namespace
    if request.include_closed:
        kwargs["include_closed"] = request.include_closed

    result = sync_svc.sync(
        connector=connector,
        tenant_id=rbac.tenant_id,
        owner_id=rbac.user_id,
        since=request.since,
        **kwargs,
    )

    return SyncResponse(
        success=result.success,
        source=result.source,
        fetched=result.fetched,
        ingested=result.ingested,
        skipped=result.skipped,
        failed=result.failed,
        errors=result.errors,
        duration_ms=result.duration_ms,
        error=result.error,
    )


def _build_connector(source: str) -> BaseConnector:
    """
    Build the configured connector for the given source.
    Raises HTTP 503 if required environment variables are not set.
    """
    if source == "confluence":
        url = os.getenv("CONFLUENCE_URL")
        token = os.getenv("CONFLUENCE_TOKEN")
        if not url or not token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Confluence not configured. Set CONFLUENCE_URL and CONFLUENCE_TOKEN.",
            )
        from agent.connectors.confluence import ConfluenceConnector
        return ConfluenceConnector(
            url=url,
            token=token,
            username=os.getenv("CONFLUENCE_USERNAME", ""),
        )

    if source == "sharepoint":
        client_id = os.getenv("SP_CLIENT_ID")
        client_secret = os.getenv("SP_CLIENT_SECRET")
        sp_tenant_id = os.getenv("SP_TENANT_ID")
        site_url = os.getenv("SP_SITE_URL")
        if not all([client_id, client_secret, sp_tenant_id, site_url]):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SharePoint not configured. Set SP_CLIENT_ID, SP_CLIENT_SECRET, SP_TENANT_ID, SP_SITE_URL.",
            )
        from agent.connectors.sharepoint import SharePointConnector
        return SharePointConnector(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=sp_tenant_id,
            site_url=site_url,
        )

    if source == "jira":
        url = os.getenv("JIRA_URL")
        token = os.getenv("JIRA_TOKEN")
        if not url or not token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Jira not configured. Set JIRA_URL and JIRA_TOKEN.",
            )
        from agent.connectors.jira import JiraConnector
        return JiraConnector(
            url=url,
            token=token,
            username=os.getenv("JIRA_USERNAME", ""),
        )

    if source == "wiki":
        host = os.getenv("WIKI_HOST")
        if not host:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Wiki not configured. Set WIKI_HOST.",
            )
        from agent.connectors.wiki import MediaWikiConnector
        return MediaWikiConnector(
            host=host,
            bot_username=os.getenv("WIKI_BOT_USER", ""),
            bot_password=os.getenv("WIKI_BOT_PASSWORD", ""),
        )

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown source: {source}")
