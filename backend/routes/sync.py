"""
backend/routes/sync.py

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

from fastapi import APIRouter, Depends, HTTPException, status

from backend.auth import get_rbac_context
from backend.dependencies import get_pipeline_service, get_registry_service
from backend.models import SyncRequest, SyncResponse
from agent.connectors import ConnectorNotConfiguredError, UnknownSourceError, build_connector
from agent.pipeline import PipelineService
from agent.registry import RegistryService
from agent.shared import RBACContext
from agent.sync import SyncService

router = APIRouter(prefix="/sync", tags=["sync"])


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
    try:
        connector = build_connector(source)
    except UnknownSourceError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ConnectorNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

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
