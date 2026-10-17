"""
backend/routes/documents.py

What problem does this solve?
- Operators need to inspect pipeline results (what's indexed, what failed),
  filter by status for retry queues, and delete documents that should no
  longer be searchable.

Routes:
  GET    /documents           — list documents for caller's tenant
  GET    /documents/{id}      — fetch one document's pipeline record
  DELETE /documents/{id}      — remove the registry record and its chunks

Why scope all queries to rbac.tenant_id?
- Operators must never see another tenant's documents. The tenant_id comes
  from the JWT — no query parameter override is accepted.

Why delete through DocumentService?
- The record lives in `documents` (registry module) and the chunks in
  `document_chunks` (embeddings module). DocumentService clears both, so
  this route never touches either store directly.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.auth import get_rbac_context
from backend.dependencies import get_document_service, get_registry_service
from backend.models import DeleteResponse, DocumentListResponse, DocumentResponse
from agent.documents import DocumentService
from agent.registry import DocumentRecord, RegistryService
from agent.shared import RBACContext

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
def list_documents(
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    rbac: RBACContext = Depends(get_rbac_context),
    registry_svc: RegistryService = Depends(get_registry_service),
) -> DocumentListResponse:
    """
    List documents ingested by the caller's tenant.

    - status: filter by pipeline status
      (completed | duplicate | failed_ingestion | failed_preprocessing |
       failed_chunking | failed_embedding | pending)
    - limit: page size (max 100)
    - offset: pagination offset
    """
    result = registry_svc.list_by_tenant(
        tenant_id=rbac.tenant_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error,
        )
    docs = [_record_to_response(r) for r in result.records]
    return DocumentListResponse(
        documents=docs,
        total=len(docs),
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    rbac: RBACContext = Depends(get_rbac_context),
    registry_svc: RegistryService = Depends(get_registry_service),
) -> DocumentResponse:
    """Fetch pipeline record for a single document."""
    result = registry_svc.get(document_id)
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error,
        )
    if result.record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found",
        )
    if result.record.tenant_id != rbac.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    return _record_to_response(result.record)


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(
    document_id: str,
    rbac: RBACContext = Depends(get_rbac_context),
    registry_svc: RegistryService = Depends(get_registry_service),
    document_svc: DocumentService = Depends(get_document_service),
) -> DeleteResponse:
    """
    Remove a document: deletes both the registry record and all chunks from pgvector.
    """
    get_result = registry_svc.get(document_id)
    if not get_result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=get_result.error,
        )
    if get_result.record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found",
        )
    if get_result.record.tenant_id != rbac.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    delete_result = document_svc.delete(document_id)
    if not delete_result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=delete_result.error,
        )
    return DeleteResponse(deleted=delete_result.deleted, document_id=document_id)


# ── Private ────────────────────────────────────────────────────────────────────

def _record_to_response(record: DocumentRecord) -> DocumentResponse:
    return DocumentResponse(
        id=record.id,
        file_path=record.file_path,
        status=record.status,
        tenant_id=record.tenant_id,
        owner_id=record.owner_id,
        total_chunks=record.total_chunks,
        failed_stage=record.failed_stage,
        error=record.error,
        is_duplicate=record.is_duplicate,
        duplicate_of=record.duplicate_of,
        similarity_score=record.similarity_score,
        total_duration_ms=record.total_duration_ms,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
