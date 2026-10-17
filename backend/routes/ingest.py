"""
backend/routes/ingest.py

What problem does this solve?
- Exposes document ingestion over HTTP so any client (web app, CLI, scheduled job)
  can upload a document without Python imports or direct service access.

POST /ingest
- Accepts: multipart/form-data with `file` + optional RBAC fields
- Returns: IngestResponse (success, document_id, chunk count, timing)

Why multipart/form-data instead of base64 JSON?
- Multipart streams the file directly — no base64 encoding overhead (33% size
  penalty) and no need to buffer the entire file in a JSON string.
- Standard for file upload APIs; works with curl, Postman, and all HTTP clients.

Why hand the bytes to DocumentService.ingest()?
- Exact dedup, replacing a changed file, and running the pipeline are RAG
  logic owned by the documents module. This route only validates the upload
  and maps the result to an HTTP response.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status

from backend.auth import get_rbac_context
from backend.dependencies import get_document_service
from backend.models import BatchIngestResponse, IngestResponse
from agent.documents import DocumentService
from agent.pipeline import PipelineResult
from agent.shared import RBACContext

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest_document(
    file: UploadFile = File(..., description="Document file to ingest"),
    visibility: str = Form(default="public"),
    access_roles: str = Form(default=""),
    replace: bool = Form(default=False, description="Delete and re-ingest if file with same name already exists"),
    rbac: RBACContext = Depends(get_rbac_context),
    document_svc: DocumentService = Depends(get_document_service),
) -> IngestResponse:
    """
    Upload a document and run it through the full ingestion pipeline.

    - Supported formats: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, PNG, JPG, EML
    - access_roles: comma-separated list of roles (e.g. "hr,legal")
    - visibility: "public" | "restricted" | "private"
    - replace: if true, delete existing document with same content hash before re-ingesting
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    result = document_svc.ingest(
        file_bytes=file_bytes,
        original_filename=file.filename or "upload.bin",
        tenant_id=rbac.tenant_id,
        owner_id=rbac.user_id,
        access_roles=[r.strip() for r in access_roles.split(",") if r.strip()],
        visibility=visibility,
        replace=replace,
    )
    return _to_response(result)


@router.post("/batch", response_model=BatchIngestResponse, status_code=status.HTTP_200_OK)
async def ingest_batch(
    files: List[UploadFile] = File(..., description="Multiple document files to ingest"),
    visibility: str = Form(default="public"),
    access_roles: str = Form(default=""),
    rbac: RBACContext = Depends(get_rbac_context),
    document_svc: DocumentService = Depends(get_document_service),
) -> BatchIngestResponse:
    """
    Upload multiple documents in one request and run each through the ingestion pipeline.

    - Returns per-file results and aggregate counts.
    - Files are processed sequentially (BGE-M3 is single-threaded).
    - Supported formats: same as POST /ingest
    - Same filename with changed content replaces the old version, as in POST /ingest
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )

    roles = [r.strip() for r in access_roles.split(",") if r.strip()]
    results: list[IngestResponse] = []
    succeeded = failed = duplicates = 0

    for upload in files:
        file_bytes = await upload.read()
        if not file_bytes:
            results.append(IngestResponse(
                success=False,
                error=f"{upload.filename}: empty file",
            ))
            failed += 1
            continue

        result = document_svc.ingest(
            file_bytes=file_bytes,
            original_filename=upload.filename or "upload.bin",
            tenant_id=rbac.tenant_id,
            owner_id=rbac.user_id,
            access_roles=roles,
            visibility=visibility,
        )
        results.append(_to_response(result))
        if result.is_duplicate:
            duplicates += 1
        elif result.success:
            succeeded += 1
        else:
            failed += 1

    return BatchIngestResponse(
        total=len(files),
        succeeded=succeeded,
        failed=failed,
        duplicates=duplicates,
        results=results,
    )


# ── Private ────────────────────────────────────────────────────────────────────

def _to_response(result: PipelineResult) -> IngestResponse:
    return IngestResponse(
        success=result.success,
        document_id=result.document_id,
        total_chunks=result.total_chunks,
        is_duplicate=result.is_duplicate,
        duplicate_of=result.duplicate_of,
        similarity_score=result.similarity_score,
        total_duration_ms=result.total_duration_ms,
        failed_stage=result.failed_stage,
        error=result.error,
    )
