"""
api/routes/ingest.py

What problem does this solve?
- Exposes PipelineService over HTTP so any client (web app, CLI, scheduled job)
  can upload a document without Python imports or direct service access.

POST /ingest
- Accepts: multipart/form-data with `file` + optional RBAC fields
- Returns: IngestResponse (success, document_id, chunk count, timing)

Why multipart/form-data instead of base64 JSON?
- Multipart streams the file directly — no base64 encoding overhead (33% size
  penalty) and no need to buffer the entire file in a JSON string.
- Standard for file upload APIs; works with curl, Postman, and all HTTP clients.

Why compute SHA-256 before calling PipelineService?
- Exact dedup: if the same file bytes were already ingested by this tenant,
  return the existing result immediately without running the full pipeline.
  Saves GPU time and DB writes for identical re-uploads.

Why write to a named temp file?
- PipelineService.run() takes a file path, not bytes. UploadFile is a stream
  that must be read fully before passing to the pipeline. A temp file bridges
  the streaming HTTP layer and the file-path pipeline layer.
"""

import hashlib
import os
import tempfile

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status

from agent.api.auth import get_rbac_context
from agent.api.dependencies import get_pipeline_service, get_registry_service
from agent.api.models import IngestResponse
from agent.retrieval.models import RBACContext
from agent.services.pipeline_service import PipelineService
from agent.services.registry_service import RegistryService

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest_document(
    file: UploadFile = File(..., description="Document file to ingest"),
    visibility: str = Form(default="public"),
    access_roles: str = Form(default=""),
    rbac: RBACContext = Depends(get_rbac_context),
    pipeline_svc: PipelineService = Depends(get_pipeline_service),
    registry_svc: RegistryService = Depends(get_registry_service),
) -> IngestResponse:
    """
    Upload a document and run it through the full ingestion pipeline.

    - Supported formats: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, PNG, JPG, EML
    - access_roles: comma-separated list of roles (e.g. "hr,legal")
    - visibility: "public" | "restricted" | "private"
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Exact-duplicate check: same bytes, same tenant → skip pipeline
    existing = registry_svc.get_by_file_hash(file_hash, rbac.tenant_id)
    if existing.success and existing.record is not None:
        rec = existing.record
        return IngestResponse(
            success=True,
            document_id=rec.id,
            total_chunks=rec.total_chunks,
            is_duplicate=True,
            duplicate_of=rec.id,
            similarity_score=1.0,
            total_duration_ms=0.0,
        )

    # Write to named temp file preserving original extension
    original_name = file.filename or "upload.bin"
    suffix = os.path.splitext(original_name)[1] or ".bin"
    roles = [r.strip() for r in access_roles.split(",") if r.strip()]

    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="ingest_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)

        result = pipeline_svc.run(
            file_path=tmp_path,
            tenant_id=rbac.tenant_id,
            owner_id=rbac.user_id,
            access_roles=roles,
            visibility=visibility,
            file_hash=file_hash,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

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
