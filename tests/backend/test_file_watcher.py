"""
tests/backend/test_file_watcher.py

What does this cover?
- _IngestHandler._ingest(): hands the file bytes and name to
  DocumentService.ingest() with the watcher's tenant, owner and visibility.
- A file deleted before the debounce fires is ignored.
- A service error is logged and does not escape the watcher thread.
"""

from unittest.mock import MagicMock

from agent.documents import DocumentService
from agent.pipeline import PipelineResult
from backend.file_watcher import _IngestHandler


def _handler(document_svc):
    return _IngestHandler(
        tenant_id="t1",
        owner_id="system",
        visibility="public",
        get_document_service=lambda: document_svc,
    )


def _document_service(**result_fields):
    svc = MagicMock(spec=DocumentService)
    svc.ingest.return_value = PipelineResult(success=True, file_path="report.pdf", **result_fields)
    return svc


class TestIngest:

    def test_passes_file_to_document_service(self, tmp_path):
        path = tmp_path / "report.pdf"
        path.write_bytes(b"content")
        document_svc = _document_service(total_chunks=2)

        _handler(document_svc)._ingest(str(path))

        document_svc.ingest.assert_called_once_with(
            file_bytes=b"content",
            original_filename="report.pdf",
            tenant_id="t1",
            owner_id="system",
            visibility="public",
        )

    def test_deleted_file_is_ignored(self, tmp_path):
        document_svc = _document_service()
        _handler(document_svc)._ingest(str(tmp_path / "gone.pdf"))
        document_svc.ingest.assert_not_called()

    def test_service_error_does_not_propagate(self, tmp_path):
        path = tmp_path / "report.pdf"
        path.write_bytes(b"content")
        document_svc = _document_service()
        document_svc.ingest.side_effect = RuntimeError("db down")

        _handler(document_svc)._ingest(str(path))
