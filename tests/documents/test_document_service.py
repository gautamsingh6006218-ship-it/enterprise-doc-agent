"""
tests/documents/test_document_service.py

What does this cover?
- DocumentService.delete(): chunks deleted before the registry record,
                            registry result returned unchanged.
- DocumentService.remove_stale_version():
    no document with that filename → None, nothing deleted.
    registry lookup fails          → None, nothing deleted.
    same filename, same hash       → existing record, nothing deleted.
    same filename, new hash        → None, old chunks + record deleted.
- DocumentService.ingest():
    new file                       → pipeline runs on a temp copy with the
                                     original filename and the SHA-256 hash.
    same content already indexed   → duplicate of that record, no pipeline.
    replace=True                   → content check skipped, pipeline runs.
    same filename, same content    → duplicate of that record, no pipeline.
    same filename, new content     → old version deleted, then pipeline runs.
"""

import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock, call

from agent.documents.service import DocumentService
from agent.embeddings.store import PgVectorStore
from agent.pipeline import PipelineResult, PipelineService
from agent.registry.models import DocumentRecord
from agent.registry.service import RegistryResult, RegistryService


# ── Helpers ────────────────────────────────────────────────────────────────────

def _record(doc_id="doc-old", file_hash="hash-old"):
    return DocumentRecord(
        id=doc_id, file_path="report.pdf", file_hash=file_hash,
        status="completed", tenant_id="t1", owner_id="u1", total_chunks=4,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _service(lookup: RegistryResult | None = None):
    """DocumentService over mocks that share one parent to record call order."""
    svc, registry, chunks, _, calls = _ingest_service(by_name=lookup)
    return svc, registry, chunks, calls


def _ingest_service(by_hash: RegistryResult | None = None, by_name: RegistryResult | None = None):
    """DocumentService over mocks that share one parent to record call order."""
    calls = MagicMock()
    registry = MagicMock(spec=RegistryService)
    chunks = MagicMock(spec=PgVectorStore)
    pipeline = MagicMock(spec=PipelineService)
    calls.attach_mock(registry, "registry")
    calls.attach_mock(chunks, "chunks")
    calls.attach_mock(pipeline, "pipeline")
    registry.delete.return_value = RegistryResult(success=True, deleted=True)
    registry.get_by_file_hash.return_value = by_hash or RegistryResult(success=True, record=None)
    registry.get_by_original_filename.return_value = (
        by_name or RegistryResult(success=True, record=None)
    )
    pipeline.run.return_value = PipelineResult(success=True, file_path="/tmp/x", document_id="doc-new")
    svc = DocumentService(registry_service=registry, chunk_store=chunks, pipeline_service=pipeline)
    return svc, registry, chunks, pipeline, calls


# ── delete ─────────────────────────────────────────────────────────────────────

class TestDelete:

    def test_deletes_chunks_then_record(self):
        svc, _, _, calls = _service()
        svc.delete("doc-1")
        assert calls.mock_calls == [
            call.chunks.delete_by_document_id("doc-1"),
            call.registry.delete("doc-1"),
        ]

    def test_returns_registry_result(self):
        svc, registry, _, _ = _service()
        registry.delete.return_value = RegistryResult(success=False, error="db down")
        result = svc.delete("doc-1")
        assert result.success is False
        assert result.error == "db down"


# ── remove_stale_version ───────────────────────────────────────────────────────

class TestRemoveStaleVersion:

    def test_no_match_returns_none(self):
        svc, registry, chunks, _ = _service()
        assert svc.remove_stale_version("report.pdf", "hash-new", "t1") is None
        registry.get_by_original_filename.assert_called_once_with("report.pdf", "t1")
        chunks.delete_by_document_id.assert_not_called()
        registry.delete.assert_not_called()

    def test_lookup_failure_returns_none(self):
        svc, registry, chunks, _ = _service(RegistryResult(success=False, error="db down"))
        assert svc.remove_stale_version("report.pdf", "hash-new", "t1") is None
        chunks.delete_by_document_id.assert_not_called()
        registry.delete.assert_not_called()

    def test_same_hash_returns_existing_record(self):
        existing = _record(file_hash="hash-same")
        svc, registry, chunks, _ = _service(RegistryResult(success=True, record=existing))
        assert svc.remove_stale_version("report.pdf", "hash-same", "t1") is existing
        chunks.delete_by_document_id.assert_not_called()
        registry.delete.assert_not_called()

    def test_changed_hash_deletes_old_version(self):
        svc, _, _, calls = _service(RegistryResult(success=True, record=_record()))
        assert svc.remove_stale_version("report.pdf", "hash-new", "t1") is None
        assert calls.mock_calls == [
            call.registry.get_by_original_filename("report.pdf", "t1"),
            call.chunks.delete_by_document_id("doc-old"),
            call.registry.delete("doc-old"),
        ]


# ── ingest ─────────────────────────────────────────────────────────────────────

class TestIngest:

    def _ingest(self, svc, data=b"content", **kwargs):
        return svc.ingest(
            file_bytes=data, original_filename="report.pdf",
            tenant_id="t1", owner_id="u1", **kwargs,
        )

    def test_new_file_runs_pipeline(self):
        svc, _, _, pipeline, _ = _ingest_service()
        result = self._ingest(svc, access_roles=["hr"], visibility="restricted")
        assert result is pipeline.run.return_value
        kwargs = pipeline.run.call_args.kwargs
        assert kwargs["original_filename"] == "report.pdf"
        assert kwargs["file_hash"] == _sha256(b"content")
        assert kwargs["tenant_id"] == "t1"
        assert kwargs["owner_id"] == "u1"
        assert kwargs["access_roles"] == ["hr"]
        assert kwargs["visibility"] == "restricted"

    def test_pipeline_reads_temp_copy_that_is_removed_after(self):
        svc, _, _, pipeline, _ = _ingest_service()
        seen = {}

        def run(file_path, **kwargs):
            seen["path"] = file_path
            seen["bytes"] = Path(file_path).read_bytes()
            return PipelineResult(success=True, file_path=file_path)

        pipeline.run.side_effect = run
        self._ingest(svc)
        assert seen["bytes"] == b"content"
        assert seen["path"].endswith(".pdf")
        assert not os.path.exists(seen["path"])

    def test_same_content_already_indexed_is_duplicate(self):
        existing = _record(doc_id="doc-same", file_hash=_sha256(b"content"))
        svc, registry, _, pipeline, _ = _ingest_service(
            by_hash=RegistryResult(success=True, record=existing),
        )
        result = self._ingest(svc)
        registry.get_by_file_hash.assert_called_once_with(_sha256(b"content"), "t1")
        assert result.success is True
        assert result.is_duplicate is True
        assert result.document_id == "doc-same"
        assert result.duplicate_of == "doc-same"
        assert result.total_chunks == 4
        assert result.similarity_score == 1.0
        pipeline.run.assert_not_called()

    def test_replace_skips_content_check(self):
        existing = _record(doc_id="doc-same", file_hash=_sha256(b"content"))
        svc, registry, _, pipeline, _ = _ingest_service(
            by_hash=RegistryResult(success=True, record=existing),
        )
        self._ingest(svc, replace=True)
        registry.get_by_file_hash.assert_not_called()
        pipeline.run.assert_called_once()

    def test_same_name_same_content_is_duplicate(self):
        existing = _record(file_hash=_sha256(b"content"))
        svc, registry, chunks, pipeline, _ = _ingest_service(
            by_name=RegistryResult(success=True, record=existing),
        )
        result = self._ingest(svc)
        assert result.is_duplicate is True
        assert result.document_id == "doc-old"
        pipeline.run.assert_not_called()
        chunks.delete_by_document_id.assert_not_called()
        registry.delete.assert_not_called()

    def test_same_name_new_content_replaces_old_version(self):
        svc, _, _, _, calls = _ingest_service(
            by_name=RegistryResult(success=True, record=_record(file_hash="hash-old")),
        )
        self._ingest(svc)
        names = [c[0] for c in calls.mock_calls]
        assert names == [
            "registry.get_by_file_hash",
            "registry.get_by_original_filename",
            "chunks.delete_by_document_id",
            "registry.delete",
            "pipeline.run",
        ]
