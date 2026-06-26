"""
tests/pipeline/test_registry.py

What does this cover?
- DocumentRecord:        field defaults, status constants.
- _status_from_result(): maps PipelineResult → correct status string.
- PipelineService.run(): pipeline called with RBAC args, result returned,
                         registry.upsert called on success/failure/duplicate,
                         registry write failure does NOT propagate.
- RegistryService.get():       success, not-found, store error.
- RegistryService.get_by_file_hash(): found, not-found, store error.
- RegistryService.list_by_tenant():   returns records, limit capped at 500,
                                       status filter forwarded.
- RegistryService.delete():   deleted=True, deleted=False, store error.
- DocumentRegistryStore._row_to_record(): column mapping.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from agent.pipeline.models import PipelineResult, StageResult
from agent.registry.models import DocumentRecord, STATUSES
from agent.registry.store import DocumentRegistryStore
from agent.services.pipeline_service import PipelineService, _status_from_result
from agent.services.registry_service import RegistryResult, RegistryService


# ── Helpers ────────────────────────────────────────────────────────────────────

def _success_result(doc_id="doc-1", chunks=3):
    return PipelineResult(
        success=True,
        file_path="/data/report.pdf",
        document_id=doc_id,
        total_chunks=chunks,
        total_duration_ms=250.0,
        stages=[
            StageResult(stage="ingestion", success=True, duration_ms=10.0),
            StageResult(stage="preprocessing", success=True, duration_ms=20.0),
            StageResult(stage="chunking", success=True, duration_ms=30.0),
            StageResult(stage="embedding", success=True, duration_ms=190.0),
        ],
    )


def _failed_result(failed_stage="chunking"):
    return PipelineResult(
        success=False,
        file_path="/data/report.pdf",
        document_id="doc-1",
        failed_stage=failed_stage,
        error=f"{failed_stage} exploded",
        total_duration_ms=80.0,
        stages=[],
    )


def _duplicate_result():
    return PipelineResult(
        success=True,
        file_path="/data/dup.pdf",
        document_id="doc-existing",
        is_duplicate=True,
        duplicate_of="doc-original",
        similarity_score=0.91,
        total_duration_ms=15.0,
        stages=[],
    )


def _make_pipeline_service(pipeline_result, registry=None):
    pipeline = MagicMock()
    pipeline.run.return_value = pipeline_result
    return PipelineService(pipeline=pipeline, registry_store=registry), pipeline


# ── DocumentRecord ─────────────────────────────────────────────────────────────

class TestDocumentRecord:

    def test_required_fields(self):
        r = DocumentRecord(
            id="doc-1", file_path="/a.pdf", status="completed",
            tenant_id="t1", owner_id="u1"
        )
        assert r.id == "doc-1"
        assert r.status == "completed"

    def test_default_total_chunks_zero(self):
        r = DocumentRecord(id="x", file_path="/f", status="pending",
                           tenant_id="t", owner_id="u")
        assert r.total_chunks == 0

    def test_default_not_duplicate(self):
        r = DocumentRecord(id="x", file_path="/f", status="pending",
                           tenant_id="t", owner_id="u")
        assert r.is_duplicate is False
        assert r.duplicate_of is None

    def test_statuses_contains_all_expected(self):
        expected = {
            "pending", "completed", "duplicate",
            "failed_ingestion", "failed_preprocessing",
            "failed_chunking", "failed_embedding",
        }
        assert expected == STATUSES


# ── _status_from_result ────────────────────────────────────────────────────────

class TestStatusFromResult:

    def test_success_returns_completed(self):
        assert _status_from_result(_success_result()) == "completed"

    def test_duplicate_returns_duplicate(self):
        assert _status_from_result(_duplicate_result()) == "duplicate"

    def test_failed_ingestion(self):
        assert _status_from_result(_failed_result("ingestion")) == "failed_ingestion"

    def test_failed_preprocessing(self):
        assert _status_from_result(_failed_result("preprocessing")) == "failed_preprocessing"

    def test_failed_chunking(self):
        assert _status_from_result(_failed_result("chunking")) == "failed_chunking"

    def test_failed_embedding(self):
        assert _status_from_result(_failed_result("embedding")) == "failed_embedding"

    def test_failed_no_stage_defaults_to_ingestion(self):
        r = PipelineResult(success=False, file_path="/f", failed_stage=None, error="x")
        assert _status_from_result(r) == "failed_ingestion"


# ── PipelineService ────────────────────────────────────────────────────────────

class TestPipelineService:

    def test_returns_pipeline_result(self):
        svc, _ = _make_pipeline_service(_success_result())
        result = svc.run("/data/report.pdf")
        assert result.success is True
        assert result.total_chunks == 3

    def test_pipeline_called_with_rbac_args(self):
        svc, pipeline = _make_pipeline_service(_success_result())
        svc.run("/data/doc.pdf", tenant_id="t1", owner_id="u1",
                access_roles=["hr"], visibility="restricted")
        pipeline.run.assert_called_once_with(
            "/data/doc.pdf",
            tenant_id="t1",
            owner_id="u1",
            access_roles=["hr"],
            visibility="restricted",
        )

    def test_registry_upsert_called_on_success(self):
        registry = MagicMock()
        svc, _ = _make_pipeline_service(_success_result(), registry=registry)
        svc.run("/data/doc.pdf", tenant_id="t1", owner_id="u1", file_hash="abc123")
        registry.upsert.assert_called_once()
        record = registry.upsert.call_args[0][0]
        assert record.status == "completed"
        assert record.file_hash == "abc123"

    def test_registry_upsert_called_on_failure(self):
        registry = MagicMock()
        svc, _ = _make_pipeline_service(_failed_result("embedding"), registry=registry)
        svc.run("/data/doc.pdf")
        registry.upsert.assert_called_once()
        record = registry.upsert.call_args[0][0]
        assert record.status == "failed_embedding"

    def test_registry_upsert_called_on_duplicate(self):
        registry = MagicMock()
        svc, _ = _make_pipeline_service(_duplicate_result(), registry=registry)
        svc.run("/data/dup.pdf")
        registry.upsert.assert_called_once()
        record = registry.upsert.call_args[0][0]
        assert record.status == "duplicate"
        assert record.is_duplicate is True
        assert record.duplicate_of == "doc-original"

    def test_registry_none_does_not_raise(self):
        svc, _ = _make_pipeline_service(_success_result(), registry=None)
        result = svc.run("/data/doc.pdf")
        assert result.success is True

    def test_registry_write_failure_does_not_propagate(self):
        registry = MagicMock()
        registry.upsert.side_effect = RuntimeError("DB connection lost")
        svc, _ = _make_pipeline_service(_success_result(), registry=registry)
        result = svc.run("/data/doc.pdf")
        # Pipeline result still returned despite registry failure
        assert result.success is True

    def test_record_total_chunks_matches_result(self):
        registry = MagicMock()
        svc, _ = _make_pipeline_service(_success_result(chunks=7), registry=registry)
        svc.run("/data/doc.pdf")
        record = registry.upsert.call_args[0][0]
        assert record.total_chunks == 7

    def test_record_tenant_owner_forwarded(self):
        registry = MagicMock()
        svc, _ = _make_pipeline_service(_success_result(), registry=registry)
        svc.run("/data/doc.pdf", tenant_id="acme", owner_id="alice")
        record = registry.upsert.call_args[0][0]
        assert record.tenant_id == "acme"
        assert record.owner_id == "alice"

    def test_record_duration_forwarded(self):
        registry = MagicMock()
        result = _success_result()
        result.total_duration_ms = 312.5
        svc, _ = _make_pipeline_service(result, registry=registry)
        svc.run("/data/doc.pdf")
        record = registry.upsert.call_args[0][0]
        assert record.total_duration_ms == pytest.approx(312.5)


# ── RegistryService ────────────────────────────────────────────────────────────

def _make_registry_service():
    store = MagicMock(spec=DocumentRegistryStore)
    return RegistryService(store=store), store


def _sample_record(doc_id="doc-1"):
    return DocumentRecord(
        id=doc_id, file_path="/a.pdf", status="completed",
        tenant_id="t1", owner_id="u1", total_chunks=5
    )


class TestRegistryServiceGet:

    def test_success_with_record(self):
        svc, store = _make_registry_service()
        store.get.return_value = _sample_record()
        result = svc.get("doc-1")
        assert result.success is True
        assert result.record.id == "doc-1"

    def test_not_found_returns_none_record(self):
        svc, store = _make_registry_service()
        store.get.return_value = None
        result = svc.get("nonexistent")
        assert result.success is True
        assert result.record is None

    def test_store_error_returns_failure(self):
        svc, store = _make_registry_service()
        store.get.side_effect = RuntimeError("DB down")
        result = svc.get("doc-1")
        assert result.success is False
        assert "DB down" in result.error


class TestRegistryServiceGetByFileHash:

    def test_found_returns_record(self):
        svc, store = _make_registry_service()
        store.get_by_file_hash.return_value = _sample_record()
        result = svc.get_by_file_hash("sha256abc", "t1")
        assert result.success is True
        assert result.record is not None

    def test_not_found_returns_none(self):
        svc, store = _make_registry_service()
        store.get_by_file_hash.return_value = None
        result = svc.get_by_file_hash("sha256abc", "t1")
        assert result.success is True
        assert result.record is None

    def test_store_error(self):
        svc, store = _make_registry_service()
        store.get_by_file_hash.side_effect = Exception("timeout")
        result = svc.get_by_file_hash("sha256abc", "t1")
        assert result.success is False


class TestRegistryServiceList:

    def test_returns_records(self):
        svc, store = _make_registry_service()
        store.list_by_tenant.return_value = [_sample_record("d1"), _sample_record("d2")]
        result = svc.list_by_tenant("t1")
        assert result.success is True
        assert len(result.records) == 2

    def test_empty_list(self):
        svc, store = _make_registry_service()
        store.list_by_tenant.return_value = []
        result = svc.list_by_tenant("t1")
        assert result.success is True
        assert result.records == []

    def test_limit_capped_at_500(self):
        svc, store = _make_registry_service()
        store.list_by_tenant.return_value = []
        svc.list_by_tenant("t1", limit=9999)
        _, kwargs = store.list_by_tenant.call_args
        assert kwargs["limit"] == 500

    def test_status_filter_forwarded(self):
        svc, store = _make_registry_service()
        store.list_by_tenant.return_value = []
        svc.list_by_tenant("t1", status="failed_embedding")
        _, kwargs = store.list_by_tenant.call_args
        assert kwargs["status"] == "failed_embedding"

    def test_offset_forwarded(self):
        svc, store = _make_registry_service()
        store.list_by_tenant.return_value = []
        svc.list_by_tenant("t1", offset=50)
        _, kwargs = store.list_by_tenant.call_args
        assert kwargs["offset"] == 50

    def test_store_error(self):
        svc, store = _make_registry_service()
        store.list_by_tenant.side_effect = Exception("connection reset")
        result = svc.list_by_tenant("t1")
        assert result.success is False

    def test_default_records_empty_list_not_none(self):
        result = RegistryResult(success=True)
        assert result.records == []


class TestRegistryServiceDelete:

    def test_deleted_true(self):
        svc, store = _make_registry_service()
        store.delete.return_value = True
        result = svc.delete("doc-1")
        assert result.success is True
        assert result.deleted is True

    def test_deleted_false_when_not_found(self):
        svc, store = _make_registry_service()
        store.delete.return_value = False
        result = svc.delete("nonexistent")
        assert result.success is True
        assert result.deleted is False

    def test_store_error(self):
        svc, store = _make_registry_service()
        store.delete.side_effect = Exception("lock timeout")
        result = svc.delete("doc-1")
        assert result.success is False
        assert "lock timeout" in result.error


# ── DocumentRegistryStore._row_to_record ──────────────────────────────────────

class TestRowToRecord:

    def _make_description(self, col_names):
        """Simulate psycopg2 cursor.description."""
        desc = []
        for name in col_names:
            col = MagicMock()
            col.name = name
            desc.append(col)
        return desc

    def test_all_fields_mapped(self):
        cols = [
            "id", "file_path", "file_hash", "status",
            "tenant_id", "owner_id", "total_chunks",
            "failed_stage", "error", "is_duplicate",
            "duplicate_of", "similarity_score",
            "total_duration_ms", "created_at", "updated_at",
        ]
        row = (
            "doc-99", "/data/f.pdf", "sha256xyz", "completed",
            "tenant-1", "user-1", 12,
            None, None, False,
            None, 0.0,
            350.5, None, None,
        )
        desc = self._make_description(cols)
        record = DocumentRegistryStore._row_to_record(row, desc)

        assert record.id == "doc-99"
        assert record.file_path == "/data/f.pdf"
        assert record.file_hash == "sha256xyz"
        assert record.status == "completed"
        assert record.tenant_id == "tenant-1"
        assert record.owner_id == "user-1"
        assert record.total_chunks == 12
        assert record.total_duration_ms == pytest.approx(350.5)

    def test_optional_fields_default_on_missing(self):
        cols = ["id", "file_path", "status", "tenant_id", "owner_id"]
        row = ("doc-1", "/f.pdf", "pending", "t1", "u1")
        desc = self._make_description(cols)
        record = DocumentRegistryStore._row_to_record(row, desc)
        assert record.total_chunks == 0
        assert record.is_duplicate is False
        assert record.similarity_score == 0.0
