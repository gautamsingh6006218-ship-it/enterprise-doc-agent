"""
tests/pipeline/test_document_pipeline.py

What does this cover?
- DocumentPipeline.run() golden path: all 4 stages succeed → PipelineResult(success=True).
- Stage failure propagation: ingestion/preprocessing/chunking/embedding failure
  → success=False with correct failed_stage.
- Near-duplicate shortcircuit: ingestion returns is_duplicate=True → pipeline
  returns success=True, is_duplicate=True, skips stages 2-4.
- Stage timing: all stages have non-negative duration_ms in PipelineResult.stages.
- RBAC forwarding: tenant_id, owner_id, access_roles, visibility passed to ingestion.
- StageResult list length and names for each failure point.
"""

from unittest.mock import MagicMock

import pytest

from agent.pipeline.models import PipelineResult, StageResult
from agent.pipeline.orchestrator import DocumentPipeline
from agent.processing.models import ProcessedDocument
from agent.services.chunking_service import ChunkingResult
from agent.services.embedding_service import EmbeddingResult
from agent.services.ingestion_service import IngestionResult
from agent.services.preprocessing_service import PreprocessingResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_document(doc_id="doc-1"):
    doc = MagicMock()
    doc.id = doc_id
    doc.text = "Sample text."
    return doc


def _mock_processed_document():
    return MagicMock(spec=ProcessedDocument)


def _mock_chunk():
    c = MagicMock()
    c.id = "doc-1_chunk_0"
    return c


def _make_pipeline(
    ingest_result=None,
    preprocess_result=None,
    chunk_result=None,
    embed_result=None,
):
    """Build a DocumentPipeline with all 4 services mocked."""
    doc = _mock_document()
    processed = _mock_processed_document()
    chunk = _mock_chunk()

    ingest_svc = MagicMock()
    ingest_svc.ingest.return_value = ingest_result or IngestionResult(
        success=True, document=doc
    )

    preprocess_svc = MagicMock()
    preprocess_svc.process.return_value = preprocess_result or PreprocessingResult(
        success=True, processed_document=processed
    )

    chunking_svc = MagicMock()
    chunking_svc.chunk.return_value = chunk_result or ChunkingResult(
        success=True, chunks=[chunk], stats={"total_chunks": 1}
    )

    embedding_svc = MagicMock()
    embedding_svc.embed.return_value = embed_result or EmbeddingResult(
        success=True, embedded_count=1
    )

    pipeline = DocumentPipeline(
        ingestion_service=ingest_svc,
        preprocessing_service=preprocess_svc,
        chunking_service=chunking_svc,
        embedding_service=embedding_svc,
    )
    return pipeline, ingest_svc, preprocess_svc, chunking_svc, embedding_svc


# ── Golden path ────────────────────────────────────────────────────────────────

class TestGoldenPath:

    def test_success_result(self):
        pipeline, *_ = _make_pipeline()
        result = pipeline.run("/data/report.pdf")
        assert result.success is True

    def test_document_id_populated(self):
        pipeline, ingest_svc, *_ = _make_pipeline()
        result = pipeline.run("/data/report.pdf")
        assert result.document_id == "doc-1"

    def test_total_chunks_equals_embedded_count(self):
        pipeline, *_ = _make_pipeline()
        result = pipeline.run("/data/report.pdf")
        assert result.total_chunks == 1

    def test_file_path_preserved(self):
        pipeline, *_ = _make_pipeline()
        result = pipeline.run("/data/report.pdf")
        assert result.file_path == "/data/report.pdf"

    def test_four_stages_recorded(self):
        pipeline, *_ = _make_pipeline()
        result = pipeline.run("/data/report.pdf")
        assert len(result.stages) == 4

    def test_stage_names_in_order(self):
        pipeline, *_ = _make_pipeline()
        result = pipeline.run("/data/report.pdf")
        names = [s.stage for s in result.stages]
        assert names == ["ingestion", "preprocessing", "chunking", "embedding"]

    def test_all_stages_success(self):
        pipeline, *_ = _make_pipeline()
        result = pipeline.run("/data/report.pdf")
        for stage in result.stages:
            assert stage.success is True

    def test_stage_durations_non_negative(self):
        pipeline, *_ = _make_pipeline()
        result = pipeline.run("/data/report.pdf")
        for stage in result.stages:
            assert stage.duration_ms >= 0.0

    def test_total_duration_non_negative(self):
        pipeline, *_ = _make_pipeline()
        result = pipeline.run("/data/report.pdf")
        assert result.total_duration_ms >= 0.0

    def test_not_duplicate(self):
        pipeline, *_ = _make_pipeline()
        result = pipeline.run("/data/report.pdf")
        assert result.is_duplicate is False

    def test_failed_stage_none_on_success(self):
        pipeline, *_ = _make_pipeline()
        result = pipeline.run("/data/report.pdf")
        assert result.failed_stage is None
        assert result.error is None


# ── RBAC forwarding ────────────────────────────────────────────────────────────

class TestRBACForwarding:

    def test_tenant_id_forwarded(self):
        pipeline, ingest_svc, *_ = _make_pipeline()
        pipeline.run("/data/doc.pdf", tenant_id="tenant-x")
        call_kwargs = ingest_svc.ingest.call_args.kwargs
        assert call_kwargs["tenant_id"] == "tenant-x"

    def test_owner_id_forwarded(self):
        pipeline, ingest_svc, *_ = _make_pipeline()
        pipeline.run("/data/doc.pdf", owner_id="user-99")
        call_kwargs = ingest_svc.ingest.call_args.kwargs
        assert call_kwargs["owner_id"] == "user-99"

    def test_access_roles_forwarded(self):
        pipeline, ingest_svc, *_ = _make_pipeline()
        pipeline.run("/data/doc.pdf", access_roles=["hr", "legal"])
        call_kwargs = ingest_svc.ingest.call_args.kwargs
        assert call_kwargs["access_roles"] == ["hr", "legal"]

    def test_visibility_forwarded(self):
        pipeline, ingest_svc, *_ = _make_pipeline()
        pipeline.run("/data/doc.pdf", visibility="restricted")
        call_kwargs = ingest_svc.ingest.call_args.kwargs
        assert call_kwargs["visibility"] == "restricted"

    def test_defaults_sent_when_not_specified(self):
        pipeline, ingest_svc, *_ = _make_pipeline()
        pipeline.run("/data/doc.pdf")
        call_kwargs = ingest_svc.ingest.call_args.kwargs
        assert call_kwargs["tenant_id"] == "default"
        assert call_kwargs["owner_id"] == "system"
        assert call_kwargs["visibility"] == "public"


# ── Ingestion failure ──────────────────────────────────────────────────────────

class TestIngestionFailure:

    def test_success_false(self):
        failed_ingest = IngestionResult(success=False, error="File not found")
        pipeline, *_ = _make_pipeline(ingest_result=failed_ingest)
        result = pipeline.run("/bad/path.pdf")
        assert result.success is False

    def test_failed_stage_is_ingestion(self):
        failed_ingest = IngestionResult(success=False, error="File not found")
        pipeline, *_ = _make_pipeline(ingest_result=failed_ingest)
        result = pipeline.run("/bad/path.pdf")
        assert result.failed_stage == "ingestion"

    def test_error_propagated(self):
        failed_ingest = IngestionResult(success=False, error="File not found")
        pipeline, *_ = _make_pipeline(ingest_result=failed_ingest)
        result = pipeline.run("/bad/path.pdf")
        assert result.error == "File not found"

    def test_only_one_stage_recorded(self):
        failed_ingest = IngestionResult(success=False, error="File not found")
        pipeline, *_ = _make_pipeline(ingest_result=failed_ingest)
        result = pipeline.run("/bad/path.pdf")
        assert len(result.stages) == 1
        assert result.stages[0].stage == "ingestion"

    def test_downstream_services_not_called(self):
        failed_ingest = IngestionResult(success=False, error="File not found")
        pipeline, _, preprocess_svc, chunking_svc, embedding_svc = _make_pipeline(
            ingest_result=failed_ingest
        )
        pipeline.run("/bad/path.pdf")
        preprocess_svc.process.assert_not_called()
        chunking_svc.chunk.assert_not_called()
        embedding_svc.embed.assert_not_called()

    def test_document_id_none(self):
        failed_ingest = IngestionResult(success=False, error="File not found")
        pipeline, *_ = _make_pipeline(ingest_result=failed_ingest)
        result = pipeline.run("/bad/path.pdf")
        assert result.document_id is None


# ── Near-duplicate shortcircuit ────────────────────────────────────────────────

class TestNearDuplicate:

    def _dup_ingest(self):
        doc = _mock_document("doc-existing")
        return IngestionResult(
            success=True,
            document=doc,
            is_duplicate=True,
            duplicate_of="doc-original",
            similarity_score=0.93,
        )

    def test_success_true(self):
        pipeline, *_ = _make_pipeline(ingest_result=self._dup_ingest())
        result = pipeline.run("/data/dup.pdf")
        assert result.success is True

    def test_is_duplicate_true(self):
        pipeline, *_ = _make_pipeline(ingest_result=self._dup_ingest())
        result = pipeline.run("/data/dup.pdf")
        assert result.is_duplicate is True

    def test_duplicate_of_propagated(self):
        pipeline, *_ = _make_pipeline(ingest_result=self._dup_ingest())
        result = pipeline.run("/data/dup.pdf")
        assert result.duplicate_of == "doc-original"

    def test_similarity_score_propagated(self):
        pipeline, *_ = _make_pipeline(ingest_result=self._dup_ingest())
        result = pipeline.run("/data/dup.pdf")
        assert result.similarity_score == pytest.approx(0.93)

    def test_only_ingestion_stage_recorded(self):
        pipeline, *_ = _make_pipeline(ingest_result=self._dup_ingest())
        result = pipeline.run("/data/dup.pdf")
        assert len(result.stages) == 1

    def test_downstream_services_not_called(self):
        pipeline, _, preprocess_svc, chunking_svc, embedding_svc = _make_pipeline(
            ingest_result=self._dup_ingest()
        )
        pipeline.run("/data/dup.pdf")
        preprocess_svc.process.assert_not_called()
        chunking_svc.chunk.assert_not_called()
        embedding_svc.embed.assert_not_called()

    def test_total_chunks_zero(self):
        pipeline, *_ = _make_pipeline(ingest_result=self._dup_ingest())
        result = pipeline.run("/data/dup.pdf")
        assert result.total_chunks == 0


# ── Preprocessing failure ──────────────────────────────────────────────────────

class TestPreprocessingFailure:

    def test_success_false(self):
        pipeline, *_ = _make_pipeline(
            preprocess_result=PreprocessingResult(
                success=False, error="Preprocessing failed: bad encoding"
            )
        )
        result = pipeline.run("/data/doc.pdf")
        assert result.success is False

    def test_failed_stage_is_preprocessing(self):
        pipeline, *_ = _make_pipeline(
            preprocess_result=PreprocessingResult(
                success=False, error="Preprocessing failed: bad encoding"
            )
        )
        result = pipeline.run("/data/doc.pdf")
        assert result.failed_stage == "preprocessing"

    def test_two_stages_recorded(self):
        pipeline, *_ = _make_pipeline(
            preprocess_result=PreprocessingResult(
                success=False, error="err"
            )
        )
        result = pipeline.run("/data/doc.pdf")
        assert len(result.stages) == 2

    def test_document_id_populated(self):
        pipeline, *_ = _make_pipeline(
            preprocess_result=PreprocessingResult(success=False, error="err")
        )
        result = pipeline.run("/data/doc.pdf")
        assert result.document_id == "doc-1"

    def test_chunking_not_called(self):
        pipeline, _, _, chunking_svc, embedding_svc = _make_pipeline(
            preprocess_result=PreprocessingResult(success=False, error="err")
        )
        pipeline.run("/data/doc.pdf")
        chunking_svc.chunk.assert_not_called()
        embedding_svc.embed.assert_not_called()


# ── Chunking failure ───────────────────────────────────────────────────────────

class TestChunkingFailure:

    def test_success_false(self):
        pipeline, *_ = _make_pipeline(
            chunk_result=ChunkingResult(
                success=False, error="Token counter crashed"
            )
        )
        result = pipeline.run("/data/doc.pdf")
        assert result.success is False

    def test_failed_stage_is_chunking(self):
        pipeline, *_ = _make_pipeline(
            chunk_result=ChunkingResult(success=False, error="err")
        )
        result = pipeline.run("/data/doc.pdf")
        assert result.failed_stage == "chunking"

    def test_three_stages_recorded(self):
        pipeline, *_ = _make_pipeline(
            chunk_result=ChunkingResult(success=False, error="err")
        )
        result = pipeline.run("/data/doc.pdf")
        assert len(result.stages) == 3

    def test_embedding_not_called(self):
        pipeline, _, _, _, embedding_svc = _make_pipeline(
            chunk_result=ChunkingResult(success=False, error="err")
        )
        pipeline.run("/data/doc.pdf")
        embedding_svc.embed.assert_not_called()


# ── Embedding failure ──────────────────────────────────────────────────────────

class TestEmbeddingFailure:

    def test_success_false(self):
        pipeline, *_ = _make_pipeline(
            embed_result=EmbeddingResult(
                success=False, error="GPU OOM"
            )
        )
        result = pipeline.run("/data/doc.pdf")
        assert result.success is False

    def test_failed_stage_is_embedding(self):
        pipeline, *_ = _make_pipeline(
            embed_result=EmbeddingResult(success=False, error="GPU OOM")
        )
        result = pipeline.run("/data/doc.pdf")
        assert result.failed_stage == "embedding"

    def test_four_stages_recorded(self):
        pipeline, *_ = _make_pipeline(
            embed_result=EmbeddingResult(success=False, error="GPU OOM")
        )
        result = pipeline.run("/data/doc.pdf")
        assert len(result.stages) == 4

    def test_error_propagated(self):
        pipeline, *_ = _make_pipeline(
            embed_result=EmbeddingResult(success=False, error="GPU OOM")
        )
        result = pipeline.run("/data/doc.pdf")
        assert result.error == "GPU OOM"


# ── Stage stats forwarding ─────────────────────────────────────────────────────

class TestStageStats:

    def test_chunking_stats_in_stage_result(self):
        pipeline, *_ = _make_pipeline(
            chunk_result=ChunkingResult(
                success=True,
                chunks=[_mock_chunk()],
                stats={"total_chunks": 1, "avg_tokens": 42.5},
            )
        )
        result = pipeline.run("/data/doc.pdf")
        chunking_stage = next(s for s in result.stages if s.stage == "chunking")
        assert chunking_stage.stats["total_chunks"] == 1
        assert chunking_stage.stats["avg_tokens"] == pytest.approx(42.5)

    def test_embedding_stats_in_stage_result(self):
        pipeline, *_ = _make_pipeline(
            embed_result=EmbeddingResult(
                success=True,
                embedded_count=3,
                stats={"batch_count": 1},
            )
        )
        result = pipeline.run("/data/doc.pdf")
        embed_stage = next(s for s in result.stages if s.stage == "embedding")
        assert embed_stage.stats["batch_count"] == 1
