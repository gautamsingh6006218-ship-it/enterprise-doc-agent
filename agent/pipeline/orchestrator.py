"""
pipeline/orchestrator.py

What problem does this solve?
- The 4 services (Ingestion → Preprocessing → Chunking → Embedding) exist
  independently but nothing chains them. Every caller would have to sequence
  the same 4 calls, handle partial failures, and collect stage timing.
  That logic would be duplicated in every API endpoint, CLI command, and
  Celery worker.

Why a DocumentPipeline class instead of a module-level function?
- All 4 services are injected via constructor — the pipeline is fully testable
  with mocked services without touching files or databases.
- A class can be extended (add a 5th stage like PII redaction) without
  changing callers — they still call run().

Why measure duration at every stage?
- Stage timing answers: "Is preprocessing slow for scanned PDFs?"
  "Is embedding the bottleneck for large XLSX files?"
  Without per-stage timings, optimisation is guesswork.

Why return PipelineResult immediately on duplicate (not continue)?
- A near-duplicate document already has all its chunks in PgVector.
  Running preprocessing → chunking → embedding again would store
  duplicate chunks and inflate retrieval results with near-identical text.
  Early return on duplicate saves significant CPU and DB writes.

Full stage order:
  1. IngestionService    file → Document (MIME detect, load, RBAC, near-dedup)
  2. PreprocessingService Document → ProcessedDocument (ftfy, clean, normalise, extract)
  3. ChunkingService     ProcessedDocument → list[DocumentChunk] (token-aware splitting)
  4. EmbeddingService    list[DocumentChunk] → vectors stored in PgVector
"""

import time

from agent.pipeline.models import PipelineResult, StageResult
from agent.services.chunking_service import ChunkingService
from agent.services.embedding_service import EmbeddingService
from agent.services.ingestion_service import IngestionService
from agent.services.preprocessing_service import PreprocessingService


def _ms(start: float) -> float:
    """Wall-clock milliseconds since start (perf_counter)."""
    return round((time.perf_counter() - start) * 1000, 1)


class DocumentPipeline:
    """
    What problem does this solve?
    - Single entry point that runs a file through all 4 pipeline stages
      and returns a PipelineResult with per-stage stats.

    Why inject all 4 services instead of creating them internally?
    - Tests inject mocked services — no file I/O, no DB, no GPU.
    - Production wires real services once at startup and reuses them.
    - A/B testing: swap just the embedding service without changing the pipeline.

    Why not subclass or wrap individual services?
    - Composition over inheritance. The pipeline orchestrates — it does not
      need to know the implementation details of any service.
    """

    def __init__(
        self,
        ingestion_service: IngestionService,
        preprocessing_service: PreprocessingService,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
    ) -> None:
        self._ingestion = ingestion_service
        self._preprocessing = preprocessing_service
        self._chunking = chunking_service
        self._embedding = embedding_service

    def run(
        self,
        file_path: str,
        tenant_id: str = "default",
        owner_id: str = "system",
        access_roles: list[str] | None = None,
        visibility: str = "public",
        original_filename: str = "",
    ) -> PipelineResult:
        """
        What problem does this solve?
        - Runs file_path through the complete ingestion pipeline end-to-end.
          Returns a PipelineResult describing success, stage durations, and stats.

        Why return PipelineResult instead of raising on failure?
        - Consistent with all service layers: never raises, always returns
          an inspectable result. Callers (API routes, workers) check success.

        Why stop at the first failed stage?
        - Subsequent stages cannot run without valid input from prior stages.
          Chunking needs a ProcessedDocument; embedding needs chunks.
          Continuing past a failure would produce garbage or crash.

        Args:
        - file_path:     Absolute path to the file to ingest.
        - tenant_id:     Multi-tenancy isolation key.
        - owner_id:      Who is ingesting (for RBAC and audit).
        - access_roles:  Roles that can access this document.
        - visibility:    "public" | "restricted" | "private".

        Returns PipelineResult with success, document_id, chunk count, timings.
        """
        pipeline_start = time.perf_counter()
        stages: list[StageResult] = []

        # ── Stage 1: Ingestion ─────────────────────────────────────────────
        t = time.perf_counter()
        ingest_result = self._ingestion.ingest(
            file_path,
            tenant_id=tenant_id,
            owner_id=owner_id,
            access_roles=access_roles,
            visibility=visibility,
            original_filename=original_filename,
        )
        stages.append(StageResult(
            stage="ingestion",
            success=ingest_result.success,
            duration_ms=_ms(t),
            error=ingest_result.error,
        ))

        if not ingest_result.success:
            return PipelineResult(
                success=False,
                file_path=file_path,
                failed_stage="ingestion",
                error=ingest_result.error,
                total_duration_ms=_ms(pipeline_start),
                stages=stages,
            )

        # Near-duplicate: skip pipeline, return early
        if ingest_result.is_duplicate:
            return PipelineResult(
                success=True,
                file_path=file_path,
                document_id=ingest_result.document.id,
                is_duplicate=True,
                duplicate_of=ingest_result.duplicate_of,
                similarity_score=ingest_result.similarity_score,
                total_duration_ms=_ms(pipeline_start),
                stages=stages,
            )

        document = ingest_result.document

        # ── Stage 2: Preprocessing ─────────────────────────────────────────
        t = time.perf_counter()
        preprocess_result = self._preprocessing.process(document)
        stages.append(StageResult(
            stage="preprocessing",
            success=preprocess_result.success,
            duration_ms=_ms(t),
            error=preprocess_result.error,
        ))

        if not preprocess_result.success:
            return PipelineResult(
                success=False,
                file_path=file_path,
                document_id=document.id,
                failed_stage="preprocessing",
                error=preprocess_result.error,
                total_duration_ms=_ms(pipeline_start),
                stages=stages,
            )

        processed_doc = preprocess_result.processed_document

        # ── Stage 3: Chunking ──────────────────────────────────────────────
        t = time.perf_counter()
        chunk_result = self._chunking.chunk(processed_doc)
        stages.append(StageResult(
            stage="chunking",
            success=chunk_result.success,
            duration_ms=_ms(t),
            error=chunk_result.error,
            stats=chunk_result.stats,
        ))

        if not chunk_result.success:
            return PipelineResult(
                success=False,
                file_path=file_path,
                document_id=document.id,
                failed_stage="chunking",
                error=chunk_result.error,
                total_duration_ms=_ms(pipeline_start),
                stages=stages,
            )

        # ── Stage 4: Embedding ─────────────────────────────────────────────
        t = time.perf_counter()
        embed_result = self._embedding.embed(chunk_result.chunks)
        stages.append(StageResult(
            stage="embedding",
            success=embed_result.success,
            duration_ms=_ms(t),
            error=embed_result.error,
            stats=embed_result.stats,
        ))

        if not embed_result.success:
            return PipelineResult(
                success=False,
                file_path=file_path,
                document_id=document.id,
                failed_stage="embedding",
                error=embed_result.error,
                total_duration_ms=_ms(pipeline_start),
                stages=stages,
            )

        return PipelineResult(
            success=True,
            file_path=file_path,
            document_id=document.id,
            total_chunks=embed_result.embedded_count,
            total_duration_ms=_ms(pipeline_start),
            stages=stages,
        )
