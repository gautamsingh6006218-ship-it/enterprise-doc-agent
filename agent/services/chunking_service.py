"""
services/chunking_service.py

What problem does this solve?
- API endpoints and worker tasks need a single entry point to chunk a
  ProcessedDocument into DocumentChunks without knowing about pipeline
  internals or handling exceptions themselves.

Why a service layer over calling the pipeline directly?
- Microservice boundary: when chunking becomes its own container, only
  this file's interface is exposed to callers.
- ChunkingResult (never raises) keeps Celery workers and async tasks safe.
- Dependency injection on ChunkingPipeline keeps this fully testable.

How this connects to the broader pipeline:
  IngestionService      →  Document
  PreprocessingService  →  ProcessedDocument
  ChunkingService       →  list[DocumentChunk]
  EmbeddingService      →  (chunks embedded into PgVector)
"""

from dataclasses import dataclass, field

from agent.chunking.pipeline import ChunkingPipeline
from agent.ingestion.models import DocumentChunk
from agent.processing.models import ProcessedDocument


@dataclass
class ChunkingResult:
    """
    What problem does this solve?
    - Exceptions from chunking (malformed text, tokeniser failures) don't
      propagate reliably across async task boundaries. A result object
      gives every caller a uniform interface: check success, read chunks
      or error. No try/except at every call site.

    Why include stats?
    - Monitoring: track avg_tokens_per_chunk and max_chunk_tokens across
      a batch ingestion to detect documents that produce oversized chunks.
      stats are optional (only populated on success).

    Fields:
    - success: True = all chunks produced successfully.
    - chunks:  List of DocumentChunk objects, ready for EmbeddingService.
               None on failure.
    - error:   Human-readable failure reason. Only when success=False.
    - stats:   Pipeline metrics dict (total_chunks, avg tokens, etc.).
               Only when success=True.
    """

    success: bool
    chunks: list[DocumentChunk] | None = None
    error: str | None = None
    stats: dict | None = field(default=None)


class ChunkingService:
    """
    What problem does this solve?
    - Single entry point for splitting a ProcessedDocument into
      DocumentChunks with full metadata. Every downstream consumer
      (EmbeddingService, API) calls chunk() — not the pipeline directly.

    Why does this class exist?
    - Microservice boundary: same reason as IngestionService and
      PreprocessingService. Each service wraps its pipeline in a uniform
      Result pattern. ChunkingService is the third link in the chain.
    - Error handling centralised here: callers never handle pipeline exceptions.
    - Pipeline injected in constructor: tests pass a stub pipeline, production
      gets the full default.

    Why accept pipeline as constructor arg?
    - Isolates service-layer tests (result wrapping, error handling) from
      pipeline tests (strategy selection, chunk building).
    - Deployment profiles with different routers (e.g. sentence-only for
      cost reduction) can inject a pre-configured pipeline.
    """

    def __init__(self, pipeline: ChunkingPipeline | None = None) -> None:
        """
        Why optional pipeline?
        - None creates a default pipeline with all three strategies.
        - Tests inject a minimal or mocked pipeline for isolation.
        """
        self._pipeline = pipeline or ChunkingPipeline()

    def chunk(self, processed_document: ProcessedDocument) -> ChunkingResult:
        """
        What problem does this solve?
        - Runs the full chunking pipeline on a ProcessedDocument and returns
          a ChunkingResult that is always safe to inspect — never raises.

        Why ProcessedDocument and not Document?
        - Chunking requires cleaned_text (not raw text) and extracted
          metadata (language, category) for chunk metadata.
          ProcessedDocument carries both. Using raw Document would require
          the service to call the preprocessor, violating single responsibility.

        Why ChunkingResult instead of list[DocumentChunk]?
        - Consistent with IngestionResult and PreprocessingResult: every
          service in the pipeline returns a Result, never raises.
          Callers check success before accessing chunks — no try/except.

        Args:
        - processed_document: ProcessedDocument from PreprocessingService.
                              Must have cleaned_text and extracted_metadata.

        Returns:
        - ChunkingResult(success=True, chunks=[...], stats={...}) on success.
        - ChunkingResult(success=False, error="...") on any failure.
        """
        try:
            chunks, stats = self._pipeline.run(processed_document)
            return ChunkingResult(success=True, chunks=chunks, stats=stats)
        except Exception as e:
            return ChunkingResult(
                success=False,
                error=f"Chunking failed: {e}",
            )
