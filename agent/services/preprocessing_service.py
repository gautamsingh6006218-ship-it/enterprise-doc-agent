"""
services/preprocessing_service.py

What problem does this solve?
- Callers (API endpoints, Celery workers) need one entry point to clean,
  normalise, and extract metadata from a Document — without knowing about
  the pipeline stages or handling exceptions themselves.

Why a service layer over calling the pipeline directly?
- Same reason as IngestionService: this is the microservice boundary.
  When preprocessing becomes its own container, only this file's interface matters.
- PreprocessingResult (never raises) keeps async task queues safe.
- Dependency injection on PreprocessingPipeline keeps this fully testable.

How this connects to the broader pipeline:
  IngestionService  →  Document
  PreprocessingService  →  ProcessedDocument
  ChunkingService  →  list[DocumentChunk]
"""

from dataclasses import dataclass

from agent.processing.models import ProcessedDocument
from agent.processing.pipeline import PreprocessingPipeline
from agent.ingestion.models import Document


@dataclass
class PreprocessingResult:
    """
    What problem does this solve?
    - Exceptions don't propagate reliably across Celery/async worker boundaries.
      A result object gives the caller a uniform interface: check success,
      read processed_document or error. No try/except at every call site.

    Why this return type instead of ProcessedDocument directly?
    - ProcessedDocument would force every caller to catch multiple exception
      types (ValueError, runtime errors from langdetect, regex errors, etc.).
    - PreprocessingResult is a safe envelope — success is always inspectable.

    Fields:
    - success:             True = all pipeline stages completed.
    - processed_document:  Populated only when success=True.
    - error:               Human-readable failure reason. Only when success=False.
    """

    success: bool
    processed_document: ProcessedDocument | None = None
    error: str | None = None


class PreprocessingService:
    """
    What problem does this solve?
    - Single entry point for all document preprocessing.
      Every downstream consumer calls process() — not the pipeline directly.

    Why does this class exist?
    - Microservice boundary: when preprocessing is extracted into its own
      service, only this file's API is visible to callers.
    - Error handling centralised here: callers never handle pipeline exceptions.
    - Pipeline injected in constructor: tests pass a minimal pipeline,
      production gets the full default.

    Why accept pipeline as constructor arg?
    - Tests can inject a pipeline with mocked cleaners to test just the
      service layer logic (error handling, result wrapping).
    - Different deployment profiles (fast pipeline vs full pipeline) can
      inject the appropriate pipeline without subclassing.
    """

    def __init__(self, pipeline: PreprocessingPipeline | None = None) -> None:
        """
        Why optional pipeline?
        - None creates a default pipeline with all production stages.
        - Tests inject a minimal or mocked pipeline for isolation.
        """
        self._pipeline = pipeline or PreprocessingPipeline()

    def process(self, document: Document) -> PreprocessingResult:
        """
        What problem does this solve?
        - Runs all preprocessing stages on a Document and returns a
          PreprocessingResult that is always safe to inspect — never raises.

        Why are these inputs required?
        - document: The Document from IngestionService. Must be a Document
                    instance — the pipeline needs source_type, tenant_id,
                    title, and text fields.

        Why PreprocessingResult instead of ProcessedDocument?
        - Consistent with IngestionResult pattern: uniform result objects
          across all service boundaries. Every service in the pipeline
          returns a Result, never raises.

        Returns:
        - PreprocessingResult(success=True, processed_document=...) on success.
        - PreprocessingResult(success=False, error=...) on any failure.
        """
        try:
            processed = self._pipeline.run(document)
            return PreprocessingResult(success=True, processed_document=processed)
        except Exception as e:
            return PreprocessingResult(
                success=False,
                error=f"Preprocessing failed: {e}",
            )
