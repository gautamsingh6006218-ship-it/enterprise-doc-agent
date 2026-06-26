"""
services/ingestion_service.py
------------------------------
IngestionService — the microservice boundary for document ingestion.

Responsibilities:
- Accept a file path and tenant identifier.
- Delegate format detection and loading to LoaderRegistry.
- Return a structured IngestionResult (never raises — safe for async queues).

Why a service layer over calling the registry directly?
- The service is the natural boundary when this code is split into a
  standalone microservice (Docker container, Celery worker, FastAPI app).
  The API layer calls IngestionService; the service calls the registry.
- Error handling is centralised here: the caller always receives an
  IngestionResult, never an unexpected exception.
- The registry can be injected at construction time, making the service
  fully testable with a mock/custom registry.

Future extensions (without changing this interface):
- Add async def ingest_async() for non-blocking ingestion.
- Add ingest_batch() for bulk file ingestion.
- Emit events (Kafka/SQS) after successful ingestion for downstream consumers.
- Add retry logic with exponential backoff for transient IO errors.
"""

from dataclasses import dataclass

from agent.ingestion.loader_registry import LoaderRegistry
from agent.ingestion.models import Document


@dataclass
class IngestionResult:
    """
    Outcome of a single document ingestion attempt.

    Using a result object instead of raising exceptions makes the service
    safe to use in async task queues (Celery, RQ) where exception propagation
    across worker boundaries is unreliable.

    Attributes:
        success:   True if the document was loaded successfully.
        document:  The loaded Document instance. None if success is False.
        error:     Human-readable error message. None if success is True.
    """

    success: bool
    document: Document | None = None
    error: str | None = None


class IngestionService:
    """
    Orchestrates document ingestion from file path to Document object.

    Acts as the single entry point for all ingestion operations. Downstream
    services (ChunkingService, EmbeddingService) consume the Document
    returned inside IngestionResult.

    Dependency injection:
        Pass a custom LoaderRegistry to override which loaders are used.
        This is used in tests to inject a registry with only the loaders
        needed for that test, without loading all built-in loaders.
    """

    def __init__(self, registry: LoaderRegistry | None = None) -> None:
        """
        Initialise with an optional custom registry.

        Args:
            registry: LoaderRegistry instance. Defaults to a new registry
                      with all built-in loaders registered.
        """
        # Use provided registry or fall back to a fresh one with all defaults.
        self._registry = registry or LoaderRegistry()

    def ingest(self, file_path: str, tenant_id: str = "default") -> IngestionResult:
        """
        Load a document from disk and return the result.

        Catches all exceptions internally and maps them to IngestionResult
        with success=False, so callers never need to handle exceptions.

        Args:
            file_path:  Absolute or relative path to the source file.
            tenant_id:  Tenant identifier for multi-tenant isolation.
                        Defaults to "default" for single-tenant deployments.

        Returns:
            IngestionResult with:
                - success=True, document=<Document> on success.
                - success=False, error=<message> on any failure.
        """
        try:
            document = self._registry.load_document(file_path, tenant_id=tenant_id)
            return IngestionResult(success=True, document=document)

        except (FileNotFoundError, ValueError) as e:
            # Known, expected errors — file missing or format unsupported.
            return IngestionResult(success=False, error=str(e))

        except Exception as e:
            # Catch-all for unexpected errors (corrupted file, permission
            # denied, etc.) — prevents a single bad file from crashing the
            # ingestion queue.
            return IngestionResult(success=False, error=f"Unexpected error: {e}")

    @property
    def supported_formats(self) -> list[str]:
        """
        List of file extensions the service can currently ingest.

        Reflects whatever loaders are registered in the underlying registry.
        """
        return self._registry.supported_extensions
