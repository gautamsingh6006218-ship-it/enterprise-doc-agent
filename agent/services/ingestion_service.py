"""
services/ingestion_service.py

What problem does this solve?
- Callers need one place to hand a file path and get back a fully loaded,
  RBAC-tagged Document — without caring which loader to use or how errors
  are handled.
- Without this layer, every API endpoint or worker would duplicate loader
  selection logic and error handling.

Why does this file exist?
- Microservice boundary: this is the only public interface for ingestion.
  When ingestion becomes its own Docker container, only this file's API matters.
- Separates RBAC concerns from loaders. Loaders extract text only.
  IngestionService attaches owner, roles, and visibility after loading.
- Returns IngestionResult (never raises) so async task queues (Celery, RQ)
  never crash on a bad file — they log the error and move on.
"""

from dataclasses import dataclass

from agent.ingestion.loader_registry import LoaderRegistry
from agent.ingestion.models import Document


@dataclass
class IngestionResult:
    """
    What problem does this solve?
    - Exceptions don't propagate cleanly across async worker boundaries
      (Celery, Kafka consumers). A result object is safer than try/except
      at every call site.

    Why this return type instead of raising exceptions?
    - Callers (FastAPI endpoints, Celery tasks) always receive a result they
      can inspect — no unexpected crashes in the ingestion queue.
    - success=False + error message gives the API enough info to return a
      meaningful HTTP 422 without catching multiple exception types.

    Fields:
    - success:   True = document loaded and RBAC applied. False = failed.
    - document:  Populated only when success=True.
    - error:     Human-readable reason. Populated only when success=False.
    """

    success: bool
    document: Document | None = None
    error: str | None = None


class IngestionService:
    """
    What problem does this solve?
    - Orchestrates: file → correct loader → Document → RBAC fields attached.
    - Hides loader selection, RBAC attachment, and error handling from callers.

    Why does this class exist?
    - Single entry point for all ingestion. Every API route, CLI command, and
      worker calls IngestionService.ingest() — not the registry directly.
    - Dependency injection via constructor: tests inject a custom registry
      instead of using all built-in loaders. No patching needed.

    Why accept registry as a constructor arg?
    - Allows tests to pass a registry with only the loaders they need.
    - Allows production code to inject an OCR-enabled registry without
      changing this file.
    """

    def __init__(self, registry: LoaderRegistry | None = None) -> None:
        """
        Why optional registry?
        - Default (None) creates a registry with all built-in loaders —
          correct for production use.
        - Tests pass a minimal registry to keep test scope narrow and fast.
        """
        self._registry = registry or LoaderRegistry()

    def ingest(
        self,
        file_path: str,
        tenant_id: str = "default",
        owner_id: str = "system",
        access_roles: list[str] | None = None,
        visibility: str = "public",
    ) -> IngestionResult:
        """
        What problem does this solve?
        - Turns a raw file path into a Document with RBAC metadata attached,
          ready for ChunkingService to process next.

        Why are these inputs required?
        - file_path:     The source file. Required — nothing to load without it.
        - tenant_id:     Isolates this document to one tenant's data partition.
                         Required for multi-tenant vector store filtering.
        - owner_id:      Who is ingesting this file. Stored for audit trail and
                         "private" visibility enforcement.
        - access_roles:  Which roles can read this document. Empty list = unrestricted
                         within the tenant. Stored in chunk metadata for vector
                         store filtering at query time.
        - visibility:    Coarse access level.
                         "public"     → all tenant users.
                         "restricted" → only roles in access_roles.
                         "private"    → owner only.

        Why IngestionResult instead of Document?
        - Document would force callers to catch FileNotFoundError, ValueError,
          and any loader-specific exceptions separately.
        - IngestionResult gives a uniform interface: check success, read document
          or error. One pattern, no exception handling at call sites.

        Returns:
        - IngestionResult(success=True, document=Document) on success.
        - IngestionResult(success=False, error=<reason>) on any failure.
        """
        try:
            document = self._registry.load_document(file_path, tenant_id=tenant_id)

            # RBAC fields are attached here, not inside loaders.
            # Loaders are responsible for text extraction only — they have no
            # knowledge of who is calling or what roles should apply.
            document.owner_id = owner_id
            document.access_roles = access_roles or []
            document.visibility = visibility

            return IngestionResult(success=True, document=document)

        except (FileNotFoundError, ValueError) as e:
            # Expected failures: file missing or format not supported.
            return IngestionResult(success=False, error=str(e))

        except Exception as e:
            # Unexpected failures: corrupted file, permission denied, OOM, etc.
            # Caught here so a single bad file never crashes the ingestion queue.
            return IngestionResult(success=False, error=f"Unexpected error: {e}")

    @property
    def supported_formats(self) -> list[str]:
        """
        Why does this exist?
        - API layer needs to expose which formats are accepted (for file upload
          validation) without knowing about the registry internals.
        """
        return self._registry.supported_extensions
