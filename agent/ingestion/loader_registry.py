"""
ingestion/loader_registry.py

What problem does this solve?
- Callers should not need to know which loader handles which file extension.
  Without a registry, every entry point (API, CLI, worker) would duplicate
  if/elif extension logic — a maintenance nightmare at enterprise scale.

Why a registry pattern instead of a simple dict or if/elif?
- Open/Closed Principle: adding a new format (e.g. PPTX) only requires
  creating a loader and calling registry.register(). Zero changes to existing code.
- Single source of truth: extension → loader mapping lives in one place only.
- Testability: inject a registry with only the loaders a test needs — fast,
  no side effects from unrelated loaders.

Why a module-level singleton (registry)?
- Loaders are stateless. One shared instance per process avoids the overhead
  of re-registering all loaders on every ingest request.
- Tests that need isolation create their own LoaderRegistry() instance.

Usage:
    from agent.ingestion.loader_registry import registry
    doc = registry.load_document("/path/to/file.docx", tenant_id="acme")
"""

from pathlib import Path

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.loaders.docx import DocxLoader
from agent.ingestion.loaders.html import HtmlLoader
from agent.ingestion.loaders.pdf import PdfLoader
from agent.ingestion.loaders.txt import TxtLoader
from agent.ingestion.models import Document


class LoaderRegistry:
    """
    What problem does this solve?
    - Maps file extensions to loader instances so callers never hard-code
      format-specific logic outside of loader classes.

    Why does this class exist?
    - Central router for the ingestion pipeline.
    - Supports runtime registration so new formats can be plugged in without
      restarting the service (useful for enterprise plugin architectures).
    """

    def __init__(self) -> None:
        # Internal map: lowercase extension → loader instance.
        # One loader instance per format — loaders are stateless so sharing is safe.
        self._loaders: dict[str, BaseDocumentLoader] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """
        What problem does this solve?
        - Ensures all built-in loaders are available immediately after
          instantiation without the caller having to register them manually.

        Why called from __init__ instead of class body?
        - Loader instances are created here, not at import time.
          This avoids import-order issues and makes testing easier
          (create a LoaderRegistry() with no side effects at import).
        """
        for loader in [PdfLoader(), DocxLoader(), TxtLoader(), HtmlLoader()]:
            for ext in loader.supported_extensions:
                self._loaders[ext] = loader

    def register(self, loader: BaseDocumentLoader) -> None:
        """
        What problem does this solve?
        - Allows adding new format support at runtime without changing any
          existing code (open/closed principle).

        Why does this method exist?
        - Enterprise systems often need custom loaders (e.g. proprietary formats,
          OCR-enabled PDF loader, SharePoint connector). register() is the
          plugin entry point for those cases.

        Why does it override existing mappings silently?
        - Intentional: allows replacing the default PdfLoader with an OCR-aware
          version in production without crashing on the duplicate key.

        Args:
        - loader: Any BaseDocumentLoader subclass instance.
        """
        for ext in loader.supported_extensions:
            self._loaders[ext] = loader

    def get_loader(self, file_path: str) -> BaseDocumentLoader:
        """
        What problem does this solve?
        - Resolves which loader to use for a given file without the caller
          inspecting extensions or knowing loader class names.

        Why return the loader instead of calling load() directly?
        - Separation of concerns: get_loader() is a routing decision.
          Calling load() is an IO operation. Keeping them separate allows
          callers to inspect the loader before committing to IO.

        Why raise ValueError instead of returning None?
        - Returning None shifts the None-check burden to every call site.
          A ValueError with a descriptive message (listing supported formats)
          is immediately actionable — the caller knows exactly what went wrong.

        Args:
        - file_path: Path to the file (only the suffix is inspected).

        Returns:
        - The registered BaseDocumentLoader for that extension.

        Raises:
        - ValueError if no loader is registered for the file's extension.
        """
        ext = Path(file_path).suffix.lower()
        loader = self._loaders.get(ext)

        if not loader:
            supported = sorted(self._loaders.keys())
            raise ValueError(
                f"No loader registered for '{ext}'. Supported: {supported}"
            )

        return loader

    def load_document(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        What problem does this solve?
        - Single-call convenience for the most common use case: resolve loader
          and load document in one step.

        Why does this method exist?
        - IngestionService only needs one call, not two (get_loader + load).
          This keeps the service layer clean and the registry self-contained.

        Why return Document instead of IngestionResult?
        - The registry is a low-level routing component. It does not own
          error-handling policy — that belongs in IngestionService which
          wraps this call and converts exceptions to IngestionResult.

        Args:
        - file_path:  Path to the source file.
        - tenant_id:  Passed through to the Document for tenant isolation.

        Returns:
        - A fully populated Document instance.
        """
        loader = self.get_loader(file_path)
        return loader.load(file_path, tenant_id=tenant_id)

    @property
    def supported_extensions(self) -> list[str]:
        """
        Why does this exist?
        - IngestionService exposes this to the API layer for file upload
          validation — the API needs to know accepted formats without
          coupling to the registry internals.

        Returns sorted list so API responses are deterministic.
        """
        return sorted(self._loaders.keys())


# Module-level singleton.
# Why: loaders are stateless — one shared instance per process is sufficient.
# Tests that need isolation create their own LoaderRegistry() instance directly.
registry = LoaderRegistry()
