"""
ingestion/loader_registry.py
-----------------------------
Central registry that maps file extensions to their document loaders.

Role in the architecture:
    IngestionService --> LoaderRegistry --> correct BaseDocumentLoader
                                                   --> Document

Why a registry pattern?
- Open/Closed Principle: adding a new format (e.g. PPTX) only requires
  creating a new loader class and calling registry.register(). No existing
  code needs to change.
- Single source of truth: the caller never needs to know which loader handles
  which extension — that knowledge lives here alone.
- Testability: the registry can be instantiated with only the loaders needed
  for a test, keeping tests fast and isolated.

Module-level singleton:
    A shared `registry` instance is exported so services can import it
    directly without creating a new instance on every request.

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
    Maps file extensions to document loader instances.

    On initialisation, all built-in loaders (PDF, DOCX, TXT, HTML) are
    registered automatically. Custom loaders can be added at runtime via
    `register()` — useful for plugin-style extensibility or testing.
    """

    def __init__(self) -> None:
        # Internal map: lowercase extension (e.g. '.pdf') → loader instance.
        # One loader instance is shared across all files of the same type
        # because loaders are stateless.
        self._loaders: dict[str, BaseDocumentLoader] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all built-in loaders. Called once at init."""
        for loader in [PdfLoader(), DocxLoader(), TxtLoader(), HtmlLoader()]:
            for ext in loader.supported_extensions:
                self._loaders[ext] = loader

    def register(self, loader: BaseDocumentLoader) -> None:
        """
        Register a custom loader for one or more file extensions.

        If an extension is already mapped, the new loader replaces the
        existing one. This allows overriding default loaders (e.g. swap
        PdfLoader for an OCR-aware version in production).

        Args:
            loader: An instance of a BaseDocumentLoader subclass.
        """
        for ext in loader.supported_extensions:
            self._loaders[ext] = loader

    def get_loader(self, file_path: str) -> BaseDocumentLoader:
        """
        Return the loader responsible for the given file's extension.

        Args:
            file_path: Path to the file (only the suffix is inspected).

        Returns:
            The registered BaseDocumentLoader for that extension.

        Raises:
            ValueError: If no loader is registered for the file's extension.
                        The error message lists all currently supported formats
                        to help the caller diagnose the issue.
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
        Convenience method: resolve the loader and load the document in one call.

        This is the primary entry point used by IngestionService.

        Args:
            file_path:  Path to the source file.
            tenant_id:  Tenant identifier passed through to the Document.

        Returns:
            A fully populated Document instance.
        """
        loader = self.get_loader(file_path)
        return loader.load(file_path, tenant_id=tenant_id)

    @property
    def supported_extensions(self) -> list[str]:
        """Sorted list of all currently registered file extensions."""
        return sorted(self._loaders.keys())


# Module-level singleton — import this directly instead of instantiating a new
# registry on every request. Loaders are stateless so sharing is safe.
registry = LoaderRegistry()
