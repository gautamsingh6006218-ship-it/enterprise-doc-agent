"""
ingestion/loader_registry.py

What problem does this solve?
- A document pipeline that handles 20+ formats needs a single routing layer.
  Without it, every entry point (API, CLI, worker) duplicates format detection.

Why this registry design?
- Open/Closed Principle: adding a new format only requires creating a new
  loader and calling registry.register(). Zero changes to existing code.
- Single source of truth: extension → loader mapping lives here only.
- Testability: inject a minimal registry in tests — only load what you need.

Loader assignment rationale:
- .pdf                      → SmartPdfLoader (auto-routes to PyMuPDF/Docling/OCR)
- .docx .pptx .xlsx .csv
  .odt .eml .msg
  .png .jpg .jpeg .tiff
  .bmp .gif .rtf .epub      → UnstructuredLoader (one library for all formats)
- .txt .md .markdown        → TxtLoader (stdlib only, no extra deps)
- .html .htm                → HtmlLoader (lxml, already installed)

Why keep TxtLoader and HtmlLoader instead of routing through Unstructured?
- Both are stdlib-only (no extra packages). Faster and more predictable.
- HtmlLoader's script/style stripping is more aggressive than Unstructured's.
- TxtLoader's source_type distinction (txt vs markdown) is needed downstream.

Module-level singleton (registry):
- Loaders are stateless — one shared instance per process is safe and efficient.
- Tests that need isolation instantiate LoaderRegistry() directly.
"""

from pathlib import Path

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.loaders.html import HtmlLoader
from agent.ingestion.loaders.pdf_smart import SmartPdfLoader
from agent.ingestion.loaders.txt import TxtLoader
from agent.ingestion.loaders.unstructured_loader import UNSTRUCTURED_EXTENSIONS, UnstructuredLoader
from agent.ingestion.models import Document


class LoaderRegistry:
    """
    What problem does this solve?
    - Routes any file to the right loader based on its extension.
    - Supports runtime registration of custom loaders for enterprise plugins.

    Why does this class exist?
    - Callers never need to know which loader handles which format.
    - New formats are added by registering a loader — nothing else changes.
    """

    def __init__(self) -> None:
        # Internal map: lowercase extension → loader instance.
        # Loaders are stateless so one instance per format is safe.
        self._loaders: dict[str, BaseDocumentLoader] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """
        What problem does this solve?
        - Ensures all production loaders are registered on instantiation.
          Callers get a fully functional registry with no manual setup.

        Why instantiate loaders here and not at import time?
        - Avoids loading ML models (Docling) or registering defaults when
          tests only need a subset. Instantiation is deferred to first use.
        """
        # PDF: SmartPdfLoader auto-routes between PyMuPDF / Docling / OCR
        smart_pdf = SmartPdfLoader()
        for ext in smart_pdf.supported_extensions:
            self._loaders[ext] = smart_pdf

        # Office, email, images, and other formats via Unstructured
        unstructured = UnstructuredLoader()
        for ext in UNSTRUCTURED_EXTENSIONS:
            self._loaders[ext] = unstructured

        # Plain text and Markdown — stdlib only, fast
        txt = TxtLoader()
        for ext in txt.supported_extensions:
            self._loaders[ext] = txt

        # HTML — lxml based, strips scripts/styles aggressively
        html = HtmlLoader()
        for ext in html.supported_extensions:
            self._loaders[ext] = html

    def register(self, loader: BaseDocumentLoader) -> None:
        """
        What problem does this solve?
        - Allows plugging in custom loaders (e.g. SharePoint connector,
          OCR-only PDF loader) without changing registry source code.

        Why silently override existing mappings?
        - Intentional: production deployments replace default loaders with
          optimised versions (e.g. swap PdfLoader with OCR-enabled variant).
          Crashing on duplicate extension would block valid overrides.

        Args:
        - loader: Any BaseDocumentLoader subclass instance.
        """
        for ext in loader.supported_extensions:
            self._loaders[ext] = loader

    def get_loader(self, file_path: str) -> BaseDocumentLoader:
        """
        What problem does this solve?
        - Resolves which loader handles a file without the caller needing
          to inspect extensions or know loader class names.

        Why return the loader instead of calling load() directly?
        - Separation: routing (this method) and IO (load()) are separate concerns.
          Callers may want to inspect the loader type before committing to IO.

        Why raise ValueError with the supported list?
        - Returning None shifts None-checks to every call site.
        - A ValueError with supported formats is immediately actionable.

        Raises:
        - ValueError if no loader is registered for the extension.
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
        - Convenience: resolve loader and load document in one call.
          This is what IngestionService calls internally.

        Why return Document and not IngestionResult?
        - Registry is a low-level routing component. Error-handling policy
          (wrapping exceptions into IngestionResult) belongs in IngestionService.
        """
        loader = self.get_loader(file_path)
        return loader.load(file_path, tenant_id=tenant_id)

    @property
    def supported_extensions(self) -> list[str]:
        """
        Why does this exist?
        - API layer exposes accepted formats for file upload validation.
          Listing them here means the API never needs to know about loaders.
        """
        return sorted(self._loaders.keys())


# Module-level singleton. Loaders are stateless — one instance per process.
# Tests that need isolation create their own LoaderRegistry() instance.
registry = LoaderRegistry()
