"""
ingestion/loaders/unstructured_loader.py

What problem does this solve?
- Enterprise document collections include DOCX, PPTX, XLSX, emails, images,
  and more. Building a separate loader for each format is expensive to maintain.
  One library that handles all of them eliminates that burden.

Why Unstructured instead of format-specific libraries for each type?
- Single dependency covers 20+ formats with consistent output.
- Returns semantic elements (Title, NarrativeText, Table, ListItem) not just
  raw text — enabling smarter chunking strategies in Phase 2.
- Built-in OCR for images via Tesseract (when installed).
- Format-specific libraries (python-pptx, openpyxl) are still used internally
  by Unstructured, but we don't maintain the integration code.

Supported formats handled here:
- Office:  .docx, .pptx, .xlsx, .csv, .odt
- Email:   .eml, .msg
- Images:  .png, .jpg, .jpeg, .tiff, .bmp, .gif
- Docs:    .rtf, .epub

Formats NOT handled here (have dedicated loaders):
- .pdf   → SmartPdfLoader
- .txt, .md, .markdown → TxtLoader
- .html, .htm → HtmlLoader

Element type strategy:
- All element types are concatenated as plain text for now.
- Phase 2: Table elements will be kept as structured data for table-aware RAG.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document

try:
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False


# Formats this loader handles.
# Separated from the class so LoaderRegistry can inspect without instantiation.
UNSTRUCTURED_EXTENSIONS = [
    # Microsoft Office
    ".docx", ".pptx", ".xlsx", ".csv",
    # OpenDocument
    ".odt",
    # Email
    ".eml", ".msg",
    # Images (requires Tesseract)
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif",
    # Other document formats
    ".rtf", ".epub",
]


class UnstructuredLoader(BaseDocumentLoader):
    """
    What problem does this solve?
    - Handles all enterprise formats that don't have dedicated loaders,
      using Unstructured's auto-detection and element-based extraction.

    Why does this class exist?
    - One loader for 15+ formats beats maintaining 15 separate loaders.
    - Unstructured returns semantic element types (Title, Table, etc.) that
      will enable format-aware chunking in Phase 2.
    - Centralises the Unstructured dependency in one place.

    Why accept strategy as a constructor param?
    - "auto":  Unstructured picks the best strategy per format (default).
    - "fast":  Skip OCR and ML models — good for CI/test environments.
    - "hi_res": Maximum quality, uses layout detection ML models.
    - Allows deployment-specific tuning without code changes.
    """

    def __init__(self, strategy: str = "auto") -> None:
        """
        Why strategy parameter?
        - "fast" skips ML inference — good for text-heavy documents and tests.
        - "hi_res" enables layout models — better for image-heavy Office files.
        - "auto" balances both — correct default for unknown document collections.
        """
        self._strategy = strategy

    @property
    def supported_extensions(self) -> list[str]:
        return UNSTRUCTURED_EXTENSIONS

    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        What problem does this solve?
        - Converts any of 15+ file formats into a plain-text Document with
          semantic element metadata preserved for downstream processing.

        Why partition() instead of format-specific functions?
        - partition() auto-detects the format and calls the right backend
          (partition_docx, partition_pptx, etc.) internally.
        - We don't need to maintain format detection logic — Unstructured handles it.

        Why filter empty elements?
        - Office files contain many empty paragraphs and spacing elements.
          Including them creates whitespace-only chunks downstream.

        Why track element_types in metadata?
        - Tells the chunker what types of content are present.
          A document with Table elements needs different chunking than plain text.
        - Useful for observability: quickly see if a PPTX has tables extracted.

        Why source_type from extension instead of Unstructured's type?
        - Unstructured may return "docx" or "application/vnd.openxmlformats..."
          Our source_type field uses simple lowercase format names for consistency.

        Raises:
        - ImportError        if unstructured is not installed.
        - FileNotFoundError  if the file does not exist.
        - ValueError         if the extension is not supported.
        """
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError(
                "UnstructuredLoader requires unstructured. "
                "Run: pip install 'unstructured[docx,pptx,xlsx,csv]'"
            )

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"UnstructuredLoader cannot handle: {path.suffix}")

        raw_bytes = path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        # partition() auto-detects format and routes to the right backend.
        # strategy controls quality vs speed tradeoff.
        elements = partition(filename=str(path), strategy=self._strategy)

        # Collect non-empty element text. Double newline between elements
        # preserves natural paragraph/section breaks for the chunker.
        text_parts = [str(el).strip() for el in elements if str(el).strip()]
        full_text = "\n\n".join(text_parts)

        # element_types records what semantic types were found (Title, Table, etc.)
        # This enables format-aware chunking in Phase 2.
        element_types = list({type(el).__name__ for el in elements})

        # Derive a clean source_type from file extension
        source_type = path.suffix.lower().lstrip(".")

        return Document(
            id=str(uuid4()),
            source_type=source_type,
            source_path=str(path.resolve()),
            title=path.stem,
            text=full_text,
            tenant_id=tenant_id,
            file_hash=file_hash,
            metadata={
                "file_name": path.name,
                "file_size_bytes": len(raw_bytes),
                "element_count": len(elements),
                "element_types": sorted(element_types),
                "extraction_method": "unstructured",
                "extraction_strategy": self._strategy,
            },
        )
