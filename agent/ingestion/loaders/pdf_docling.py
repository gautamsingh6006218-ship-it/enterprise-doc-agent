"""
ingestion/loaders/pdf_docling.py

What problem does this solve?
- Complex enterprise PDFs (financial reports, technical specs, research papers)
  contain tables, multi-column layouts, charts, and mixed content that
  PyMuPDF's text extraction mangles or skips entirely.

Why Docling instead of other layout-aware tools?
- Fully open-source (MIT license), runs 100% locally — data never leaves infra.
- State-of-the-art layout detection (IBM Research, DocLayNet dataset).
- Exports clean Markdown that preserves table structure, headings, and lists.
- Handles scanned PDFs via built-in RapidOCR (no Tesseract dependency).
- LlamaParse is comparable but is a paid cloud API — not viable for enterprise.

Why export to Markdown instead of raw text?
- Markdown preserves semantic structure: ## headings, | tables |, **bold**.
- ChunkingService can split on header boundaries for semantically coherent chunks.
- Tables are not collapsed into garbled flat text — they remain readable.

Performance note:
- First call downloads AI models (~500MB). Subsequent calls are fast (cached).
- 2-10x slower than PyMuPDF on text-only PDFs — use PdfLoader for those.
- SmartPdfLoader only routes here when text density is sparse.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


class DoclingPdfLoader(BaseDocumentLoader):
    """
    What problem does this solve?
    - Extracts high-quality structured text from complex PDFs using AI-based
      layout detection and table parsing.

    Why does this class exist?
    - PdfLoader handles simple text PDFs. DoclingPdfLoader handles everything
      else: tables, charts, multi-column, mixed text+image.
    - Keeping them separate means PdfLoader stays fast and Docling is only
      invoked when needed.

    Why lazy-initialize DocumentConverter?
    - DocumentConverter downloads AI models on first use (~500MB).
    - Lazy init means import does not trigger a download — only loading does.
    - Multiple files reuse the same converter instance (cached as class var).
    """

    # Class-level cache: converter is expensive to create (loads AI models).
    # One instance shared across all DoclingPdfLoader instances in the process.
    _converter: "DocumentConverter | None" = None

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def _get_converter(self) -> "DocumentConverter":
        """
        Why lazy init and class-level cache?
        - Avoids loading 500MB+ of AI models at import time.
        - One converter per process is sufficient — it is stateless for conversion.
        """
        if DoclingPdfLoader._converter is None:
            DoclingPdfLoader._converter = DocumentConverter()
        return DoclingPdfLoader._converter

    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        What problem does this solve?
        - Extracts structured text from complex PDFs that PyMuPDF cannot handle.

        Why export to Markdown?
        - Markdown is the best format for downstream chunking: headers are
          natural split points, tables are readable, lists are preserved.
        - JSON export contains more detail but is harder to chunk cleanly.

        Why store page_count from Docling result?
        - Docling's page count is accurate for complex layouts. PyMuPDF may
          report different counts for documents with appendices or blank pages.

        Raises:
        - ImportError        if docling is not installed.
        - FileNotFoundError  if the PDF does not exist.
        - ValueError         if the file is not a .pdf.
        """
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "DoclingPdfLoader requires docling. Run: pip install docling"
            )

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"DoclingPdfLoader cannot handle: {path.suffix}")

        raw_bytes = path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        converter = self._get_converter()
        result = converter.convert(str(path))

        # Markdown export preserves tables, headings, and lists.
        # This is far better than flat text for RAG chunking quality.
        full_text = result.document.export_to_markdown()

        page_count = len(result.document.pages) if result.document.pages else 0

        return Document(
            id=str(uuid4()),
            source_type="pdf",
            source_path=str(path.resolve()),
            title=path.stem,
            text=full_text,
            tenant_id=tenant_id,
            file_hash=file_hash,
            metadata={
                "page_count": page_count,
                "file_name": path.name,
                "file_size_bytes": len(raw_bytes),
                "extraction_method": "docling",
            },
        )
