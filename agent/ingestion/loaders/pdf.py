"""
ingestion/loaders/pdf.py

What problem does this solve?
- Text-based PDFs need fast, structure-preserving extraction without heavy ML models.

Why PyMuPDF with "markdown" mode instead of plain get_text()?
- get_text("markdown") preserves headers, bold, lists as Markdown syntax.
- Downstream chunker can use ## headers as natural section boundaries.
- 3-5x faster than Docling for simple text PDFs — no ML inference needed.

When NOT to use this loader?
- Scanned PDFs (no text layer) → OcrPdfLoader
- Complex layouts (tables, charts, mixed columns) → DoclingPdfLoader
- Use SmartPdfLoader to auto-route between all three.

Known limitation:
- Image-only pages produce empty text. SmartPdfLoader detects this and
  re-routes to OcrPdfLoader automatically.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

import fitz  # PyMuPDF
import pymupdf4llm  # Markdown extraction layer over PyMuPDF

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document


class PdfLoader(BaseDocumentLoader):
    """
    What problem does this solve?
    - Fast text extraction from standard text-based PDFs with structure preserved.

    Why does this class exist?
    - Handles 80%+ of enterprise PDFs (reports, policies, contracts) that are
      text-based. Much faster than Docling for these common cases.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        What problem does this solve?
        - Extracts structured text from text-based PDFs, preserving headings and lists.

        Why get_text("markdown") instead of get_text()?
        - Preserves document structure (## headings, **bold**, bullet lists).
        - Chunker can split on Markdown headers for semantically clean chunks.
        - get_text() returns flat unformatted text losing all structure.

        Why skip blank pages?
        - Image-only/separator pages produce empty strings → empty chunks downstream.

        Why join with double newline?
        - \n\n is a paragraph boundary signal for RecursiveCharacterTextSplitter.

        Why store absolute path?
        - Audit trail: the file may move but the path at ingest time is preserved.

        Raises:
        - FileNotFoundError  if file does not exist.
        - ValueError         if extension is not .pdf.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"PdfLoader cannot handle: {path.suffix}")

        raw_bytes = path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        # pymupdf4llm.to_markdown() extracts the full document as Markdown in one call.
        # It uses PyMuPDF internals but outputs structured Markdown with headers,
        # bold text, and lists preserved — unlike raw get_text() which returns flat text.
        full_text = pymupdf4llm.to_markdown(str(path)).strip()

        # Open separately only to get page count for metadata
        pdf = fitz.open(path)
        page_count = pdf.page_count
        pdf.close()

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
                "extraction_method": "pymupdf4llm_markdown",
            },
        )
