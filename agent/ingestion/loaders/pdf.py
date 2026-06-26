"""
ingestion/loaders/pdf.py
------------------------
PDF document loader using PyMuPDF (fitz).

Why PyMuPDF over pypdf or pdfplumber?
- PyMuPDF is significantly faster on large files (C-level rendering engine).
- Better text extraction from complex layouts (columns, tables, rotated text).
- pypdf is also installed (listed in requirements.txt) but used as fallback
  only — PyMuPDF is the primary extractor here.

Current limitations (known, intentional for this phase):
- Image-only / scanned PDFs will produce empty text. OCR support (via
  Tesseract or AWS Textract) will be added as a separate loader in Phase 2.
- No per-page metadata (bounding boxes, fonts) — plain text only. Structured
  extraction will be a separate pipeline stage.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

import fitz  # PyMuPDF

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document


class PdfLoader(BaseDocumentLoader):
    """
    Loads PDF files and extracts their text content page by page.

    Text from all pages is joined with double newlines to preserve paragraph
    boundaries. Empty pages (e.g. cover images, blank separators) are skipped
    to avoid injecting whitespace-only chunks downstream.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """Handles only .pdf files."""
        return [".pdf"]

    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        Extract text from a PDF file and return a Document.

        Args:
            file_path:  Path to the .pdf file.
            tenant_id:  Tenant identifier for multi-tenant isolation.

        Returns:
            Document with full extracted text and page-level metadata.

        Raises:
            FileNotFoundError: If the file does not exist on disk.
            ValueError:        If the file is not a .pdf.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"PdfLoader cannot handle: {path.suffix}")

        # Read raw bytes once — used for both MD5 hashing and fitz.open()
        # to avoid reading the file twice from disk.
        raw_bytes = path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        pdf = fitz.open(path)
        pages_text: list[str] = []

        for page in pdf:
            text = page.get_text()
            # Skip blank/image-only pages — they produce no usable text and
            # would create empty chunks if passed to the chunker.
            if text.strip():
                pages_text.append(text)

        pdf.close()

        # Double newline separator preserves page boundaries as paragraph
        # breaks, which RecursiveCharacterTextSplitter recognises as
        # natural split points.
        full_text = "\n\n".join(pages_text).strip()

        return Document(
            id=str(uuid4()),
            source_type="pdf",
            source_path=str(path.resolve()),  # always store absolute path
            title=path.stem,
            text=full_text,
            tenant_id=tenant_id,
            file_hash=file_hash,
            metadata={
                "page_count": len(pages_text),     # non-blank pages only
                "file_name": path.name,
                "file_size_bytes": len(raw_bytes),
            },
        )
