"""
ingestion/loaders/pdf.py

What problem does this solve?
- PDF is the most common enterprise document format. Raw PDF bytes are binary
  and unreadable — this loader extracts clean plain text from them.

Why PyMuPDF (fitz) instead of pypdf or pdfplumber?
- PyMuPDF runs at C level — 3-5x faster on large files than pure-Python libs.
- Handles complex layouts (multi-column, rotated text, mixed fonts) better.
- pypdf is installed as a fallback but not used as the primary extractor.

Known limitations (intentional for this phase):
- Scanned / image-only PDFs produce empty text — OCR (Tesseract or AWS
  Textract) will be added as a separate loader in Phase 2.
- No per-page bounding boxes or font metadata — plain text extraction only.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

import fitz  # PyMuPDF

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document


class PdfLoader(BaseDocumentLoader):
    """
    What problem does this solve?
    - Converts a .pdf file into a plain-text Document with page-level metadata.

    Why does this class exist?
    - Encapsulates all PyMuPDF-specific logic in one place. If the library
      changes or we switch to a cloud OCR service, only this file changes.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """Handles .pdf only. Image-based PDFs require the OCR loader (Phase 2)."""
        return [".pdf"]

    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        What problem does this solve?
        - Turns binary PDF bytes into indexed, searchable plain text.

        Why are these inputs required?
        - file_path:  Location of the PDF on disk. Required — no file, no text.
        - tenant_id:  Stored on the returned Document for vector store
                      tenant-scoped filtering at query time.

        Why read raw bytes before opening with fitz?
        - We need the bytes for MD5 hashing (deduplication). Reading once
          avoids two separate disk reads for the same file.

        Why skip blank pages?
        - Image-only or separator pages produce empty strings. Including them
          creates empty chunks downstream, wasting vector store capacity and
          polluting search results.

        Why join pages with double newline?
        - RecursiveCharacterTextSplitter treats \n\n as a paragraph boundary —
          a natural split point. Single \n would merge page content into one block.

        Why Document instead of str?
        - Downstream services (chunker, embedder) need source path, tenant,
          hash, and page count alongside the text. A bare string loses all of that.

        Raises:
        - FileNotFoundError  if the PDF does not exist on disk.
        - ValueError         if the file is not a .pdf.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"PdfLoader cannot handle: {path.suffix}")

        # Read bytes once — reused for MD5 hash and fitz.open()
        raw_bytes = path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        pdf = fitz.open(path)
        pages_text: list[str] = []

        for page in pdf:
            text = page.get_text()
            # Skip blank or image-only pages — they produce no usable text
            # and would create empty chunks if passed to the chunker.
            if text.strip():
                pages_text.append(text)

        pdf.close()

        full_text = "\n\n".join(pages_text).strip()

        return Document(
            id=str(uuid4()),
            source_type="pdf",
            source_path=str(path.resolve()),    # always store absolute path for audit trail
            title=path.stem,
            text=full_text,
            tenant_id=tenant_id,
            file_hash=file_hash,
            metadata={
                "page_count": len(pages_text),  # non-blank pages only
                "file_name": path.name,
                "file_size_bytes": len(raw_bytes),
            },
        )
