"""
ingestion/loaders/docx.py
-------------------------
Microsoft Word (.docx) document loader using python-docx.

What is extracted:
- Body paragraphs (headings, body text, list items) — all paragraph styles.

What is NOT extracted (intentional for this phase):
- Tables: extracted as raw text in Phase 2 (needs row/column aware chunking).
- Headers / Footers: typically contain page numbers and company names that
  add noise to embeddings without semantic value.
- Embedded images: OCR pipeline — Phase 2.
- Comments / tracked changes: not surfaced by python-docx by default.

Note on .doc (old binary format):
- .doc is the legacy Word 97-2003 format and requires `antiword` or
  `LibreOffice` CLI to convert. It is not supported here — the LoaderRegistry
  will return a clear "no loader registered" error if a .doc file is passed.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

from docx import Document as DocxDocument

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document


class DocxLoader(BaseDocumentLoader):
    """
    Loads .docx files and extracts paragraph text content.

    Empty paragraphs (used in Word as visual spacing) are filtered out to
    avoid injecting blank lines into the text corpus.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """Handles .docx only. Legacy .doc is not supported."""
        return [".docx"]

    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        Extract text from a Word document and return a Document.

        Args:
            file_path:  Path to the .docx file.
            tenant_id:  Tenant identifier for multi-tenant isolation.

        Returns:
            Document with paragraph text joined by double newlines.

        Raises:
            FileNotFoundError: If the file does not exist on disk.
            ValueError:        If the file is not a .docx.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"DocxLoader cannot handle: {path.suffix}")

        raw_bytes = path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        doc = DocxDocument(str(path))

        # Filter empty paragraphs — Word uses blank paragraphs for spacing,
        # not content. Including them would pollute chunks with whitespace.
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs).strip()

        return Document(
            id=str(uuid4()),
            source_type="docx",
            source_path=str(path.resolve()),
            title=path.stem,
            text=full_text,
            tenant_id=tenant_id,
            file_hash=file_hash,
            metadata={
                "paragraph_count": len(paragraphs),
                "file_name": path.name,
                "file_size_bytes": len(raw_bytes),
            },
        )
