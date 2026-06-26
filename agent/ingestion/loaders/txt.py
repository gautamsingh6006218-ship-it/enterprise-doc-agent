"""
ingestion/loaders/txt.py
------------------------
Plain text and Markdown document loader.

Both formats are treated as raw UTF-8 text — no Markdown parsing or
rendering is performed. This is intentional:
- The chunker operates on plain text and uses newline patterns as split hints.
- Markdown syntax characters (##, **, >, etc.) are low-frequency noise that
  the embedding model handles gracefully without stripping.
- If structured Markdown parsing (headers as section boundaries) is needed,
  that will be a separate MarkdownAwareChunker, not a different loader.

Encoding strategy:
- Attempts UTF-8 decode first.
- Falls back to `errors="replace"` for files with non-UTF-8 bytes (e.g.
  Latin-1 encoded legacy docs) — produces a readable document with
  replacement characters rather than crashing the pipeline.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document


class TxtLoader(BaseDocumentLoader):
    """
    Loads plain text (.txt) and Markdown (.md, .markdown) files.

    The `source_type` field distinguishes between 'txt' and 'markdown'
    so downstream services can apply format-aware processing if needed.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """Handles .txt, .md, and .markdown files."""
        return [".txt", ".md", ".markdown"]

    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        Read a text or Markdown file and return a Document.

        Args:
            file_path:  Path to the .txt / .md / .markdown file.
            tenant_id:  Tenant identifier for multi-tenant isolation.

        Returns:
            Document with raw file content as text. source_type is set to
            'markdown' for .md/.markdown files, 'txt' otherwise.

        Raises:
            FileNotFoundError: If the file does not exist on disk.
            ValueError:        If the extension is not supported.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"TxtLoader cannot handle: {path.suffix}")

        raw_bytes = path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        # `errors="replace"` ensures legacy files with mixed encodings don't
        # crash the pipeline — replacement char is preferable to a hard failure.
        full_text = raw_bytes.decode("utf-8", errors="replace").strip()

        # Distinguish markdown from plain text so chunking strategies can
        # use header boundaries (## Section) as natural split points later.
        source_type = "markdown" if path.suffix.lower() in (".md", ".markdown") else "txt"

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
                "encoding": "utf-8",
            },
        )
