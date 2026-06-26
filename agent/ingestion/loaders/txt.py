"""
ingestion/loaders/txt.py

What problem does this solve?
- Plain text and Markdown files need no binary parsing, but the pipeline
  still needs a Document with tenant, hash, and metadata — not just raw text.

Why handle .txt and .md in the same loader?
- Both formats are UTF-8 text files with no binary structure to parse.
- Only the source_type field differs ("txt" vs "markdown") so downstream
  services can apply format-aware processing (e.g. header-based chunking).
- A separate MarkdownLoader would be pure duplication with one field changed.

Why not parse Markdown syntax (strip ##, **, etc.)?
- Embedding models handle Markdown characters gracefully — they are low-
  frequency tokens that don't degrade embedding quality.
- Stripping syntax would lose heading structure that the chunker can use
  as natural split boundaries (header-aware chunking — Phase 2).

Encoding strategy:
- UTF-8 with errors="replace": handles legacy files with mixed encodings.
- Produces readable text with replacement chars rather than crashing the pipeline.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document


class TxtLoader(BaseDocumentLoader):
    """
    What problem does this solve?
    - Reads plain text and Markdown files into a Document without any binary
      parsing overhead.

    Why does this class exist?
    - Centralises encoding handling and source_type detection for text-based
      formats. All other text formats (CSV, log files) can be added here.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """Handles .txt, .md, and .markdown. All treated as raw UTF-8 text."""
        return [".txt", ".md", ".markdown"]

    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        What problem does this solve?
        - Wraps a plain text file in a Document with all required pipeline fields.

        Why are these inputs required?
        - file_path:  The text file to read.
        - tenant_id:  Stored on Document for tenant-scoped vector filtering.

        Why read as bytes first instead of open(..., 'r')?
        - Bytes are needed for MD5 hashing. Reading as bytes once avoids two
          disk reads. Decoding is done in-memory after hashing.

        Why errors="replace" in decode?
        - Legacy enterprise files often have non-UTF-8 bytes (Latin-1, Windows
          CP1252). Raising an exception would silently drop the document from
          the index. Replacement characters are preferable to a lost document.

        Why set source_type to "markdown" for .md files?
        - Lets the ChunkingService apply header-aware splitting (\n## ) for
          Markdown in Phase 2, while plain text uses paragraph-based splitting.

        Why Document instead of str?
        - Downstream services need file_hash for deduplication, tenant_id for
          isolation, and source_path for audit — a bare string carries none of that.

        Raises:
        - FileNotFoundError  if the file does not exist.
        - ValueError         if the extension is not in supported_extensions.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"TxtLoader cannot handle: {path.suffix}")

        # Read bytes first — needed for MD5 before decoding to str.
        raw_bytes = path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        # errors="replace" keeps the pipeline alive for files with
        # non-UTF-8 bytes (legacy encodings, copy-paste artefacts).
        full_text = raw_bytes.decode("utf-8", errors="replace").strip()

        # Distinguishing markdown from plain text allows format-aware chunking
        # to use header lines (## Section) as natural split boundaries.
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
