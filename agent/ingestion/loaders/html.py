"""
ingestion/loaders/html.py
-------------------------
HTML document loader using lxml.

Why lxml over BeautifulSoup?
- lxml is already in requirements.txt (pulled in by other dependencies).
- lxml is ~3-5x faster than BeautifulSoup on large HTML files.
- The HTMLParser handles malformed/real-world HTML the same way browsers do.

Text extraction strategy:
1. Parse the raw HTML bytes with lxml's tolerant HTMLParser.
2. Strip <script> and <style> nodes in-place — they contain code/CSS, not
   document content, and would pollute embeddings.
3. Use itertext() to walk the remaining DOM and collect all text nodes.
4. Normalise whitespace by splitting on any whitespace and re-joining with
   single spaces — eliminates tabs, multiple spaces, and newlines from HTML
   indentation.

Title extraction:
- Reads the <title> tag if present; falls back to the filename stem.
  This gives a human-readable document title for display and metadata.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

from lxml import etree

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document


class HtmlLoader(BaseDocumentLoader):
    """
    Loads HTML files and extracts clean text content.

    Strips scripts, styles, and HTML structure — returns plain readable text
    suitable for chunking and embedding.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """Handles .html and .htm files."""
        return [".html", ".htm"]

    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        Parse an HTML file, strip noise nodes, and return a Document.

        Args:
            file_path:  Path to the .html / .htm file.
            tenant_id:  Tenant identifier for multi-tenant isolation.

        Returns:
            Document with clean text (no tags, scripts, or styles).
            Title is populated from the <title> tag if present.

        Raises:
            FileNotFoundError: If the file does not exist on disk.
            ValueError:        If the extension is not supported.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"HtmlLoader cannot handle: {path.suffix}")

        raw_bytes = path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        # HTMLParser is lenient — handles real-world malformed HTML without
        # raising exceptions (unlike the strict XML parser).
        parser = etree.HTMLParser()
        tree = etree.fromstring(raw_bytes, parser)

        # Remove <script> and <style> subtrees before text extraction.
        # These contain code/CSS that has zero semantic value for RAG and
        # would significantly degrade embedding quality if included.
        for tag in tree.iter("script", "style"):
            tag.getparent().remove(tag)

        # itertext() walks all remaining text nodes depth-first.
        # Splitting and re-joining collapses all whitespace variants
        # (tabs, multiple spaces, newlines from HTML indentation) into
        # single spaces, producing clean readable text.
        tokens = " ".join(tree.itertext()).split()
        full_text = " ".join(tokens).strip()

        # Prefer <title> tag over filename — HTML titles are usually more
        # descriptive (e.g. "Q3 2024 Sales Report" vs "q3_sales_report").
        title_nodes = tree.findall(".//title")
        title = (
            title_nodes[0].text.strip()
            if title_nodes and title_nodes[0].text
            else path.stem
        )

        return Document(
            id=str(uuid4()),
            source_type="html",
            source_path=str(path.resolve()),
            title=title,
            text=full_text,
            tenant_id=tenant_id,
            file_hash=file_hash,
            metadata={
                "file_name": path.name,
                "file_size_bytes": len(raw_bytes),
            },
        )
