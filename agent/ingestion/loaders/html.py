"""
ingestion/loaders/html.py

What problem does this solve?
- HTML files contain tags, scripts, styles, and metadata that are meaningless
  for RAG. Passing raw HTML to an embedding model degrades retrieval quality.

Why lxml instead of BeautifulSoup?
- lxml is already in requirements.txt (pulled in by other deps) — no extra install.
- lxml is 3-5x faster on large HTML files (C-level XML/HTML parser).
- HTMLParser handles real-world malformed HTML the same way browsers do.

Text extraction strategy:
1. Parse bytes with lxml HTMLParser (tolerant of malformed HTML).
2. Strip <script> and <style> nodes — code and CSS have no semantic value.
3. Walk remaining DOM with itertext() to collect all visible text nodes.
4. Collapse whitespace (HTML indentation tabs/spaces) into single spaces.

Title extraction:
- Reads <title> tag first (usually more descriptive than a filename).
- Falls back to filename stem if no <title> is present.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

from lxml import etree

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document


class HtmlLoader(BaseDocumentLoader):
    """
    What problem does this solve?
    - Strips all HTML structure and noise, returning only human-readable text
      suitable for chunking and semantic embedding.

    Why does this class exist?
    - Isolates lxml dependency and HTML-specific cleaning in one place.
      Changing the extraction strategy (e.g. preserving heading hierarchy)
      only touches this file.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """Handles .html and .htm. Both are identical in structure."""
        return [".html", ".htm"]

    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        What problem does this solve?
        - Converts an HTML page into clean, indexed plain text by stripping
          all tags and noise elements.

        Why are these inputs required?
        - file_path:  The HTML file to parse.
        - tenant_id:  Propagated to Document for tenant-scoped vector queries.

        Why use HTMLParser instead of the strict XML parser?
        - Real-world HTML is almost never valid XML (missing closing tags,
          unquoted attributes, etc.). HTMLParser is lenient and won't crash
          on production documents. The strict parser would raise on any malformed file.

        Why remove <script> and <style> before itertext()?
        - itertext() walks ALL text nodes, including JavaScript code inside
          <script> and CSS inside <style>. These contain zero semantic value
          for RAG and would badly pollute embeddings if included.

        Why split + rejoin whitespace instead of strip()?
        - HTML indentation adds tabs and multiple spaces between text nodes.
          split() on any whitespace + rejoin with single space collapses all
          of that into clean readable prose. strip() alone wouldn't fix mid-text
          whitespace.

        Why prefer <title> over filename for the title field?
        - HTML page titles are set by authors to be descriptive
          (e.g. "Q3 2024 Sales Report"). Filenames are often slugs or IDs
          (e.g. "q3_sr_v2_final"). Title tag gives better metadata for search UI.

        Why Document instead of str?
        - Same as all loaders: file_hash, tenant_id, source_path, and created_at
          are all required by downstream services.

        Raises:
        - FileNotFoundError  if the file does not exist.
        - ValueError         if the extension is not .html or .htm.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"HtmlLoader cannot handle: {path.suffix}")

        raw_bytes = path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        # HTMLParser is lenient — handles malformed real-world HTML without
        # raising exceptions unlike the strict XML parser.
        parser = etree.HTMLParser()
        tree = etree.fromstring(raw_bytes, parser)

        # Remove <script> and <style> subtrees before any text extraction.
        # JavaScript and CSS have no semantic value for RAG — including them
        # would heavily pollute embedding vectors.
        for tag in tree.iter("script", "style"):
            tag.getparent().remove(tag)

        # itertext() walks all remaining text nodes depth-first.
        # split() + rejoin collapses all whitespace variants (tabs, multiple
        # spaces, newlines from HTML indentation) into single spaces.
        tokens = " ".join(tree.itertext()).split()
        full_text = " ".join(tokens).strip()

        # Prefer <title> tag — it is typically more descriptive than the filename.
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
