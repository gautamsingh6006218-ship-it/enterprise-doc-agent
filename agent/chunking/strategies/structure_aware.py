"""
chunking/strategies/structure_aware.py

What problem does this solve?
- Markdown documents have ## headers that define semantic sections.
  Cutting across a header boundary produces chunks that mix topics.
  A "Troubleshooting" section merged into an "Installation" chunk
  degrades retrieval because both topics map to the same embedding.

Why two passes instead of just RecursiveCharacterTextSplitter?
- Pass 1 (MarkdownHeaderTextSplitter): produces semantically complete
  sections; each section is about one topic.
- Pass 2 (RecursiveCharacterTextSplitter with tiktoken): splits any
  oversized section into token-budgeted sub-chunks, each prefixed with
  the header breadcrumb so context is not lost.
- Using only RecursiveCharacterTextSplitter would produce chunks that
  cut across headers silently — no way to enforce section boundaries.

Why include the header breadcrumb ("[h1 > h2]") in chunk text?
- When a chunk about "Installation > Troubleshooting" is retrieved, the
  LLM sees it came from the Troubleshooting sub-section of Installation.
  Without the breadcrumb, the chunk reads as context-free text.

Why from_tiktoken_encoder instead of character-based splitting?
- Character counts diverge from token counts by up to 4x depending on
  content (code blocks, URLs, non-ASCII). Token-based ensures the
  512-token target is respected regardless of content type.
"""

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from agent.chunking.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    TIKTOKEN_ENCODING,
)
from agent.chunking.strategies.base import BaseChunkingStrategy, ChunkResult
from agent.chunking.token_counter import TokenCounter

# Headers to split on — order matters (largest → smallest heading level)
_MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


class StructureAwareStrategy(BaseChunkingStrategy):
    """
    What problem does this solve?
    - Markdown and HTML documents have structural hierarchy (headers, sections)
      that should be preserved as chunk boundaries.

    Why handle both markdown and HTML in one class?
    - Both are "structured text" — the router routes them to this strategy.
    - Markdown: two-pass (header split → token overflow split).
    - HTML: single-pass token split (HTML tags stripped by loaders, so
      the text is already flat prose or list-formatted).

    Why not a separate HtmlStrategy class?
    - HTML after loader processing is flat text without structural markup.
      Using a separate class would duplicate the token-splitter logic.
      One strategy, two code paths, selected by source_type at call time.

    Why inject chunk_size/chunk_overlap?
    - Tests need small chunk sizes to produce multiple chunks from short strings.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_TARGET_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        """
        Why build splitters in __init__ instead of chunk()?
        - from_tiktoken_encoder() loads the tiktoken encoding on first call.
          Caching avoids repeated initialisation across thousands of documents.

        Args:
        - chunk_size:    Target token count per chunk (default 512).
        - chunk_overlap: Overlap tokens between consecutive chunks (default 51).
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Recursive token splitter: used for overflow sections and all HTML
        self._token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=TIKTOKEN_ENCODING,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Header splitter: used only for markdown in Pass 1
        # strip_headers=True (default) removes the header lines from page_content
        # — that's intentional; we prepend the breadcrumb ourselves.
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MARKDOWN_HEADERS,
            strip_headers=True,
        )

        self._counter = TokenCounter()

    @property
    def strategy_name(self) -> str:
        return "StructureAwareStrategy"

    def chunk(self, text: str, source_type: str = "markdown") -> ChunkResult:
        """
        What problem does this solve?
        - Produces semantically coherent chunks that respect document structure.

        Args:
        - text:        Cleaned text from PreprocessingService.
        - source_type: "markdown" triggers two-pass header-aware splitting.
                       Any other value uses single-pass token splitting.

        Returns ChunkResult with texts, token_counts, and strategy_name.
        """
        if not text.strip():
            return ChunkResult(
                texts=[],
                token_counts=[],
                strategy_name=self.strategy_name,
            )

        if source_type == "markdown":
            texts = self._chunk_markdown(text)
        else:
            texts = self._chunk_html(text)

        token_counts = self._counter.count_batch(texts)
        return ChunkResult(
            texts=texts,
            token_counts=token_counts,
            strategy_name=self.strategy_name,
        )

    def _chunk_markdown(self, text: str) -> list[str]:
        """
        What problem does this solve?
        - Splits markdown into header-scoped sections, then ensures each
          section is within the token budget.

        Why prepend the breadcrumb?
        - MarkdownHeaderTextSplitter strips header lines from page_content.
          Without the breadcrumb, retrieved chunks appear context-free.

        Why fall back to token splitter when no sections produced?
        - A markdown file with no '#' headers returns an empty list from
          MarkdownHeaderTextSplitter. Falling back preserves chunking.
        """
        sections = self._header_splitter.split_text(text)

        if not sections:
            return self._token_splitter.split_text(text)

        texts: list[str] = []
        for section in sections:
            # Build "h1 > h2 > h3" breadcrumb from header metadata
            breadcrumb_parts = [v for v in section.metadata.values() if v]
            breadcrumb = " > ".join(breadcrumb_parts)

            content = (
                f"[{breadcrumb}]\n{section.page_content}"
                if breadcrumb
                else section.page_content
            )

            # Skip entirely empty sections (blank sections between headers)
            if not content.strip():
                continue

            # Split any section that exceeds the token budget
            token_count = self._counter.count(content)
            if token_count > self._chunk_size:
                sub_chunks = self._token_splitter.split_text(content)
                texts.extend(sub_chunks)
            else:
                texts.append(content)

        return texts if texts else self._token_splitter.split_text(text)

    def _chunk_html(self, text: str) -> list[str]:
        """
        What problem does this solve?
        - HTML loaders produce flat text (tags stripped, content preserved).
          Standard token splitting is appropriate — no structural markers remain.

        Why not use MarkdownHeaderTextSplitter on HTML?
        - HTML headers become plain text after extraction (e.g. "Introduction"
          not "## Introduction"). The header splitter would produce no splits.
        """
        return self._token_splitter.split_text(text)
