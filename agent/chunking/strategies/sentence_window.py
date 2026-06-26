"""
chunking/strategies/sentence_window.py

What problem does this solve?
- Prose documents (PDF, DOCX, TXT, email) have meaningful sentence
  boundaries. Splitting mid-sentence breaks semantic units and degrades
  retrieval quality: the LLM gets fragments, not complete thoughts.

Why SentenceChunker (chonkie) instead of RecursiveCharacterTextSplitter?
- chonkie splits on sentence boundaries first, then packs sentences into
  token-sized windows. LangChain's recursive splitter splits on '\n\n',
  '\n', then ' ' — it cuts mid-sentence when paragraphs exceed the budget.
- chonkie returns token_count per chunk, saving a second tokenisation pass.

Why "Sentence Window" specifically?
- At ingestion: chunks are small (≤512 tokens) for precise embedding.
- At retrieval: prev_chunk_id / next_chunk_id stored in metadata allow
  the retrieval layer to expand to ±SENTENCE_WINDOW_SIZE neighbours,
  giving the LLM ~2560 tokens of context while keeping stored embeddings precise.
- This is the production retrieval pattern used by LlamaIndex and LangChain
  "sentence window retrieval".
"""

from chonkie import SentenceChunker

from agent.chunking.config import CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS
from agent.chunking.strategies.base import BaseChunkingStrategy, ChunkResult


class SentenceWindowStrategy(BaseChunkingStrategy):
    """
    What problem does this solve?
    - Primary chunking strategy for all prose documents.
      Respects sentence boundaries, stays within token budget.

    Why is this the primary/default strategy?
    - The majority of enterprise documents (PDFs, Word docs, emails, reports)
      are prose. Sentence-aware chunking outperforms character/token splitting
      on all standard RAG benchmarks for prose retrieval.

    Why inject chunk_size/chunk_overlap instead of hard-coding?
    - Tests use small sizes (e.g. chunk_size=50) to get multiple chunks from
      short test strings without needing full-length test documents.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_TARGET_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        """
        Why instantiate SentenceChunker here (not in chunk())?
        - SentenceChunker loads the sentence boundary model on first use.
          Building it once in __init__ avoids re-initialisation per document.

        Args:
        - chunk_size:    Target token count per chunk (default 512).
        - chunk_overlap: Overlap tokens between consecutive chunks (default 51).
        """
        self._chunker = SentenceChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def strategy_name(self) -> str:
        return "SentenceWindowStrategy"

    def chunk(self, text: str, source_type: str = "unknown") -> ChunkResult:
        """
        What problem does this solve?
        - Splits prose text into sentence-boundary-respecting token chunks.

        Why return empty ChunkResult for empty/whitespace text?
        - PreprocessingService may produce empty cleaned_text for scanned
          PDFs with no recognised text. The pipeline must handle zero chunks
          gracefully — an empty ChunkResult signals "nothing to embed".

        Why not filter out short chunks?
        - Short final chunks (e.g. a 10-token closing section) are still
          useful for retrieval. The sentence window context expands them
          at retrieval time. Filtering would silently drop document endings.

        Args:
        - text:        Cleaned text from PreprocessingService.
        - source_type: Not used by this strategy — kept for interface compliance.

        Returns ChunkResult with one entry per sentence-window chunk.
        """
        if not text.strip():
            return ChunkResult(
                texts=[],
                token_counts=[],
                strategy_name=self.strategy_name,
            )

        chunks = self._chunker.chunk(text)

        return ChunkResult(
            texts=[c.text for c in chunks],
            token_counts=[c.token_count for c in chunks],
            strategy_name=self.strategy_name,
        )
