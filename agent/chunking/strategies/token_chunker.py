"""
chunking/strategies/token_chunker.py

What problem does this solve?
- Tabular documents (XLSX, CSV) have no sentence boundaries. Rows are
  short, repetitive, and semantically independent. Splitting by sentence
  would produce thousands of 3-10 token chunks — too small to embed usefully.
- A hard token budget ensures each chunk contains enough context (~512 tokens
  worth of rows) to be retrievable by semantic search.

Why chonkie TokenChunker instead of RecursiveCharacterTextSplitter?
- TokenChunker splits on exact token boundaries — no mid-word cuts,
  no character-count approximations.
- For tabular data, token boundaries approximate row boundaries well
  because rows are already newline-separated.
- chonkie returns token_count per chunk (no second tokenisation pass).

Why is this the ONLY strategy for XLSX/CSV?
- Tables have no headers to anchor section context.
- Sentences in tabular data are meaningless — "SKU|Price|Qty" is not prose.
- Structure-aware splitting (markdown headers) produces nothing on tables.
"""

from chonkie import TokenChunker

from agent.chunking.config import CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS
from agent.chunking.strategies.base import BaseChunkingStrategy, ChunkResult


class TokenStrategy(BaseChunkingStrategy):
    """
    What problem does this solve?
    - Enforces a hard token budget on tabular content where sentence and
      header boundaries don't exist.

    Why not use SentenceWindowStrategy for CSV?
    - Sentence splitters see "SKU1,9.99,50. SKU2,12.50,30." as two sentences.
      The period in "9.99" and "12.50" triggers sentence boundaries mid-row.
      TokenChunker ignores punctuation and splits cleanly by token count.

    Why inject chunk_size/chunk_overlap?
    - Tests use small sizes to produce multiple chunks from short strings.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_TARGET_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        """
        Why build TokenChunker once in __init__?
        - TokenChunker loads the tiktoken encoding on first use.
          Caching on the instance avoids repeated initialisation.

        Args:
        - chunk_size:    Token budget per chunk (default 512).
        - chunk_overlap: Overlap tokens between consecutive chunks (default 51).
        """
        self._chunker = TokenChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @property
    def strategy_name(self) -> str:
        return "TokenStrategy"

    def chunk(self, text: str, source_type: str = "unknown") -> ChunkResult:
        """
        What problem does this solve?
        - Splits tabular text into fixed token-size chunks within budget.

        Why return empty ChunkResult for empty text?
        - Empty XLSX sheets (all-blank export) should produce zero chunks,
          not an error. The pipeline skips empty results gracefully.

        Args:
        - text:        Cleaned tabular text (row-per-line format from loaders).
        - source_type: Not used by this strategy — kept for interface compliance.

        Returns ChunkResult with one entry per token-boundary chunk.
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
