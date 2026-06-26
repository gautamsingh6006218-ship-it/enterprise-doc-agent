"""
chunking/strategies/base.py

What problem does this solve?
- ChunkingRouter needs a uniform interface to call any strategy without
  knowing its implementation. Without an ABC, the router must import
  every strategy class and if/elif on type — closed to extension.

Why an ABC instead of a Protocol?
- ABC enforces implementation at class-definition time (inheriting without
  implementing strategy_name/chunk raises TypeError on instantiation).
  Protocol checks are duck-typing at call time — that's too late for catching
  a partially-implemented strategy at startup.

Why a ChunkResult dataclass instead of list[str]?
- Strategies compute token counts during chunking (chonkie already has them).
  Returning them avoids a second tokenization pass in the pipeline.
- strategy_name is included so pipeline stats identify which strategy ran.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChunkResult:
    """
    What problem does this solve?
    - Strategies return texts + pre-computed token counts + strategy name
      in a single object. The pipeline builds DocumentChunk metadata from this.

    Why include token_counts from the strategy (not the pipeline)?
    - chonkie SentenceChunker and TokenChunker already track token counts
      per chunk during splitting. Re-counting in the pipeline would be
      redundant tokenization for ~80% of documents.
    - StructureAwareStrategy uses LangChain and must count explicitly — done
      once inside the strategy, not twice.

    Fields:
    - texts:         List of chunk text strings, in document order.
    - token_counts:  Token count per chunk, parallel to texts.
    - strategy_name: Which strategy produced this result — stored in
                     DocumentChunk metadata for observability.
    """

    texts: list[str] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    strategy_name: str = "unknown"


class BaseChunkingStrategy(ABC):
    """
    What problem does this solve?
    - All three strategies (SentenceWindow, StructureAware, Token) share the
      same interface so ChunkingRouter can treat them uniformly.

    Why does this class exist?
    - Open/closed principle: new strategies (semantic, hierarchical) can be
      added by subclassing, without touching ChunkingRouter or ChunkingPipeline.
    - Dependency injection: ChunkingPipeline accepts a BaseChunkingStrategy —
      tests can inject a minimal stub without importing chonkie or LangChain.

    Why source_type passed to chunk() instead of __init__?
    - The same strategy instance is reused across different documents.
      Passing source_type per call allows one strategy object to handle
      subtle format differences (e.g. "html" vs "markdown" in StructureAware)
      without maintaining per-document state on the strategy.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """
        Why required?
        - ChunkingPipeline stores strategy_name in DocumentChunk.metadata
          for observability. Without this, debugging "why was this chunk
          produced" requires examining router logic, not the chunk itself.
        """

    @abstractmethod
    def chunk(self, text: str, source_type: str = "unknown") -> ChunkResult:
        """
        What problem does this solve?
        - Splits a cleaned document text into chunks within the token budget.

        Why take source_type even though the router already selected the strategy?
        - StructureAwareStrategy uses source_type to switch between markdown
          header splitting and HTML recursive splitting. One strategy class
          handles both, selected by source_type at call time.

        Args:
        - text:        Cleaned, normalised text from PreprocessingService.
        - source_type: Original document format ("pdf", "markdown", "html", etc.)

        Returns ChunkResult with texts, token_counts, and strategy_name.
        """
