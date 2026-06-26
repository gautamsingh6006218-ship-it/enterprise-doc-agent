"""
chunking/router.py

What problem does this solve?
- ChunkingPipeline needs to select the right strategy per document without
  a long if/elif chain embedded in the pipeline itself. Without a router,
  every new source_type requires editing the pipeline code.

Why a ChunkingRouter class instead of a module-level dict?
- Strategies are stateful (they hold cached chunker instances). A class
  controls strategy lifecycle: one instance per router, shared safely.
- The router accepts injected strategy instances, enabling test isolation
  (inject a stub strategy without importing chonkie or LangChain).

How STRATEGY_MAP connects to the router:
- config.py defines STRATEGY_MAP: {source_type → strategy_class_name}.
- ChunkingRouter maps strategy_class_name → strategy instance.
- Adding a new source_type = update config.py only. Router logic unchanged.

Open/closed: add a new strategy by:
  1. Create NewStrategy(BaseChunkingStrategy)
  2. Add "NewStrategy" → instance in ChunkingRouter._strategies
  3. Map source_types to "NewStrategy" in config.STRATEGY_MAP
  No other files change.
"""

from agent.chunking.config import STRATEGY_MAP
from agent.chunking.strategies.base import BaseChunkingStrategy
from agent.chunking.strategies.sentence_window import SentenceWindowStrategy
from agent.chunking.strategies.structure_aware import StructureAwareStrategy
from agent.chunking.strategies.token_chunker import TokenStrategy


class ChunkingRouter:
    """
    What problem does this solve?
    - Decouples strategy selection from the pipeline. The pipeline calls
      route(source_type) and receives the right BaseChunkingStrategy.

    Why does this class exist?
    - Without it, ChunkingPipeline must import all three strategy classes
      and contain routing logic. That violates single responsibility:
      the pipeline should orchestrate, not decide.

    Why accept strategy instances instead of building them internally?
    - Dependency injection: tests pass lightweight stubs without triggering
      chonkie/LangChain initialisation. Production uses the defaults.
    - Allows swapping strategies at runtime (A/B testing different chunkers).
    """

    def __init__(
        self,
        sentence_strategy: BaseChunkingStrategy | None = None,
        structure_strategy: BaseChunkingStrategy | None = None,
        token_strategy: BaseChunkingStrategy | None = None,
    ) -> None:
        """
        Why three separate parameters instead of a dict?
        - Type safety: each parameter is typed as BaseChunkingStrategy.
          A dict would accept any value without type checking.
        - Explicitness: callers know exactly which strategy they're overriding.

        Args:
        - sentence_strategy:   For prose (PDF, DOCX, TXT, email).
        - structure_strategy:  For structured text (Markdown, HTML).
        - token_strategy:      For tabular data (XLSX, CSV).
        """
        self._strategies: dict[str, BaseChunkingStrategy] = {
            "SentenceWindowStrategy": sentence_strategy or SentenceWindowStrategy(),
            "StructureAwareStrategy": structure_strategy or StructureAwareStrategy(),
            "TokenStrategy":          token_strategy or TokenStrategy(),
        }

    def route(self, source_type: str) -> BaseChunkingStrategy:
        """
        What problem does this solve?
        - Returns the correct strategy for a document's source_type.

        Why look up by lowercase source_type?
        - Loaders produce lowercase source_type ("pdf", "docx"). Defensive
          lowercasing prevents "PDF" or "Pdf" from falling through to the default.

        Why fall back to "unknown" instead of raising?
        - Unknown file types should still be chunked (via SentenceWindow).
          Raising would block ingestion for uncommon file formats.

        Args:
        - source_type: Document.source_type (e.g. "pdf", "markdown", "xlsx").

        Returns the matching BaseChunkingStrategy instance.
        """
        strategy_name = STRATEGY_MAP.get(
            source_type.lower(), STRATEGY_MAP.get("unknown", "SentenceWindowStrategy")
        )
        return self._strategies.get(
            strategy_name,
            self._strategies["SentenceWindowStrategy"],
        )
