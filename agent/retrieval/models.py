"""
retrieval/models.py

What problem does this solve?
- The retrieval pipeline passes results between search, RRF, reranking, and
  window expansion. Without shared model classes, each step defines its own
  dict structure and the pipeline breaks when keys diverge.

Why separate from ingestion/models.py?
- RetrievedContext is a retrieval-time shape, not an ingestion shape. Keeping
  it separate prevents the ingestion models from growing retrieval concerns
  and makes each module's contract explicit.
- SearchResult and RBACContext cross module boundaries, so they live in
  agent.shared.
"""

from dataclasses import dataclass, field
from typing import Any

from agent.shared import SearchResult


@dataclass
class RetrievedContext:
    """
    What problem does this solve?
    - The LLM needs the final chunks plus context about how they were retrieved
      (for observability and debugging). Without a container object, the service
      layer returns bare lists and stats are lost.

    Why include window_texts separately?
    - window_texts are the ±2 neighbouring chunks fetched for context expansion.
      They are passed to the LLM but NOT used for embedding-based relevance.
      Keeping them separate lets the LLM prompt builder distinguish "retrieved"
      from "context window" chunks.

    Fields:
    - query:           The original query string — for prompt building.
    - chunks:          Final reranked chunks (top RERANK_TOP_K). Embed these.
    - window_texts:    Neighbouring chunks for context. Not re-embedded.
    - retrieval_stats: Metrics for monitoring: n_dense, n_keyword, n_after_rrf,
                       n_after_rerank, strategy used.
    """

    query: str
    chunks: list[SearchResult]
    window_texts: list[str] = field(default_factory=list)
    retrieval_stats: dict[str, Any] = field(default_factory=dict)
