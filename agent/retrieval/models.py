"""
retrieval/models.py

What problem does this solve?
- The retrieval pipeline passes results between search, RRF, reranking, and
  window expansion. Without shared model classes, each step defines its own
  dict structure and the pipeline breaks when keys diverge.

Why separate from ingestion/models.py?
- SearchResult and RetrievedContext are retrieval-time shapes, not ingestion
  shapes. Keeping them separate prevents the ingestion models from growing
  retrieval concerns and makes each layer's contract explicit.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RBACContext:
    """
    What problem does this solve?
    - Every search query must be scoped to a tenant and filtered by the
      requesting user's roles. Without a dedicated context object, RBAC
      parameters would be passed as loose kwargs, making call signatures
      inconsistent and easy to omit.

    Why all three fields required (no defaults)?
    - tenant_id:  Missing = cross-tenant data leak. Always required.
    - user_id:    Required to enforce "private" visibility (owner only).
    - user_roles: Required to enforce "restricted" visibility.

    Fields:
    - tenant_id:  Isolates query to one enterprise tenant's data.
    - user_id:    The requesting user — checked against owner_id for private docs.
    - user_roles: The user's roles — checked against access_roles for restricted docs.
    """

    tenant_id: str
    user_id: str
    user_roles: list[str]


@dataclass
class SearchResult:
    """
    What problem does this solve?
    - Dense search, keyword search, and reranking all return a ranked list
      of chunks. A shared model keeps the pipeline type-safe across all steps.

    Why include score?
    - RRF fusion needs the rank position (derived from score order) from each
      retrieval path. Reranking produces a new score from the cross-encoder.
    - Storing score lets callers inspect retrieval confidence without re-querying.

    Fields:
    - chunk_id:    Matches DocumentChunk.id — used as the key in RRF fusion.
    - document_id: Parent document for provenance and window expansion.
    - text:        The chunk text sent to the reranker and returned to the LLM.
    - score:       Similarity/BM25/RRF/rerank score — higher = more relevant.
    - metadata:    Full chunk metadata (RBAC, source_type, token_count, etc.).
    """

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


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
