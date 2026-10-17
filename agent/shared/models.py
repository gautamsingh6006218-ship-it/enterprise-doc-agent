"""
shared/models.py

What problem does this solve?
- RBACContext and SearchResult cross module boundaries: auth builds the
  RBACContext, the vector store filters by it, retrieval ranks SearchResults,
  and the LLM service cites them. When they lived in retrieval/models.py,
  embeddings had to import retrieval while retrieval imported embeddings —
  a module cycle.

Why a shared kernel instead of picking an owner module?
- Both types are plain value objects with no behaviour. Every module may
  depend on agent.shared; agent.shared depends on nothing. Keep it that small.
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
