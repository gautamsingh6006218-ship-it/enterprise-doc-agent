"""
services/retrieval_service.py

What problem does this solve?
- API endpoints and LLM orchestrators need one entry point to turn a query
  string into a RetrievedContext, without knowing about encoder, store, RRF,
  reranker, or window expansion internals.

Why a service layer over calling the pipeline directly?
- Microservice boundary: the retrieval pipeline is the most latency-sensitive
  component. When it's extracted to its own container, only this file's
  interface is visible to callers.
- RetrievalResult (never raises) keeps FastAPI routes and async handlers safe.
- Dependency injection on the pipeline enables unit testing without real
  encoders, stores, or rerankers.

How this connects to the broader pipeline:
  EmbeddingService      →  chunks stored in PgVector
  RetrievalService      →  RetrievedContext  ← this file
  FastAPI /query route  →  LLM prompt + response
"""

from dataclasses import dataclass, field

from agent.retrieval.models import RBACContext, RetrievedContext
from agent.retrieval.pipeline import RetrievalPipeline


@dataclass
class RetrievalResult:
    """
    What problem does this solve?
    - Network failures, empty query strings, and encoder errors should not
      crash the FastAPI route. A result object gives a uniform interface.

    Fields:
    - success:   True = pipeline completed and at least one chunk found.
    - context:   RetrievedContext with ranked chunks + window texts.
    - error:     Human-readable failure reason. Only when success=False.
    """

    success: bool
    context: RetrievedContext | None = None
    error: str | None = None


class RetrievalService:
    """
    What problem does this solve?
    - Single entry point for all retrieval operations.
      Every API route calls query() — not the pipeline directly.

    Why accept pipeline as constructor arg?
    - Tests inject a stub pipeline without loading BGE-M3 or connecting to PgVector.
    - Production creates the real pipeline with encoder + store + reranker once
      at startup, then injects it here.

    Why separate query() from a hypothetical search_with_filters()?
    - RBAC context is passed explicitly (not inferred from session) to keep
      the service stateless. Different callers (API, batch job, admin tool)
      pass different RBAC contexts without subclassing.
    """

    def __init__(self, pipeline: RetrievalPipeline) -> None:
        """
        Why is pipeline required (no None default)?
        - Unlike EmbeddingService (encoder/store can be deferred), the retrieval
          pipeline cannot operate without all three components (encoder, store,
          reranker). Requiring it at construction fails fast on misconfiguration.

        Args:
        - pipeline: RetrievalPipeline with encoder, store, and reranker configured.
        """
        self._pipeline = pipeline

    def query(
        self,
        query: str,
        rbac: RBACContext,
    ) -> RetrievalResult:
        """
        What problem does this solve?
        - Runs the full retrieval pipeline (encode → search → RRF → rerank →
          window expand) and returns a RetrievalResult — safe to inspect,
          never raises.

        Why validate the query string here?
        - Empty queries would produce zero results (not an error) but waste
          encoder and DB resources. A fast validation at the service boundary
          returns a clear error before any network calls.

        Why pass rbac explicitly instead of reading from a session/context?
        - Stateless services are easier to test and reason about. The caller
          (FastAPI route, Celery task) assembles RBACContext from the auth token
          and passes it here. No shared mutable state.

        Args:
        - query: User's natural language query. Must be non-empty.
        - rbac:  Tenant and user context for RBAC filtering.

        Returns:
        - RetrievalResult(success=True, context=...) on success.
        - RetrievalResult(success=False, error="...") on failure.
        """
        if not query or not query.strip():
            return RetrievalResult(
                success=False,
                error="Query must be a non-empty string.",
            )

        try:
            context = self._pipeline.search(query=query.strip(), rbac=rbac)
            return RetrievalResult(success=True, context=context)
        except Exception as e:
            return RetrievalResult(
                success=False,
                error=f"Retrieval failed: {e}",
            )
