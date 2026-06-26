"""
api/routes/query.py

What problem does this solve?
- Exposes RetrievalService over HTTP so the frontend / LLM orchestration layer
  can search the vector store without Python imports.

POST /query
- Accepts: JSON body with query string + optional top_k
- Returns: QueryResponse (matching chunks, window texts, retrieval stats)

Why POST instead of GET for search?
- Query strings in GET URLs are logged in proxy/load balancer access logs.
  Enterprise queries may contain sensitive terms. POST body is not logged
  by default in most infrastructure.
- Pydantic validation on the request body catches bad inputs before they
  hit the vector store.

Why pass RBACContext from JWT directly to retrieval?
- The retrieval pipeline enforces tenant_id + visibility + access_roles
  in every SQL query. The JWT is the source of truth for who is asking.
  No additional role lookup needed at query time.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from agent.api.auth import get_rbac_context
from agent.api.dependencies import get_retrieval_service
from agent.api.models import ChunkResult, QueryRequest, QueryResponse
from agent.retrieval.models import RBACContext
from agent.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def search(
    request: QueryRequest,
    rbac: RBACContext = Depends(get_rbac_context),
    retrieval_svc: RetrievalService = Depends(get_retrieval_service),
) -> QueryResponse:
    """
    Semantic search over ingested documents.

    Returns ranked chunks from BGE-M3 dense + BM25 keyword retrieval,
    fused via RRF and reranked by bge-reranker-v2-m3.
    """
    result = retrieval_svc.query(request.query, rbac)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error or "Retrieval failed",
        )

    context = result.context
    chunks = [
        ChunkResult(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            text=c.text,
            score=c.score,
            metadata=c.metadata or {},
        )
        for c in context.chunks
    ]

    return QueryResponse(
        query=context.query,
        chunks=chunks,
        window_texts=context.window_texts or [],
        retrieval_stats=context.retrieval_stats or {},
    )
