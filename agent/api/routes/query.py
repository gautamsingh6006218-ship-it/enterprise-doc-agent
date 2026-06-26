from fastapi import APIRouter, Depends, HTTPException, status

from agent.api.auth import get_rbac_context
from agent.api.dependencies import get_llm_service, get_retrieval_service
from agent.api.models import ChunkResult, QueryRequest, QueryResponse
from agent.retrieval.models import RBACContext
from agent.services.llm_service import LLMService
from agent.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def search(
    request: QueryRequest,
    rbac: RBACContext = Depends(get_rbac_context),
    retrieval_svc: RetrievalService = Depends(get_retrieval_service),
    llm_svc: LLMService = Depends(get_llm_service),
) -> QueryResponse:
    """
    Semantic search over ingested documents.

    - Retrieval: BGE-M3 dense + BM25 keyword, fused via RRF, reranked by bge-reranker-v2-m3.
    - Set generate_answer=true to get an LLM-generated answer from retrieved chunks (Llama via Ollama).
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

    answer: str | None = None
    answer_model: str | None = None

    if request.generate_answer:
        llm_result = llm_svc.answer(request.query, context.chunks)
        if llm_result.success:
            answer = llm_result.answer
            answer_model = llm_result.model
        else:
            # LLM failure must not fail the whole query — chunks are still returned
            answer = f"[LLM error: {llm_result.error}]"

    return QueryResponse(
        query=context.query,
        chunks=chunks,
        window_texts=context.window_texts or [],
        retrieval_stats=context.retrieval_stats or {},
        answer=answer,
        answer_model=answer_model,
    )
