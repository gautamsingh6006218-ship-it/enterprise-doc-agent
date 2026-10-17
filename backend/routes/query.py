from fastapi import APIRouter, Depends, HTTPException, status

from backend.auth import get_rbac_context
from backend.dependencies import get_qa_service
from backend.models import ChunkResult, QueryRequest, QueryResponse
from agent.qa import QAService
from agent.shared import RBACContext

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def search(
    request: QueryRequest,
    rbac: RBACContext = Depends(get_rbac_context),
    qa_svc: QAService = Depends(get_qa_service),
) -> QueryResponse:
    """
    Semantic search over ingested documents.

    - Retrieval: BGE-M3 dense + BM25 keyword, fused via RRF, reranked by bge-reranker-v2-m3.
    - Set generate_answer=true to get an LLM-generated answer from retrieved chunks (Llama via Ollama).
    """
    result = qa_svc.ask(request.query, rbac, generate_answer=request.generate_answer)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error,
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
        answer=result.answer,
        answer_model=result.answer_model,
    )
