"""
qa/service.py

What problem does this solve?
- Answering a question takes two RAG steps: retrieve the chunks the caller
  may read, then optionally have the LLM answer from them. That composition
  is RAG logic, so it lives here and the HTTP route only maps the result.

Why does an LLM failure still return success=True?
- The retrieved chunks are still useful on their own. The caller gets them,
  and the answer field carries the LLM error.
"""

from dataclasses import dataclass

from agent.llm import LLMService
from agent.retrieval import RetrievalService, RetrievedContext
from agent.shared import RBACContext


@dataclass
class QAResult:
    """
    Fields:
    - success:      False only when retrieval failed.
    - context:      Retrieved chunks and stats. None if retrieval failed.
    - answer:       LLM answer, the LLM error text, or None if not requested.
    - answer_model: Model that wrote the answer. None if there is no answer.
    - error:        Retrieval failure reason. None on success.
    """

    success: bool
    context: RetrievedContext | None = None
    answer: str | None = None
    answer_model: str | None = None
    error: str | None = None


class QAService:

    def __init__(self, retrieval_service: RetrievalService, llm_service: LLMService) -> None:
        self._retrieval = retrieval_service
        self._llm = llm_service

    def ask(self, question: str, rbac: RBACContext, generate_answer: bool = False) -> QAResult:
        retrieval = self._retrieval.query(question, rbac)
        if not retrieval.success:
            return QAResult(success=False, error=retrieval.error or "Retrieval failed")

        context = retrieval.context
        if not generate_answer:
            return QAResult(success=True, context=context)

        llm_result = self._llm.answer(question, context.chunks)
        if not llm_result.success:
            return QAResult(
                success=True, context=context, answer=f"[LLM error: {llm_result.error}]"
            )
        return QAResult(
            success=True,
            context=context,
            answer=llm_result.answer,
            answer_model=llm_result.model,
        )
