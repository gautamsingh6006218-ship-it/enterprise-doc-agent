"""
tests/qa/test_qa_service.py

What does this cover?
- QAService.ask():
    retrieval fails             → success=False with the retrieval error,
                                  LLM not called.
    generate_answer=False       → retrieved context only, LLM not called.
    generate_answer=True        → LLM answers from the retrieved chunks.
    LLM fails                   → success=True, chunks kept, error in answer.
"""

from unittest.mock import MagicMock

from agent.llm import LLMResult, LLMService
from agent.qa import QAService
from agent.retrieval import RetrievalResult, RetrievalService, RetrievedContext
from agent.shared import RBACContext, SearchResult

RBAC = RBACContext(tenant_id="t1", user_id="u1", user_roles=[])
CHUNK = SearchResult(chunk_id="c1", document_id="doc-1", text="Leave is 20 days.", score=0.9)


def _service(retrieval: RetrievalResult | None = None, llm: LLMResult | None = None):
    retrieval_svc = MagicMock(spec=RetrievalService)
    retrieval_svc.query.return_value = retrieval or RetrievalResult(
        success=True, context=RetrievedContext(query="leave policy?", chunks=[CHUNK]),
    )
    llm_svc = MagicMock(spec=LLMService)
    llm_svc.answer.return_value = llm or LLMResult(success=True, answer="20 days.", model="llama3.2")
    return QAService(retrieval_service=retrieval_svc, llm_service=llm_svc), retrieval_svc, llm_svc


class TestAsk:

    def test_retrieval_failure_returns_error(self):
        svc, _, llm_svc = _service(retrieval=RetrievalResult(success=False, error="vector DB down"))
        result = svc.ask("leave policy?", RBAC, generate_answer=True)
        assert result.success is False
        assert result.error == "vector DB down"
        llm_svc.answer.assert_not_called()

    def test_without_answer_returns_context_only(self):
        svc, retrieval_svc, llm_svc = _service()
        result = svc.ask("leave policy?", RBAC)
        assert result.success is True
        assert result.context.chunks == [CHUNK]
        assert result.answer is None
        retrieval_svc.query.assert_called_once_with("leave policy?", RBAC)
        llm_svc.answer.assert_not_called()

    def test_answer_generated_from_retrieved_chunks(self):
        svc, _, llm_svc = _service()
        result = svc.ask("leave policy?", RBAC, generate_answer=True)
        assert result.answer == "20 days."
        assert result.answer_model == "llama3.2"
        llm_svc.answer.assert_called_once_with("leave policy?", [CHUNK])

    def test_llm_failure_keeps_chunks(self):
        svc, _, _ = _service(llm=LLMResult(success=False, error="Ollama not running"))
        result = svc.ask("leave policy?", RBAC, generate_answer=True)
        assert result.success is True
        assert result.context.chunks == [CHUNK]
        assert result.answer == "[LLM error: Ollama not running]"
        assert result.answer_model is None
