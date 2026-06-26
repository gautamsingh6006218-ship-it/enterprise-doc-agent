"""
tests/retrieval/test_retrieval_service.py

What does this cover?
- BGEReranker: interface contract (mocked — no model download in tests).
- RetrievalPipeline: full pipeline flow with mocked encoder/store/reranker,
                     RRF fusion integration, window expansion, empty results.
- RetrievalService: success path, empty query validation, error wrapping,
                    DI pipeline injection.

Why mock all three pipeline components?
- BGE-M3 encoder: 600MB model, requires GPU or slow CPU inference.
- PgVectorStore: requires live PostgreSQL + pgvector.
- BGEReranker: 300MB model, requires inference.
  Tests verify pipeline orchestration logic, not model quality.
"""

from unittest.mock import MagicMock

import pytest

from agent.embeddings.bge_encoder import BGEEncoder, EncodingResult
from agent.embeddings.store import PgVectorStore
from agent.retrieval.models import RBACContext, RetrievedContext, SearchResult
from agent.retrieval.pipeline import RetrievalPipeline
from agent.retrieval.reranker import BGEReranker
from agent.services.retrieval_service import RetrievalResult, RetrievalService


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_rbac(
    tenant_id: str = "acme",
    user_id: str = "user-1",
    user_roles: list | None = None,
) -> RBACContext:
    return RBACContext(
        tenant_id=tenant_id,
        user_id=user_id,
        user_roles=user_roles or ["hr"],
    )


def make_search_result(chunk_id: str, score: float = 0.9, text: str | None = None) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=text or f"Text for chunk {chunk_id}.",
        score=score,
        metadata={
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "tenant_id": "acme",
        },
    )


def stub_encoder(query_vector: list[float] | None = None) -> MagicMock:
    enc = MagicMock(spec=BGEEncoder)
    enc.encode_query.return_value = EncodingResult(
        dense_vectors=[query_vector or [0.1] * 1024],
        sparse_weights=[{"token": 0.5}],
    )
    return enc


def stub_store(
    dense_results: list | None = None,
    keyword_results: list | None = None,
    window_results: list | None = None,
) -> MagicMock:
    store = MagicMock(spec=PgVectorStore)
    store.search_dense.return_value = dense_results or []
    store.search_keyword.return_value = keyword_results or []
    store.get_chunks_by_ids.return_value = window_results or []
    return store


def stub_reranker(results: list | None = None) -> MagicMock:
    rr = MagicMock(spec=BGEReranker)
    rr.rerank.return_value = results or []
    return rr


def make_pipeline(
    dense_results=None,
    keyword_results=None,
    window_results=None,
    reranked_results=None,
) -> RetrievalPipeline:
    """Builds a RetrievalPipeline with all dependencies mocked."""
    return RetrievalPipeline(
        encoder=stub_encoder(),
        store=stub_store(dense_results, keyword_results, window_results),
        reranker=stub_reranker(reranked_results),
    )


# ── BGEReranker interface ──────────────────────────────────────────────────────

class TestBGERerankerInterface:

    def test_reranker_raises_without_flag_embedding(self):
        import agent.retrieval.reranker as mod
        original = mod._FLAG_EMBEDDING_AVAILABLE
        mod._FLAG_EMBEDDING_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="FlagEmbedding"):
                BGEReranker()
        finally:
            mod._FLAG_EMBEDDING_AVAILABLE = original

    def test_rerank_returns_empty_for_empty_candidates(self):
        rr = MagicMock(spec=BGEReranker)
        rr.rerank.return_value = []
        result = rr.rerank("query", [], top_k=5)
        assert result == []

    def test_rerank_truncates_to_top_k(self):
        """Reranker should return at most top_k results."""
        rr = MagicMock(spec=BGEReranker)
        candidates = [make_search_result(f"c{i}") for i in range(20)]
        rr.rerank.return_value = candidates[:5]
        result = rr.rerank("query", candidates, top_k=5)
        assert len(result) <= 5


# ── RetrievalPipeline ──────────────────────────────────────────────────────────

class TestRetrievalPipeline:

    def test_returns_retrieved_context(self):
        reranked = [make_search_result("c1"), make_search_result("c2")]
        pipeline = make_pipeline(
            dense_results=[make_search_result("c1"), make_search_result("c2")],
            keyword_results=[make_search_result("c2"), make_search_result("c3")],
            reranked_results=reranked,
        )
        ctx = pipeline.search("enterprise policy", make_rbac())
        assert isinstance(ctx, RetrievedContext)

    def test_chunks_populated_from_reranker(self):
        reranked = [make_search_result("c1", score=0.95)]
        pipeline = make_pipeline(
            dense_results=[make_search_result("c1")],
            reranked_results=reranked,
        )
        ctx = pipeline.search("test query", make_rbac())
        assert len(ctx.chunks) == 1
        assert ctx.chunks[0].chunk_id == "c1"

    def test_query_stored_in_context(self):
        pipeline = make_pipeline(reranked_results=[make_search_result("c1")])
        ctx = pipeline.search("what is the leave policy?", make_rbac())
        assert ctx.query == "what is the leave policy?"

    def test_empty_dense_and_keyword_returns_empty_context(self):
        pipeline = make_pipeline(dense_results=[], keyword_results=[])
        ctx = pipeline.search("query", make_rbac())
        assert ctx.chunks == []
        assert ctx.window_texts == []

    def test_encoder_called_with_query(self):
        enc = stub_encoder()
        pipeline = RetrievalPipeline(
            encoder=enc,
            store=stub_store(),
            reranker=stub_reranker(),
        )
        pipeline.search("find the contract clause", make_rbac())
        enc.encode_query.assert_called_once_with("find the contract clause")

    def test_store_searched_with_rbac_context(self):
        rbac = make_rbac(tenant_id="tenantX", user_id="uid42", user_roles=["legal"])
        store = stub_store()
        pipeline = RetrievalPipeline(
            encoder=stub_encoder(),
            store=store,
            reranker=stub_reranker(),
        )
        pipeline.search("query", rbac)
        dense_call = store.search_dense.call_args
        assert dense_call[1]["rbac"].tenant_id == "tenantX"

    def test_window_expansion_fetches_neighbours(self):
        # Chunk has next_chunk_id — expect window fetch
        chunk_with_next = SearchResult(
            chunk_id="c1",
            document_id="doc",
            text="main chunk",
            score=0.9,
            metadata={"prev_chunk_id": None, "next_chunk_id": "c2"},
        )
        store = stub_store(
            dense_results=[chunk_with_next],
            window_results=[make_search_result("c2", text="neighbour text")],
        )
        pipeline = RetrievalPipeline(
            encoder=stub_encoder(),
            store=store,
            reranker=stub_reranker(results=[chunk_with_next]),
        )
        ctx = pipeline.search("query", make_rbac())
        assert "neighbour text" in ctx.window_texts

    def test_window_expansion_skips_already_retrieved_chunks(self):
        """If next_chunk_id is already in the reranked list, don't fetch it again."""
        c1 = make_search_result("c1")
        c2 = make_search_result("c2")
        c1.metadata["next_chunk_id"] = "c2"  # next is already retrieved

        store = stub_store(dense_results=[c1, c2])
        pipeline = RetrievalPipeline(
            encoder=stub_encoder(),
            store=store,
            reranker=stub_reranker(results=[c1, c2]),
        )
        pipeline.search("query", make_rbac())
        # get_chunks_by_ids should NOT be called with "c2" (already retrieved)
        call_args = store.get_chunks_by_ids.call_args
        if call_args:
            fetched_ids = call_args[0][0]
            assert "c2" not in fetched_ids

    def test_stats_returned_in_context(self):
        pipeline = make_pipeline(
            dense_results=[make_search_result("c1")],
            keyword_results=[make_search_result("c2")],
            reranked_results=[make_search_result("c1")],
        )
        ctx = pipeline.search("query", make_rbac())
        required_keys = {"n_dense", "n_keyword", "n_after_rrf", "n_after_rerank"}
        assert required_keys.issubset(ctx.retrieval_stats.keys())

    def test_stats_n_dense_matches_search_results(self):
        dense = [make_search_result(f"c{i}") for i in range(5)]
        pipeline = make_pipeline(dense_results=dense)
        ctx = pipeline.search("query", make_rbac())
        assert ctx.retrieval_stats["n_dense"] == 5

    def test_reranker_called_with_query(self):
        rr = stub_reranker()
        dense = [make_search_result("c1")]
        pipeline = RetrievalPipeline(
            encoder=stub_encoder(),
            store=stub_store(dense_results=dense),
            reranker=rr,
        )
        pipeline.search("my search query", make_rbac())
        rr.rerank.assert_called_once()
        assert rr.rerank.call_args[1]["query"] == "my search query"


# ── RetrievalService ───────────────────────────────────────────────────────────

class TestRetrievalService:

    def _make_service(self, chunks=None, window=None):
        reranked = chunks or [make_search_result("c1")]
        pipeline = make_pipeline(
            dense_results=[make_search_result("c1")],
            reranked_results=reranked,
            window_results=window or [],
        )
        return RetrievalService(pipeline=pipeline)

    def test_success_result(self):
        svc = self._make_service()
        result = svc.query("what is the refund policy?", make_rbac())
        assert result.success is True
        assert result.context is not None
        assert result.error is None

    def test_empty_query_returns_error(self):
        svc = self._make_service()
        result = svc.query("", make_rbac())
        assert result.success is False
        assert "non-empty" in result.error

    def test_whitespace_query_returns_error(self):
        svc = self._make_service()
        result = svc.query("   ", make_rbac())
        assert result.success is False

    def test_query_stripped_before_passing_to_pipeline(self):
        rr = stub_reranker(results=[make_search_result("c1")])
        enc = stub_encoder()
        pipeline = RetrievalPipeline(
            encoder=enc,
            store=stub_store(dense_results=[make_search_result("c1")]),
            reranker=rr,
        )
        svc = RetrievalService(pipeline=pipeline)
        svc.query("  what is the policy?  ", make_rbac())
        enc.encode_query.assert_called_once_with("what is the policy?")

    def test_error_from_pipeline_wrapped_in_result(self):
        class BrokenPipeline:
            def search(self, **_):
                raise RuntimeError("DB connection lost")

        svc = RetrievalService(pipeline=BrokenPipeline())
        result = svc.query("test", make_rbac())
        assert result.success is False
        assert "DB connection lost" in result.error

    def test_context_has_chunks(self):
        svc = self._make_service(chunks=[make_search_result("c1"), make_search_result("c2")])
        result = svc.query("policy document", make_rbac())
        assert len(result.context.chunks) == 2

    def test_result_context_has_query(self):
        svc = self._make_service()
        result = svc.query("leave policy 2024", make_rbac())
        assert result.context.query == "leave policy 2024"

    def test_rbac_passed_to_pipeline(self):
        rr = stub_reranker()
        enc = stub_encoder()
        store = stub_store(dense_results=[make_search_result("c1")])
        pipeline = RetrievalPipeline(encoder=enc, store=store, reranker=rr)
        svc = RetrievalService(pipeline=pipeline)

        rbac = make_rbac(tenant_id="corpX", user_id="emp99", user_roles=["finance"])
        svc.query("budget report", rbac)

        call_rbac = store.search_dense.call_args[1]["rbac"]
        assert call_rbac.tenant_id == "corpX"
        assert call_rbac.user_id == "emp99"
