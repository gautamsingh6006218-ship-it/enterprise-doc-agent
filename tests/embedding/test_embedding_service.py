"""
tests/embedding/test_embedding_service.py

What does this cover?
- BGEEncoder: interface contract (no real model weights needed — mocked).
- PgVectorStore: upsert, search_dense, search_keyword, get_chunks_by_ids
                 with a mock psycopg2 connection.
- EmbeddingService: success path, empty input, missing encoder/store,
                    error wrapping via DI, stats returned.
- RRF: reciprocal_rank_fusion correctness, build_id_index, edge cases.

Why mock the encoder and store?
- BGE-M3 model weights are ~600MB. Downloading them in CI is slow and
  requires GPU/CPU resources that test environments may not have.
- psycopg2 requires a live PostgreSQL instance. Tests must run offline.
- We test the service LOGIC (batching, error wrapping, stat collection),
  not the correctness of BGE-M3 embeddings or SQL queries.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from agent.embeddings.bge_encoder import BGEEncoder, EncodingResult
from agent.embeddings.store import PgVectorStore
from agent.ingestion.models import DocumentChunk
from agent.retrieval.models import RBACContext, SearchResult
from agent.retrieval.rrf import build_id_index, reciprocal_rank_fusion
from agent.services.embedding_service import EmbeddingResult, EmbeddingService


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_chunk(
    chunk_id: str = "doc-001_chunk_0",
    doc_id: str = "doc-001",
    text: str = "Sample chunk text for embedding.",
    tenant_id: str = "acme",
    owner_id: str = "user-1",
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id=doc_id,
        text=text,
        chunk_index=int(chunk_id.split("_")[-1]),
        metadata={
            "tenant_id":     tenant_id,
            "owner_id":      owner_id,
            "access_roles":  ["hr"],
            "visibility":    "public",
            "token_count":   10,
            "prev_chunk_id": None,
            "next_chunk_id": None,
        },
    )


def make_chunks(n: int = 3) -> list[DocumentChunk]:
    return [make_chunk(chunk_id=f"doc-001_chunk_{i}", text=f"Chunk {i} text.") for i in range(n)]


def stub_encoder(n: int = 3) -> MagicMock:
    """Returns a mock BGEEncoder that produces dummy 1024-dim vectors."""
    enc = MagicMock(spec=BGEEncoder)
    enc.encode_documents.return_value = EncodingResult(
        dense_vectors=[[0.1] * 1024] * n,
        sparse_weights=[{"token": 0.5}] * n,
    )
    return enc


def stub_store() -> MagicMock:
    """Returns a mock PgVectorStore that accepts upsert without a DB."""
    store = MagicMock(spec=PgVectorStore)
    store.upsert_chunks.return_value = None
    return store


# ── BGEEncoder interface ───────────────────────────────────────────────────────

class TestBGEEncoderInterface:
    """Tests that verify the EncodingResult contract — not the model itself."""

    def test_encoding_result_has_dense_and_sparse(self):
        result = EncodingResult(
            dense_vectors=[[0.1] * 1024],
            sparse_weights=[{"hello": 0.8}],
        )
        assert len(result.dense_vectors) == 1
        assert len(result.sparse_weights) == 1

    def test_encoding_result_empty(self):
        result = EncodingResult(dense_vectors=[], sparse_weights=[])
        assert result.dense_vectors == []
        assert result.sparse_weights == []

    def test_bge_encoder_raises_without_flag_embedding(self):
        """If FlagEmbedding is not installed, BGEEncoder raises ImportError."""
        import agent.embeddings.bge_encoder as mod
        original = mod._FLAG_EMBEDDING_AVAILABLE
        mod._FLAG_EMBEDDING_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="FlagEmbedding"):
                BGEEncoder()
        finally:
            mod._FLAG_EMBEDDING_AVAILABLE = original


# ── EmbeddingService ───────────────────────────────────────────────────────────

class TestEmbeddingService:

    def test_success_with_chunks(self):
        chunks = make_chunks(3)
        svc = EmbeddingService(encoder=stub_encoder(3), store=stub_store())
        result = svc.embed(chunks)
        assert result.success is True
        assert result.embedded_count == 3
        assert result.error is None

    def test_empty_chunks_returns_success_zero_count(self):
        svc = EmbeddingService(encoder=stub_encoder(0), store=stub_store())
        result = svc.embed([])
        assert result.success is True
        assert result.embedded_count == 0

    def test_missing_encoder_returns_error(self):
        svc = EmbeddingService(encoder=None, store=stub_store())
        result = svc.embed(make_chunks(1))
        assert result.success is False
        assert "not configured" in result.error

    def test_missing_store_returns_error(self):
        svc = EmbeddingService(encoder=stub_encoder(1), store=None)
        result = svc.embed(make_chunks(1))
        assert result.success is False
        assert "not configured" in result.error

    def test_encoder_called_with_chunk_texts(self):
        chunks = make_chunks(2)
        enc = stub_encoder(2)
        svc = EmbeddingService(encoder=enc, store=stub_store())
        svc.embed(chunks)
        enc.encode_documents.assert_called_once_with(
            [c.text for c in chunks], return_sparse=True
        )

    def test_store_upsert_called_with_correct_ids(self):
        chunks = make_chunks(2)
        store = stub_store()
        svc = EmbeddingService(encoder=stub_encoder(2), store=store)
        svc.embed(chunks)
        call_kwargs = store.upsert_chunks.call_args[1]
        assert call_kwargs["chunk_ids"] == [c.id for c in chunks]

    def test_store_upsert_called_with_metadata(self):
        chunks = make_chunks(2)
        store = stub_store()
        svc = EmbeddingService(encoder=stub_encoder(2), store=store)
        svc.embed(chunks)
        call_kwargs = store.upsert_chunks.call_args[1]
        assert call_kwargs["metadatas"] == [c.metadata for c in chunks]

    def test_error_from_encoder_wrapped_in_result(self):
        enc = MagicMock(spec=BGEEncoder)
        enc.encode_documents.side_effect = RuntimeError("GPU OOM")
        svc = EmbeddingService(encoder=enc, store=stub_store())
        result = svc.embed(make_chunks(1))
        assert result.success is False
        assert "GPU OOM" in result.error

    def test_error_from_store_wrapped_in_result(self):
        store = MagicMock(spec=PgVectorStore)
        store.upsert_chunks.side_effect = Exception("DB connection lost")
        svc = EmbeddingService(encoder=stub_encoder(1), store=store)
        result = svc.embed(make_chunks(1))
        assert result.success is False
        assert "DB connection lost" in result.error

    def test_stats_returned_on_success(self):
        chunks = make_chunks(2)
        svc = EmbeddingService(encoder=stub_encoder(2), store=stub_store())
        result = svc.embed(chunks)
        assert result.stats is not None
        assert result.stats["total_chunks"] == 2
        assert result.stats["vector_dim"] == 1024

    def test_sparse_stored_true_when_weights_nonempty(self):
        svc = EmbeddingService(encoder=stub_encoder(2), store=stub_store())
        result = svc.embed(make_chunks(2))
        assert result.stats["sparse_stored"] is True


# ── RRF ────────────────────────────────────────────────────────────────────────

def make_result(chunk_id: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=f"Text for {chunk_id}",
        score=score,
    )


class TestRRF:

    def test_single_list_preserves_order(self):
        results = [make_result("a"), make_result("b"), make_result("c")]
        fused = reciprocal_rank_fusion([results])
        ids = [cid for cid, _ in fused]
        assert ids == ["a", "b", "c"]

    def test_item_in_both_lists_scores_higher(self):
        list1 = [make_result("a"), make_result("b")]
        list2 = [make_result("b"), make_result("c")]
        fused = dict(reciprocal_rank_fusion([list1, list2]))
        # "b" appears in both lists — should outscore "a" and "c"
        assert fused["b"] > fused["a"]
        assert fused["b"] > fused["c"]

    def test_empty_result_list_ignored(self):
        results = [make_result("x"), make_result("y")]
        fused = reciprocal_rank_fusion([results, []])
        assert len(fused) == 2

    def test_all_empty_lists_returns_empty(self):
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_scores_decrease_with_rank(self):
        results = [make_result(f"chunk_{i}") for i in range(5)]
        fused = reciprocal_rank_fusion([results])
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)

    def test_k_parameter_affects_scores(self):
        results = [make_result("x")]
        score_k60 = dict(reciprocal_rank_fusion([results], k=60))["x"]
        score_k10 = dict(reciprocal_rank_fusion([results], k=10))["x"]
        # Smaller k → higher score for rank-1
        assert score_k10 > score_k60

    def test_build_id_index_all_results_indexed(self):
        list1 = [make_result("a"), make_result("b")]
        list2 = [make_result("c")]
        index = build_id_index([list1, list2])
        assert set(index.keys()) == {"a", "b", "c"}

    def test_build_id_index_last_write_wins(self):
        r1 = SearchResult("x", "doc", "text from list 1", 0.9)
        r2 = SearchResult("x", "doc", "text from list 2", 0.5)
        index = build_id_index([[r1], [r2]])
        assert index["x"].text == "text from list 2"

    def test_build_id_index_empty_input(self):
        assert build_id_index([]) == {}
        assert build_id_index([[]]) == {}


# ── PgVectorStore interface ────────────────────────────────────────────────────

class TestPgVectorStoreInterface:
    """
    Tests that verify the store's public contract using a mock connection.
    We test that the correct SQL is invoked and results are correctly mapped.
    We do NOT test the actual SQL execution or pgvector type handling.
    """

    def _make_dict_row(self, chunk_id="c1", doc_id="d1", text="hello", score=0.9):
        """Creates a dict-like row mimicking psycopg2 DictRow."""
        row = {
            "id": chunk_id,
            "document_id": doc_id,
            "text": text,
            "metadata": {"tenant_id": "acme"},
            "score": score,
        }
        return row

    def test_search_dense_returns_search_results(self):
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            self._make_dict_row("c1", score=0.95),
            self._make_dict_row("c2", score=0.88),
        ]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        store = PgVectorStore(connection=mock_conn)
        rbac = RBACContext(tenant_id="acme", user_id="u1", user_roles=["hr"])

        results = store.search_dense(
            query_vector=[0.1] * 1024,
            rbac=rbac,
            top_k=10,
        )
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score == 0.95

    def test_search_keyword_returns_search_results(self):
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            self._make_dict_row("c3", score=0.7),
        ]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        store = PgVectorStore(connection=mock_conn)
        rbac = RBACContext(tenant_id="acme", user_id="u1", user_roles=["hr"])

        results = store.search_keyword(
            query_text="enterprise contract",
            rbac=rbac,
            top_k=10,
        )
        assert len(results) == 1
        assert results[0].chunk_id == "c3"

    def test_get_chunks_by_ids_empty_input_returns_empty(self):
        mock_conn = MagicMock()
        store = PgVectorStore(connection=mock_conn)
        result = store.get_chunks_by_ids([])
        assert result == []
        mock_conn.cursor.assert_not_called()

    def test_get_chunks_by_ids_returns_search_results(self):
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            {"id": "c1", "document_id": "d1", "text": "window text", "metadata": {}},
        ]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        store = PgVectorStore(connection=mock_conn)
        results = store.get_chunks_by_ids(["c1"])
        assert len(results) == 1
        assert results[0].score == 0.0  # window chunks have score 0

    def test_store_raises_without_pgvector(self):
        import agent.embeddings.store as mod
        original = mod._PGVECTOR_AVAILABLE
        mod._PGVECTOR_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="pgvector"):
                PgVectorStore(connection_string="postgresql://test")
        finally:
            mod._PGVECTOR_AVAILABLE = original

    def test_store_raises_without_connection_or_dsn(self):
        with pytest.raises(ValueError, match="PGVECTOR_URL"):
            import os
            env_backup = os.environ.pop("PGVECTOR_URL", None)
            try:
                PgVectorStore()
            finally:
                if env_backup:
                    os.environ["PGVECTOR_URL"] = env_backup
