"""
retrieval/pipeline.py

What problem does this solve?
- A user query triggers four sequential steps: encode query → search (dense +
  keyword) → RRF fusion → rerank → window expansion. Without a pipeline class,
  the service layer would mix orchestration, result assembly, and stat collection.

Stage order:
  1. Encode query           encode_query() → dense vector + sparse weights
  2. Dense search           PgVector cosine similarity → top 50
  3. Keyword search         PostgreSQL FTS ts_rank_cd → top 50
  4. RRF fusion             reciprocal_rank_fusion(dense, keyword) → top 20
  5. Rerank                 BGE cross-encoder on top 20 → top 5
  6. Window expansion       fetch prev/next chunk neighbours for LLM context

Why these top-k values at each stage?
  - 50 candidates from each path: enough recall to survive RBAC filtering
    (if 40% of chunks are restricted, 50 ensures 30 valid results for RRF).
  - 20 after RRF: cross-encoder is O(n) — 20 pairs ≈ 100ms on CPU.
  - 5 final: ~2560 tokens of context sent to LLM after window expansion.

Why inject encoder, store, and reranker separately?
- Tests can mock each component independently.
- Production can swap the encoder without changing the reranker (model A/B test).
"""

from agent.embeddings.bge_encoder import BGEEncoder
from agent.embeddings.config import RERANK_TOP_K, RETRIEVAL_TOP_K
from agent.embeddings.store import PgVectorStore
from agent.retrieval.models import RBACContext, RetrievedContext, SearchResult
from agent.retrieval.reranker import BGEReranker
from agent.retrieval.rrf import build_id_index, reciprocal_rank_fusion

# After RRF, send this many candidates to the cross-encoder reranker.
# 20 is the sweet spot: enough for diverse reranking, fast enough for
# interactive response times (<200ms total).
_RRF_CANDIDATES = 20


class RetrievalPipeline:
    """
    What problem does this solve?
    - Orchestrates the full retrieval flow: encode → search → RRF → rerank
      → window expand → produce RetrievedContext for the LLM.

    Why a pipeline class instead of a service method?
    - Same pattern as ChunkingPipeline: separates orchestration from service-
      boundary concerns (result wrapping, error handling in RetrievalService).
    - Tests can run the pipeline directly to test retrieval logic without
      the result-wrapping layer.

    Why all three components injected?
    - Each can be mocked independently: test RRF logic without a real encoder,
      test window expansion without a real reranker.
    """

    def __init__(
        self,
        encoder: BGEEncoder,
        store: PgVectorStore,
        reranker: BGEReranker,
    ) -> None:
        """
        Args:
        - encoder:  BGEEncoder — encodes queries into dense vectors.
        - store:    PgVectorStore — runs dense + keyword search.
        - reranker: BGEReranker — cross-encoder reranking of RRF results.
        """
        self._encoder = encoder
        self._store = store
        self._reranker = reranker

    def search(
        self,
        query: str,
        rbac: RBACContext,
        top_k: int = RETRIEVAL_TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
    ) -> RetrievedContext:
        """
        What problem does this solve?
        - Full pipeline: query string → RetrievedContext with reranked chunks
          and expanded context window, ready for the LLM prompt.

        Why two retrieval paths (dense + keyword)?
        - Dense retrieval: good for semantic similarity ("employment agreement"
          matches "job contract").
        - Keyword retrieval: good for exact terms (clause numbers, product SKUs,
          person names). BGE-M3 embeddings can fail on rare exact-match queries.
        - Together they cover each other's blind spots.

        Why expand the sentence window AFTER reranking?
        - Window expansion fetches surrounding chunks for LLM context. Expanding
          before reranking would add noise (irrelevant neighbour chunks might
          displace relevant candidates). Expand only the final top-5.

        Args:
        - query:        User's natural language query.
        - rbac:         Tenant + user context for RBAC filtering.
        - top_k:        Candidates per retrieval path (default 50).
        - rerank_top_k: Final results after reranking (default 5).

        Returns RetrievedContext with chunks, window texts, and stats.
        """
        # ── Stage 1: Encode query ──────────────────────────────────────────
        query_encoding = self._encoder.encode_query(query)
        query_vector = query_encoding.dense_vectors[0]

        # ── Stage 2: Dense search ──────────────────────────────────────────
        dense_results = self._store.search_dense(
            query_vector=query_vector,
            rbac=rbac,
            top_k=top_k,
        )

        # ── Stage 3: Keyword search ────────────────────────────────────────
        keyword_results = self._store.search_keyword(
            query_text=query,
            rbac=rbac,
            top_k=top_k,
        )

        # ── Stage 4: RRF fusion ────────────────────────────────────────────
        result_lists = [r for r in [dense_results, keyword_results] if r]
        if not result_lists:
            return RetrievedContext(
                query=query,
                chunks=[],
                window_texts=[],
                retrieval_stats={
                    "n_dense": 0, "n_keyword": 0, "n_after_rrf": 0,
                    "n_after_rerank": 0,
                },
            )

        id_index = build_id_index(result_lists)
        fused = reciprocal_rank_fusion(result_lists)

        # Take top _RRF_CANDIDATES for reranking
        rrf_candidates: list[SearchResult] = []
        for chunk_id, rrf_score in fused[:_RRF_CANDIDATES]:
            if chunk_id in id_index:
                candidate = id_index[chunk_id]
                rrf_candidates.append(
                    SearchResult(
                        chunk_id=candidate.chunk_id,
                        document_id=candidate.document_id,
                        text=candidate.text,
                        score=rrf_score,
                        metadata=candidate.metadata,
                    )
                )

        # ── Stage 5: Rerank ────────────────────────────────────────────────
        reranked = self._reranker.rerank(
            query=query,
            candidates=rrf_candidates,
            top_k=rerank_top_k,
        )

        # ── Stage 6: Sentence window expansion ────────────────────────────
        window_chunk_ids: list[str] = []
        for result in reranked:
            prev_id = result.metadata.get("prev_chunk_id")
            next_id = result.metadata.get("next_chunk_id")
            if prev_id:
                window_chunk_ids.append(prev_id)
            if next_id:
                window_chunk_ids.append(next_id)

        # Remove IDs already in reranked results (avoid duplicates)
        reranked_ids = {r.chunk_id for r in reranked}
        window_ids_to_fetch = [
            cid for cid in window_chunk_ids if cid not in reranked_ids
        ]

        window_results = self._store.get_chunks_by_ids(window_ids_to_fetch)
        window_texts = [w.text for w in window_results]

        return RetrievedContext(
            query=query,
            chunks=reranked,
            window_texts=window_texts,
            retrieval_stats={
                "n_dense":        len(dense_results),
                "n_keyword":      len(keyword_results),
                "n_after_rrf":    len(rrf_candidates),
                "n_after_rerank": len(reranked),
                "n_window_texts": len(window_texts),
            },
        )
