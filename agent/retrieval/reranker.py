"""
retrieval/reranker.py

What problem does this solve?
- RRF produces a fused ranked list, but both dense and keyword retrieval use
  embedding similarity — they can agree on a wrong result if the query is
  ambiguous. A cross-encoder reranker reads the query AND each chunk together,
  capturing fine-grained relevance that bi-encoders miss.

Why bge-reranker-v2-m3 (local) over Cohere Rerank (API)?
- Enterprise documents often contain HR, legal, and financial content that
  must not leave the organisation's infrastructure. Cohere's API sends chunk
  text to an external server — a compliance and privacy risk.
- bge-reranker-v2-m3 runs fully on-premises. Zero network latency,
  zero per-call cost, zero data egress.
- Quality: bge-reranker-v2-m3 is in the same family as BGE-M3 (same BAAI lab,
  complementary training data). MTEB reranking leaderboard: top-5 open models.

Why a cross-encoder instead of the bi-encoder (BGE-M3) for reranking?
- Bi-encoders encode query and chunk independently — they cannot model
  token-level interactions between the two. Cross-encoders encode
  [query + chunk] jointly, catching cases where relevance depends on exact
  phrasing ("not liable" vs "liable" in a legal clause).

Why rerank only top_k candidates (not all 100)?
- Cross-encoding is O(n) where n = number of (query, chunk) pairs.
  At 5ms/pair on CPU, 20 candidates = 100ms — acceptable latency.
  100 candidates = 500ms — too slow for interactive queries.
"""

try:
    from FlagEmbedding import FlagReranker
    _FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    _FLAG_EMBEDDING_AVAILABLE = False

from agent.embeddings.config import BGE_RERANKER_MODEL, RERANK_TOP_K
from agent.retrieval.models import SearchResult


class BGEReranker:
    """
    What problem does this solve?
    - Wraps FlagReranker to rerank a candidate list by cross-encoder relevance.

    Why a class instead of a function?
    - FlagReranker loads ~300MB of model weights on instantiation.
      Caching on the instance avoids repeated disk reads across queries.
    - Dependency-injectable: tests pass a mock reranker without loading weights.

    Why return SearchResult objects (not just scores)?
    - The pipeline needs the full SearchResult (text, metadata) for the
      sentence window expansion step. Returning SearchResult objects keeps
      the pipeline consistent.
    """

    def __init__(
        self,
        model_name: str = BGE_RERANKER_MODEL,
        use_fp16: bool = True,
    ) -> None:
        """
        Why check _FLAG_EMBEDDING_AVAILABLE at instantiation?
        - Allows the package to import without FlagEmbedding installed.
          Fails only when BGEReranker is actually constructed.

        Args:
        - model_name: HuggingFace model ID. Default: BAAI/bge-reranker-v2-m3.
        - use_fp16:   Half-precision for ~2× speed on GPU. Default: True.
        """
        if not _FLAG_EMBEDDING_AVAILABLE:
            raise ImportError(
                "FlagEmbedding is required for BGEReranker. "
                "Install with: pip install FlagEmbedding"
            )
        self._model = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = RERANK_TOP_K,
    ) -> list[SearchResult]:
        """
        What problem does this solve?
        - Takes the RRF-fused candidate list and re-scores each candidate by
          jointly encoding (query, chunk) through the cross-encoder.
          Returns the top_k most relevant chunks.

        Why call compute_score with normalize=True?
        - Raw cross-encoder logits are unbounded. normalize=True applies
          sigmoid to produce scores in [0, 1], making them interpretable
          as relevance probabilities and comparable across queries.

        Why not rerank in batches?
        - FlagReranker.compute_score already handles batching internally.
          Passing all pairs at once is simpler and the library optimises
          batch size for the available hardware.

        Args:
        - query:      The user's search query.
        - candidates: SearchResult list from RRF (up to RETRIEVAL_TOP_K × 2).
        - top_k:      Number of results to return after reranking.

        Returns list[SearchResult] sorted by cross-encoder score descending,
        truncated to top_k. Each result's score is updated to the rerank score.
        """
        if not candidates:
            return []

        pairs = [[query, c.text] for c in candidates]
        scores: list[float] = self._model.compute_score(pairs, normalize=True)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for candidate, score in ranked[:top_k]:
            # Update score to cross-encoder relevance score
            reranked = SearchResult(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                text=candidate.text,
                score=float(score),
                metadata=candidate.metadata,
            )
            results.append(reranked)

        return results
