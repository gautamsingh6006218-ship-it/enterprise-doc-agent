"""
retrieval/rrf.py

What problem does this solve?
- Dense retrieval and keyword retrieval produce two separate ranked lists.
  The lists have different score scales (cosine similarity vs. BM25 tf-rank)
  that cannot be directly compared or averaged. Without fusion, you must
  pick one path and discard the other's signal.

Why RRF (Reciprocal Rank Fusion) instead of score averaging or weighted sum?
- RRF cares about rank position, not raw score magnitude. This makes it
  robust to scale differences between cosine similarity (0-1) and BM25
  tf-rank (unbounded). No calibration needed.
- Empirically, RRF matches or outperforms score-based fusion at typical
  enterprise RAG scales (BEIR benchmark, 2023).
- Formula: score(d) = Σ 1 / (k + rank_i(d))  where k=60 is the smoothing constant.
  k=60 was established in the original RRF paper (Cormack et al., 2009).

Why k=60?
- k prevents the top-ranked document in any single list from dominating.
  With k=60, rank-1 contributes 1/61 ≈ 0.0164 and rank-50 contributes
  1/110 ≈ 0.0091. The difference is small — rank matters but not exclusively.
  k < 60 amplifies top-rank advantage; k > 60 flattens rank signal.

Why not LangChain EnsembleRetriever?
- We have clean microservices with our own SearchResult model.
  EnsembleRetriever requires LangChain Document objects and Retriever wrappers
  — an abstraction mismatch. RRF is 12 lines; the dependency is not worth it.
"""

from agent.retrieval.models import SearchResult

_RRF_K = 60  # Cormack et al. 2009 default — do not change without benchmarking


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    """
    What problem does this solve?
    - Merges N ranked lists into one by rank position, producing a single
      ranked list suitable for reranking.

    Why take list[list[SearchResult]] instead of list[list[str]]?
    - We need to pass the full SearchResult objects through to the pipeline
      so text and metadata are available without a second DB lookup.
      The fusion operates on chunk_id keys internally.

    Why return list[tuple[str, float]] (id, score) and not list[SearchResult]?
    - The fused list may contain items from either result list. The caller
      (RetrievalPipeline) resolves the full SearchResult objects from an
      id→result index it already holds. Returning tuples keeps this function
      pure and stateless.

    Args:
    - result_lists: Each inner list is a ranked list of SearchResult objects.
                    Order within each list = rank (index 0 = best).
    - k:            RRF smoothing constant (default 60).

    Returns list of (chunk_id, rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, result in enumerate(results):
            cid = result.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def build_id_index(
    result_lists: list[list[SearchResult]],
) -> dict[str, SearchResult]:
    """
    What problem does this solve?
    - After RRF returns (chunk_id, score) tuples, the pipeline needs the full
      SearchResult for each id (text, metadata). This builds the lookup index.

    Why a separate function instead of building it inside reciprocal_rank_fusion?
    - Keeps reciprocal_rank_fusion pure (no side effects). The index is only
      needed by the caller, not by the fusion logic itself.

    Args:
    - result_lists: All SearchResult lists — same input as reciprocal_rank_fusion.

    Returns dict mapping chunk_id → SearchResult (last-write wins for duplicates).
    """
    index: dict[str, SearchResult] = {}
    for results in result_lists:
        for result in results:
            index[result.chunk_id] = result
    return index
