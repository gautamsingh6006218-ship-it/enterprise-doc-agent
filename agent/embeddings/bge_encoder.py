"""
embeddings/bge_encoder.py

What problem does this solve?
- EmbeddingService needs to call BGE-M3 to produce dense + sparse vectors for
  every DocumentChunk. Without an encoder class, the service would import and
  manage FlagEmbedding directly — mixing model lifecycle with service logic.

Why BGE-M3 and not two separate models (dense + BM25)?
- BGE-M3 is a single model that produces BOTH dense (1024-dim) and sparse
  (learned lexical) vectors in one forward pass. Compared to running a dense
  model + a separate BM25 index:
    • One model to download, version, and manage
    • Sparse weights are semantically-aware (not raw TF-IDF)
    • Single GPU load — no memory overhead from a second model

Why fp16?
- BGE-M3 in fp16 uses ~1.2GB GPU RAM vs ~2.4GB fp32.
  Quality difference is negligible for embedding tasks (< 0.1% on MTEB).
  fp16 doubles throughput on modern GPUs.

Why a class (not module-level functions)?
- FlagModel is expensive to initialise (loads weights from disk/HuggingFace).
  Caching it on the instance avoids re-initialisation for every batch.
  The singleton pattern in EmbeddingService reuses one encoder instance.
"""

from dataclasses import dataclass

try:
    from FlagEmbedding import BGEM3FlagModel
    _FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    _FLAG_EMBEDDING_AVAILABLE = False

from agent.embeddings.config import BGE_M3_MODEL, EMBEDDING_BATCH_SIZE


@dataclass
class EncodingResult:
    """
    What problem does this solve?
    - encode_documents() and encode_query() return both dense and sparse vectors.
      A dataclass keeps them paired and named instead of returning an unnamed tuple.

    Fields:
    - dense_vectors:   list[list[float]] — one 1024-dim vector per input text.
    - sparse_weights:  list[dict[str, float]] — one {token: weight} dict per text.
                       Empty dicts when return_sparse=False.
    """

    dense_vectors: list[list[float]]
    sparse_weights: list[dict[str, float]]


class BGEEncoder:
    """
    What problem does this solve?
    - Wraps BGEM3FlagModel to produce dense + sparse vectors for document chunks
      and search queries, with batching and a clean injectable interface.

    Why accept model_name/batch_size as constructor args?
    - Tests inject a mock BGEEncoder without loading the 600MB model weights.
    - Deployment profiles with smaller GPUs can reduce batch_size.

    Why separate encode_documents() from encode_query()?
    - At query time, only one vector is needed (no batching needed).
    - encode_query() is a convenience wrapper that avoids the caller having
      to unpack a single-element list.
    """

    def __init__(
        self,
        model_name: str = BGE_M3_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        use_fp16: bool = True,
    ) -> None:
        """
        Why check _FLAG_EMBEDDING_AVAILABLE here instead of at import?
        - Import-time failure prevents the whole agent package from loading
          even when the encoder is never used (e.g. read-only retrieval mode).
          Checking at instantiation gives a clear, actionable error message.

        Args:
        - model_name:  HuggingFace model ID. Default: BAAI/bge-m3.
        - batch_size:  Chunks encoded per forward pass. Default: 32.
        - use_fp16:    Use half-precision weights. Default: True.
        """
        if not _FLAG_EMBEDDING_AVAILABLE:
            raise ImportError(
                "FlagEmbedding is required for BGEEncoder. "
                "Install with: pip install FlagEmbedding"
            )
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self._batch_size = batch_size

    def encode_documents(
        self,
        texts: list[str],
        return_sparse: bool = True,
    ) -> EncodingResult:
        """
        What problem does this solve?
        - Encodes a batch of document chunk texts into dense + sparse vectors
          for storage in PgVector.

        Why batch internally?
        - Callers pass the full list of chunks (possibly thousands).
          Internally batching avoids OOM errors on large documents.

        Why max_length=8192?
        - BGE-M3 supports 8192 token context. Our chunks are ≤1024 tokens,
          so this is headroom — but setting it explicitly prevents the library
          from silently truncating longer chunks.

        Args:
        - texts:          List of chunk texts to encode.
        - return_sparse:  Also compute sparse (lexical) weights. Default True.

        Returns EncodingResult with one vector per input text.
        """
        if not texts:
            return EncodingResult(dense_vectors=[], sparse_weights=[])

        output = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

        dense = [v.tolist() for v in output["dense_vecs"]]
        sparse = output.get("lexical_weights", [{} for _ in texts]) if return_sparse else [{} for _ in texts]

        return EncodingResult(dense_vectors=dense, sparse_weights=sparse)

    def encode_query(self, query: str) -> EncodingResult:
        """
        What problem does this solve?
        - Encodes a single search query into dense + sparse vectors for
          similarity search against stored chunk vectors.

        Why not reuse encode_documents() for queries?
        - Convenience: retrieval always encodes exactly one query. Wrapping
          here avoids the caller unpacking a single-element list.

        Args:
        - query: The user's natural language search query.

        Returns EncodingResult with exactly one dense vector and one sparse dict.
        """
        return self.encode_documents([query], return_sparse=True)
