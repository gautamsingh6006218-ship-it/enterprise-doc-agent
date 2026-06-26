"""
services/embedding_service.py

What problem does this solve?
- ChunkingService produces list[DocumentChunk]. Before retrieval can work,
  every chunk must have a dense vector stored in PgVector. Without a service
  layer, every caller (API endpoint, Celery worker) would manage encoder
  batching, store upsert, and error handling themselves.

Why a service layer over calling encoder + store directly?
- Microservice boundary: same pattern as IngestionService and
  PreprocessingService. When embedding becomes its own container, only this
  file's interface is exposed to callers.
- EmbeddingResult (never raises) keeps async workers safe.
- Batched encoding handled here — callers pass all chunks, service manages GPU.

How this connects to the broader pipeline:
  IngestionService      →  Document
  PreprocessingService  →  ProcessedDocument
  ChunkingService       →  list[DocumentChunk]
  EmbeddingService      →  chunks stored in PgVector  ← this file
  RetrievalService      →  RetrievedContext
"""

from dataclasses import dataclass, field

from agent.embeddings.bge_encoder import BGEEncoder
from agent.embeddings.store import PgVectorStore
from agent.ingestion.models import DocumentChunk


@dataclass
class EmbeddingResult:
    """
    What problem does this solve?
    - Exceptions during encoding (OOM, model load failure) or DB writes
      (connection lost, constraint violation) must not crash the caller.
      A result object gives a uniform interface: check success, read stats.

    Fields:
    - success:         True = all chunks encoded and stored.
    - embedded_count:  Number of chunks successfully written to PgVector.
    - error:           Human-readable failure reason. Only when success=False.
    - stats:           Metrics dict (model, batch_size, total_tokens). Only on success.
    """

    success: bool
    embedded_count: int = 0
    error: str | None = None
    stats: dict | None = field(default=None)


class EmbeddingService:
    """
    What problem does this solve?
    - Single entry point to encode DocumentChunks and persist their vectors
      in PgVector. Every downstream consumer calls embed() — not encoder/store.

    Why accept encoder and store as constructor args?
    - Tests inject mock encoder (no 600MB model) and mock store (no PostgreSQL).
    - Production creates real encoder + store from environment config.
    - A/B testing different models: swap the encoder instance, nothing else changes.
    """

    def __init__(
        self,
        encoder: BGEEncoder | None = None,
        store: PgVectorStore | None = None,
    ) -> None:
        """
        Why not create encoder/store here from defaults?
        - BGEEncoder loads model weights on instantiation. Creating it in
          __init__ with defaults would load weights even in tests.
          None defaults allow lazy construction: tests inject mocks, production
          creates real objects outside the service.

        Args:
        - encoder: BGEEncoder instance. Must be provided for non-test use.
        - store:   PgVectorStore instance. Must be provided for non-test use.
        """
        self._encoder = encoder
        self._store = store

    def embed(self, chunks: list[DocumentChunk]) -> EmbeddingResult:
        """
        What problem does this solve?
        - Encodes all chunks in one batched call to BGE-M3, then upserts
          all vectors in one batch to PgVector.

        Why upsert instead of insert?
        - Re-ingesting a document (updated content, model upgrade) must update
          existing vectors, not create duplicates. ON CONFLICT DO UPDATE in
          the store handles this.

        Why compute sparse_weights but not use them for search (yet)?
        - sparse_weights are stored in JSONB for future sparse vector search
          (pgvector sparsevec). The current search uses PostgreSQL FTS for
          keyword retrieval. Storing sparse weights now means zero re-ingestion
          when we enable BGE-M3 sparse search.

        Why not return the vectors in EmbeddingResult?
        - Vectors are large (1024 floats × n chunks). Returning them from the
          service would double memory usage. Callers that need vectors query
          the store directly.

        Args:
        - chunks: list[DocumentChunk] from ChunkingService. Must be non-empty.

        Returns:
        - EmbeddingResult(success=True, embedded_count=n, stats={...}) on success.
        - EmbeddingResult(success=False, error="...") on any failure.
        """
        if not chunks:
            return EmbeddingResult(success=True, embedded_count=0, stats={"note": "empty input"})

        if self._encoder is None or self._store is None:
            return EmbeddingResult(
                success=False,
                error="EmbeddingService not configured: encoder and store must be provided.",
            )

        try:
            texts = [c.text for c in chunks]

            # ── Encode all chunks ──────────────────────────────────────────
            encoding = self._encoder.encode_documents(texts, return_sparse=True)

            # ── Upsert to PgVector ─────────────────────────────────────────
            self._store.upsert_chunks(
                chunk_ids=    [c.id           for c in chunks],
                document_ids= [c.document_id  for c in chunks],
                texts=        texts,
                dense_vectors=encoding.dense_vectors,
                sparse_weights=encoding.sparse_weights,
                metadatas=    [c.metadata     for c in chunks],
            )

            return EmbeddingResult(
                success=True,
                embedded_count=len(chunks),
                stats={
                    "total_chunks":   len(chunks),
                    "vector_dim":     len(encoding.dense_vectors[0]) if encoding.dense_vectors else 0,
                    "sparse_stored":  any(bool(w) for w in encoding.sparse_weights),
                },
            )

        except Exception as e:
            return EmbeddingResult(
                success=False,
                error=f"Embedding failed: {e}",
            )
