"""
chunking/pipeline.py

What problem does this solve?
- After preprocessing, a ProcessedDocument must be split into DocumentChunks
  with all metadata attached (RBAC, token counts, prev/next links).
  Without a pipeline, the chunking_service would mix orchestration, RBAC
  propagation, and chunk construction into one large function.

Why a separate pipeline from ChunkingService?
- Same reason as PreprocessingPipeline: testability and layering.
  ChunkingPipeline is injected into ChunkingService via DI.
  Tests can test pipeline logic without the result-wrapping layer,
  and service tests can inject a stub pipeline without running real chunking.

Stage order:
  1. Count tokens in cleaned_text   (for stats — how large was the document?)
  2. Route to strategy              (which splitter fits this source_type?)
  3. Chunk via strategy             (produce texts + token_counts)
  4. Build DocumentChunk objects    (attach RBAC, sentence window links)
  5. Compute and return stats       (avg/max/min chunk size for monitoring)

Why attach prev_chunk_id / next_chunk_id here (not at retrieval)?
- Vector store entries are immutable after indexing. If these links are
  missing from metadata, the retrieval layer cannot expand context windows
  without a separate DB lookup. Attaching them at ingestion time makes
  every chunk fully self-describing.
"""

import uuid

from agent.chunking.router import ChunkingRouter
from agent.chunking.token_counter import TokenCounter
from agent.ingestion.models import DocumentChunk
from agent.processing.models import ProcessedDocument


class ChunkingPipeline:
    """
    What problem does this solve?
    - Orchestrates the full chunking flow: token counting → strategy selection
      → chunking → DocumentChunk construction with full metadata.

    Why does this class exist?
    - Separates orchestration (pipeline) from service-boundary concerns
      (ChunkingService: result wrapping, error handling).
    - Dependency injection: router and token_counter can be replaced in tests.

    Why inject router AND token_counter separately?
    - Router selects the strategy; TokenCounter counts pre-chunk input tokens.
      These are independent concerns. Injecting separately keeps each
      replaceable without affecting the other.
    """

    def __init__(
        self,
        router: ChunkingRouter | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """
        Args:
        - router:         Routes source_type → strategy. Default builds all three.
        - token_counter:  Counts input tokens for stats. Default uses cl100k_base.
        """
        self._router = router or ChunkingRouter()
        self._counter = token_counter or TokenCounter()

    def run(
        self, processed_doc: ProcessedDocument
    ) -> tuple[list[DocumentChunk], dict]:
        """
        What problem does this solve?
        - Full pipeline: ProcessedDocument → list[DocumentChunk] + stats dict.

        Why ProcessedDocument and not Document?
        - ChunkingPipeline needs cleaned_text (not raw text) and
          extracted_metadata.language/category for chunk metadata.
          ProcessedDocument carries all three: document, cleaned_text,
          extracted_metadata.

        Why return stats as a separate dict instead of embedding in chunks?
        - Stats are pipeline-level metrics (avg tokens, total chunks).
          They belong in ChunkingResult for logging/monitoring, not in
          individual DocumentChunk metadata.

        Why generate chunk IDs as "{document_id}_chunk_{index}"?
        - Deterministic: re-ingesting the same document produces the same
          chunk IDs. This matters for vector store upsert logic (update
          existing point instead of creating a duplicate).

        Why store prev_chunk_id / next_chunk_id in metadata?
        - Sentence window retrieval: when a chunk is retrieved by similarity,
          the retrieval layer fetches chunk[i-2..i+2] using these pointers.
          This gives the LLM 5× more context than the retrieved chunk alone.

        Why propagate ALL RBAC fields to every chunk?
        - Vector store filters on tenant_id, owner_id, visibility, access_roles
          at query time. Each chunk must be independently filterable without
          a join back to the parent document.

        Returns:
        - list[DocumentChunk]: Zero or more chunks, ready for EmbeddingService.
        - dict: Pipeline stats (total_input_tokens, total_chunks, avg/max/min).
        """
        document = processed_doc.document
        text = processed_doc.cleaned_text

        # ── Stage 1: Pre-chunk token count for stats ───────────────────────
        total_input_tokens = self._counter.count(text)

        # ── Stage 2: Strategy selection ────────────────────────────────────
        strategy = self._router.route(document.source_type)

        # ── Stage 3: Chunking ──────────────────────────────────────────────
        chunk_result = strategy.chunk(text, source_type=document.source_type)

        if not chunk_result.texts:
            return [], {
                "total_input_tokens": total_input_tokens,
                "total_chunks": 0,
                "strategy": strategy.strategy_name,
                "avg_tokens_per_chunk": 0,
                "max_chunk_tokens": 0,
                "min_chunk_tokens": 0,
            }

        # ── Stage 4: Build DocumentChunk objects ───────────────────────────
        total = len(chunk_result.texts)
        chunks: list[DocumentChunk] = []

        for i, (chunk_text, token_count) in enumerate(
            zip(chunk_result.texts, chunk_result.token_counts)
        ):
            chunk_id = f"{document.id}_chunk_{i}"
            prev_chunk_id = f"{document.id}_chunk_{i - 1}" if i > 0 else None
            next_chunk_id = f"{document.id}_chunk_{i + 1}" if i < total - 1 else None

            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document.id,
                text=chunk_text,
                chunk_index=i,
                metadata={
                    # ── RBAC (propagated for vector store filtering) ────────
                    "tenant_id":    document.tenant_id,
                    "owner_id":     document.owner_id,
                    "access_roles": document.access_roles,
                    "visibility":   document.visibility,

                    # ── Source provenance ──────────────────────────────────
                    "source_type":  document.source_type,
                    "source_path":  document.source_path,
                    "title":        document.title,
                    "document_id":  document.id,
                    "file_hash":    document.file_hash,
                    "created_at":   document.created_at.isoformat(),

                    # ── Chunking context ───────────────────────────────────
                    "chunk_index":  i,
                    "total_chunks": total,
                    "token_count":  token_count,
                    "strategy":     strategy.strategy_name,

                    # ── Sentence window links (retrieval context expansion) ─
                    "prev_chunk_id": prev_chunk_id,
                    "next_chunk_id": next_chunk_id,

                    # ── Extracted metadata (from PreprocessingService) ──────
                    "language":  processed_doc.extracted_metadata.language,
                    "category":  processed_doc.extracted_metadata.category,
                },
            )
            chunks.append(chunk)

        # ── Stage 5: Pipeline stats ────────────────────────────────────────
        token_counts = chunk_result.token_counts
        stats = {
            "total_input_tokens":  total_input_tokens,
            "total_chunks":        total,
            "strategy":            strategy.strategy_name,
            "avg_tokens_per_chunk": round(sum(token_counts) / total, 1),
            "max_chunk_tokens":    max(token_counts),
            "min_chunk_tokens":    min(token_counts),
        }

        return chunks, stats
