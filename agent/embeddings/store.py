"""
embeddings/store.py

What problem does this solve?
- Dense vectors, sparse weights, and chunk metadata need to be persisted in
  PostgreSQL + pgvector so the RetrievalService can search them at query time.
  Without a store class, every service that touches the DB has to manage
  connection lifecycle, SQL strings, and pgvector type registration.

Why PgVector (PostgreSQL) instead of a dedicated vector DB (Pinecone, Weaviate)?
- PostgreSQL already handles our relational metadata (RBAC fields).
  Keeping vectors in the same DB eliminates the "dual write" problem: no risk
  of a chunk being in the vector store but missing from the metadata store.
- pgvector HNSW achieves ~10ms p99 at 500K vectors — sufficient for enterprise RAG.
- Zero additional infrastructure: same Postgres instance, same backups.

Why store sparse_weights as JSONB?
- BGE-M3 sparse output is {token_string: weight} — variable-length, not fixed-dim.
  JSONB stores it without a schema change. pgvector's sparsevec type would
  require fixed-dim sparse vectors (compatible only if we switch to SPLADE).
  JSONB also makes the weights inspectable with standard SQL tools.

Why inject the connection instead of creating it internally?
- Tests inject a mock/fake connection without starting PostgreSQL.
- Production code passes a real psycopg2 connection from a connection pool.
  DI keeps the store testable without a running database.
"""

import json
import os

import psycopg2
import psycopg2.extras

from agent.embeddings.config import DB_TABLE, RETRIEVAL_TOP_K
from agent.retrieval.models import RBACContext, SearchResult

try:
    from pgvector.psycopg2 import register_vector as _register_vector
    _PGVECTOR_AVAILABLE = True
except ImportError:
    _PGVECTOR_AVAILABLE = False


# ── RBAC SQL fragment ──────────────────────────────────────────────────────────
# Reused in every query. Enforces tenant isolation + visibility rules.
# Why three visibility levels?
#   public     → any user in the tenant can read
#   restricted → only users whose roles overlap with access_roles
#   private    → only the document owner
_RBAC_WHERE = """
    metadata->>'tenant_id' = %(tenant_id)s
    AND (
        metadata->>'visibility' = 'public'
        OR (
            metadata->>'visibility' = 'restricted'
            AND EXISTS (
                SELECT 1
                FROM jsonb_array_elements_text(metadata->'access_roles') AS r
                WHERE r = ANY(%(user_roles)s)
            )
        )
        OR (
            metadata->>'visibility' = 'private'
            AND metadata->>'owner_id' = %(user_id)s
        )
    )
"""


class PgVectorStore:
    """
    What problem does this solve?
    - Central data access layer for all chunk storage and retrieval operations.
      EmbeddingService calls upsert_chunks(); RetrievalService calls
      search_dense() and search_keyword().

    Why two search methods instead of one unified search?
    - Dense and keyword search use completely different PostgreSQL mechanisms
      (vector index vs GIN/tsvector). Keeping them separate allows independent
      tuning (different top_k, different RBAC param handling) and makes the
      RetrievalPipeline's RRF fusion explicit.

    Why accept connection as constructor arg?
    - Tests inject a mock connection without needing a running PostgreSQL instance.
    - Production creates a connection from PGVECTOR_URL env var.
    """

    def __init__(self, connection=None, connection_string: str | None = None) -> None:
        """
        Why prefer injected connection over connection_string?
        - Injected connection: test isolation, connection pool integration.
        - connection_string: convenience for scripts and services that manage
          their own lifecycle.

        Args:
        - connection:        psycopg2 connection object (takes precedence).
        - connection_string: PostgreSQL DSN. Falls back to PGVECTOR_URL env var.
        """
        if connection is not None:
            self._conn = connection
            self._owns_connection = False
        else:
            if not _PGVECTOR_AVAILABLE:
                raise ImportError(
                    "pgvector package required. Install: pip install pgvector"
                )
            dsn = connection_string or os.environ.get("PGVECTOR_URL")
            if not dsn:
                raise ValueError(
                    "No database connection. Pass connection= or set PGVECTOR_URL."
                )
            self._conn = psycopg2.connect(dsn)
            _register_vector(self._conn)
            self._owns_connection = True

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        document_ids: list[str],
        texts: list[str],
        dense_vectors: list[list[float]],
        sparse_weights: list[dict],
        metadatas: list[dict],
    ) -> None:
        """
        What problem does this solve?
        - Persists chunks with their embeddings in a single batch upsert.
          Re-ingesting a document updates existing rows instead of duplicating.

        Why INSERT … ON CONFLICT DO UPDATE (upsert)?
        - Re-ingestion is expected (document content changes, re-processing after
          model upgrade). Duplicate chunk IDs must update, not error.

        Why batch with executemany and not individual inserts?
        - Reduces round-trips. 500 chunks = 1 network call vs 500 for individual inserts.

        Args:
        - chunk_ids:     List of DocumentChunk.id values.
        - document_ids:  List of DocumentChunk.document_id values.
        - texts:         List of chunk texts.
        - dense_vectors: List of 1024-dim float lists.
        - sparse_weights: List of {token: weight} dicts.
        - metadatas:     List of metadata dicts (RBAC + provenance).
        """
        import numpy as np

        sql = f"""
            INSERT INTO {DB_TABLE}
                (id, document_id, text, dense_vector, sparse_weights, metadata)
            VALUES
                (%(id)s, %(document_id)s, %(text)s, %(dense_vector)s,
                 %(sparse_weights)s::jsonb, %(metadata)s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                text           = EXCLUDED.text,
                dense_vector   = EXCLUDED.dense_vector,
                sparse_weights = EXCLUDED.sparse_weights,
                metadata       = EXCLUDED.metadata,
                updated_at     = NOW()
        """
        rows = [
            {
                "id":             chunk_ids[i],
                "document_id":    document_ids[i],
                "text":           texts[i],
                "dense_vector":   np.array(dense_vectors[i], dtype=np.float32),
                "sparse_weights": json.dumps(sparse_weights[i]),
                "metadata":       json.dumps(metadatas[i]),
            }
            for i in range(len(chunk_ids))
        ]
        with self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows, page_size=100)
        self._conn.commit()

    def search_dense(
        self,
        query_vector: list[float],
        rbac: RBACContext,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[SearchResult]:
        """
        What problem does this solve?
        - Finds the top_k chunks most similar to the query vector using cosine
          distance, scoped to the requesting user's RBAC context.

        Why cosine distance (<=>)?
        - BGE-M3 produces unit-norm vectors. Cosine similarity = dot product
          for unit-norm vectors. <=> gives distance (lower = more similar);
          we convert to score = 1 - distance.

        Why RBAC in SQL (not post-filter)?
        - Post-filtering would require fetching more than top_k candidates
          and hoping enough pass RBAC. SQL-level filtering uses the metadata
          GIN index and guarantees exactly top_k RBAC-valid results.

        Args:
        - query_vector: 1024-dim float list from BGEEncoder.encode_query().
        - rbac:         RBACContext for tenant + user filtering.
        - top_k:        Number of results to return.

        Returns list[SearchResult] sorted by score descending.
        """
        import numpy as np

        sql = f"""
            SELECT
                id,
                document_id,
                text,
                metadata,
                1 - (dense_vector <=> %(qvec)s::vector) AS score
            FROM {DB_TABLE}
            WHERE {_RBAC_WHERE}
              AND dense_vector IS NOT NULL
            ORDER BY dense_vector <=> %(qvec)s::vector
            LIMIT %(top_k)s
        """
        params = {
            "qvec":       np.array(query_vector, dtype=np.float32),
            "tenant_id":  rbac.tenant_id,
            "user_id":    rbac.user_id,
            "user_roles": rbac.user_roles,
            "top_k":      top_k,
        }
        with self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            SearchResult(
                chunk_id=row["id"],
                document_id=row["document_id"],
                text=row["text"],
                score=float(row["score"]),
                metadata=dict(row["metadata"]),
            )
            for row in rows
        ]

    def search_keyword(
        self,
        query_text: str,
        rbac: RBACContext,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[SearchResult]:
        """
        What problem does this solve?
        - Finds chunks that contain the query's keywords, even if the semantic
          meaning differs (e.g. searching for a product SKU or exact clause number).
          Dense search misses exact-match cases; keyword search catches them.

        Why PostgreSQL FTS (tsvector/tsquery)?
        - Built into PostgreSQL, no extra service. GIN index gives sub-10ms
          queries at 500K documents. ts_rank produces BM25-like scoring.
        - plainto_tsquery handles raw user input (no special query syntax needed).

        Why ts_rank_cd over ts_rank?
        - ts_rank_cd (cover density) weights chunks where query terms appear
          close together more highly — better for sentence-level chunks.

        Args:
        - query_text: Raw user query string.
        - rbac:       RBACContext for tenant + user filtering.
        - top_k:      Number of results to return.

        Returns list[SearchResult] sorted by BM25-style score descending.
        """
        sql = f"""
            SELECT
                id,
                document_id,
                text,
                metadata,
                ts_rank_cd(
                    to_tsvector('english', text),
                    plainto_tsquery('english', %(query)s)
                ) AS score
            FROM {DB_TABLE}
            WHERE {_RBAC_WHERE}
              AND to_tsvector('english', text) @@ plainto_tsquery('english', %(query)s)
            ORDER BY score DESC
            LIMIT %(top_k)s
        """
        params = {
            "query":      query_text,
            "tenant_id":  rbac.tenant_id,
            "user_id":    rbac.user_id,
            "user_roles": rbac.user_roles,
            "top_k":      top_k,
        }
        with self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            SearchResult(
                chunk_id=row["id"],
                document_id=row["document_id"],
                text=row["text"],
                score=float(row["score"]),
                metadata=dict(row["metadata"]),
            )
            for row in rows
        ]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[SearchResult]:
        """
        What problem does this solve?
        - Sentence window expansion: after identifying the top chunks,
          the retrieval pipeline fetches prev_chunk_id and next_chunk_id
          neighbours to give the LLM wider context.

        Why no RBAC filter here?
        - Window chunks are fetched by explicit IDs that were already returned
          by an RBAC-filtered search. If the user can see chunk[i], they can
          see its neighbours in the same document.

        Args:
        - chunk_ids: List of chunk IDs to fetch.

        Returns list[SearchResult] (score=0.0 — these are context, not retrieved).
        """
        if not chunk_ids:
            return []

        sql = f"""
            SELECT id, document_id, text, metadata
            FROM {DB_TABLE}
            WHERE id = ANY(%(ids)s)
        """
        with self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, {"ids": chunk_ids})
            rows = cur.fetchall()

        return [
            SearchResult(
                chunk_id=row["id"],
                document_id=row["document_id"],
                text=row["text"],
                score=0.0,
                metadata=dict(row["metadata"]),
            )
            for row in rows
        ]

    def close(self) -> None:
        """Close the database connection if we own it (created from connection_string)."""
        if self._owns_connection and self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
