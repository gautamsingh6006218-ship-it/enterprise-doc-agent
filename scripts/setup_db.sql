-- setup_db.sql
--
-- What problem does this solve?
-- Creates the schema for the enterprise RAG vector store + document registry.
-- Runs automatically on first docker compose up via docker-entrypoint-initdb.d.
-- Re-runnable: all statements use IF NOT EXISTS / OR REPLACE.
--
-- Tables:
--   document_chunks  — stores chunk text, dense vectors, sparse weights, metadata
--   documents        — pipeline run registry (status, timing, dedup, error tracking)
--
-- Indexes:
--   HNSW on dense_vector          — O(log n) approximate nearest neighbour search
--   GIN  on text (tsvector)       — full-text search for keyword/BM25 retrieval
--   GIN  on metadata              — JSONB field access for RBAC filtering
--   BTREE on document_id          — fast lookup for sentence window expansion
--   BTREE on documents.tenant_id  — list all docs for a tenant
--   BTREE on documents (tenant_id, status) — filter by status for dashboards
--   BTREE on documents.file_hash  — exact-duplicate check

-- ── Extension ──────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Chunks table ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_chunks (
    -- Identity
    id              TEXT        PRIMARY KEY,
    document_id     TEXT        NOT NULL,

    -- Content
    text            TEXT        NOT NULL,

    -- BGE-M3 dense embedding (1024-dim, cosine similarity)
    -- NULL until EmbeddingService processes the chunk
    dense_vector    vector(1024),

    -- BGE-M3 sparse lexical weights: { "token": weight, ... }
    -- Stored for future sparse vector search via pgvector sparsevec
    sparse_weights  JSONB       DEFAULT '{}',

    -- Full chunk metadata: RBAC fields, source_type, token_count, prev/next IDs
    -- Stored as JSONB so new metadata fields don't require schema migrations
    metadata        JSONB       NOT NULL DEFAULT '{}',

    -- Audit
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Indexes ────────────────────────────────────────────────────────────────────

-- HNSW index: approximate nearest neighbour for dense vector search
-- m=16 (max connections per node), ef_construction=64 (build quality)
-- cosine distance (<=>): correct for BGE-M3 which produces unit-norm vectors
-- Why HNSW over IVFFlat?
--   HNSW has better recall at equivalent speed and does not require a training
--   pass (IVFFlat requires CLUSTER ON after large inserts). For ≤2M vectors,
--   HNSW is the production default.
CREATE INDEX IF NOT EXISTS idx_chunks_dense_hnsw
    ON document_chunks
    USING hnsw (dense_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index on tsvector: enables fast full-text (BM25-style) keyword search
-- 'english' config: stemming + stop-word removal
-- Why functional GIN and not a stored tsvector column?
--   Simpler schema — no extra column to keep in sync on updates.
CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON document_chunks
    USING gin (to_tsvector('english', text));

-- GIN index on metadata JSONB: fast access to RBAC fields in WHERE clauses
-- Required for: metadata->>'tenant_id' = $1 to use index (not seq scan)
CREATE INDEX IF NOT EXISTS idx_chunks_metadata
    ON document_chunks
    USING gin (metadata);

-- BTREE index on document_id: fast lookup for sentence window expansion
-- (fetch all chunks belonging to a document by prev_chunk_id / next_chunk_id)
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON document_chunks (document_id);

-- ── Documents registry table ──────────────────────────────────────────────────
-- What problem does this solve?
--   Tracks every pipeline run result: which files were ingested, their status
--   (completed / duplicate / failed_*), how many chunks were stored, and
--   any failure details. Enables: exact-duplicate checks, operator dashboards,
--   selective retries (resubmit only failed_embedding docs), and audit trails.
--
-- Why a separate table from document_chunks?
--   document_chunks stores content (one row per chunk, potentially 1000s per doc).
--   documents stores metadata (one row per document). Joining them for status
--   queries would be expensive and conceptually wrong.
CREATE TABLE IF NOT EXISTS documents (
    -- Identity
    id                  TEXT        PRIMARY KEY,

    -- Source
    file_path           TEXT        NOT NULL,
    file_hash           TEXT        NOT NULL DEFAULT '',   -- SHA-256 for exact dedup

    -- Pipeline status: 'completed' | 'duplicate' | 'failed_ingestion' |
    --   'failed_preprocessing' | 'failed_chunking' | 'failed_embedding' | 'pending'
    status              TEXT        NOT NULL DEFAULT 'pending',

    -- RBAC
    tenant_id           TEXT        NOT NULL,
    owner_id            TEXT        NOT NULL,

    -- Results
    total_chunks        INT         NOT NULL DEFAULT 0,
    failed_stage        TEXT,                              -- NULL on success
    error               TEXT,                              -- NULL on success

    -- Near-dedup
    is_duplicate        BOOLEAN     NOT NULL DEFAULT FALSE,
    duplicate_of        TEXT,                              -- document_id of the match
    similarity_score    FLOAT       NOT NULL DEFAULT 0.0,

    -- Timing
    total_duration_ms   FLOAT       NOT NULL DEFAULT 0.0,

    -- Audit
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- BTREE index: list all documents for a tenant (most common API query)
CREATE INDEX IF NOT EXISTS idx_documents_tenant_id
    ON documents (tenant_id);

-- Composite index: filter by tenant + status for operator dashboards
--   e.g., WHERE tenant_id = $1 AND status = 'failed_embedding'
CREATE INDEX IF NOT EXISTS idx_documents_tenant_status
    ON documents (tenant_id, status);

-- BTREE index: exact-duplicate check by SHA-256 hash within a tenant
CREATE INDEX IF NOT EXISTS idx_documents_file_hash
    ON documents (file_hash, tenant_id);

-- ── Auto-update updated_at ─────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_updated_at ON document_chunks;
CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON document_chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS set_updated_at_documents ON documents;
CREATE TRIGGER set_updated_at_documents
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
