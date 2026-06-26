"""
embeddings/config.py

What problem does this solve?
- Embedding model names, dimensions, and batch sizes are referenced by the
  encoder, store, and tests. Without a central config, changing the model
  requires hunting through multiple files.

Why constants and not environment variables?
- Model choice and vector dimensions are architectural decisions — they change
  only when switching embedding models. Env vars are for infrastructure values
  (DB URLs, ports, API keys) that differ per deployment environment.
- Changing BGE_M3_MODEL here automatically updates encoder, store schema,
  and all downstream consumers.
"""

# ── Embedding model ────────────────────────────────────────────────────────────
# BGE-M3: single model that produces dense (1024-dim) AND sparse (lexical) vectors.
# This eliminates the need for a separate BM25 index while enabling hybrid retrieval.
# License: MIT | Source: https://huggingface.co/BAAI/bge-m3
BGE_M3_MODEL = "BAAI/bge-m3"

# ── Reranker model ─────────────────────────────────────────────────────────────
# bge-reranker-v2-m3: cross-encoder reranker from the same BAAI family as BGE-M3.
# Runs fully local — enterprise documents never leave the machine.
# License: MIT | Source: https://huggingface.co/BAAI/bge-reranker-v2-m3
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ── Vector dimensions ──────────────────────────────────────────────────────────
# BGE-M3 dense output: 1024 dimensions.
# This value must match the vector(1024) column in the database schema.
# If you switch models, update both this constant AND re-run setup_db.sql.
DENSE_VECTOR_DIM = 1024

# ── Batching ───────────────────────────────────────────────────────────────────
# 32 chunks per encode call — balance between GPU memory (~2GB fp16) and speed.
# Reduce to 8-16 if running on CPU or low-memory GPU.
EMBEDDING_BATCH_SIZE = 32

# ── Database ───────────────────────────────────────────────────────────────────
# Table that stores chunks, dense vectors, sparse weights, and metadata.
DB_TABLE = "document_chunks"

# ── Retrieval limits ───────────────────────────────────────────────────────────
# How many candidates each retrieval path returns before RRF fusion.
# Higher = better recall, more reranking cost. 50 is the standard production value.
RETRIEVAL_TOP_K = 50

# Final chunks returned to the LLM after reranking.
# 5 × ~512 tokens ≈ 2560 tokens — fits comfortably in any LLM context window.
RERANK_TOP_K = 5
