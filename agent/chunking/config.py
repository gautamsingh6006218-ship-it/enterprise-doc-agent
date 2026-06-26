"""
chunking/config.py

What problem does this solve?
- Chunking parameters are referenced across token_counter, strategies,
  and pipeline. Without a central config, changing the chunk size requires
  hunting through multiple files.

Why a config module instead of env vars or a settings class?
- These are architectural constants, not deployment-specific values.
  They change only when the embedding model changes (e.g. switching from
  nomic-embed-text to a 4096-token model).
- Env vars are for secrets and environment-specific values (DB URLs, ports).
  Token budgets are model-architecture decisions — they belong in code.

How to change chunk size for a different embedding model:
- Update CHUNK_TARGET_TOKENS and CHUNK_MAX_TOKENS here.
- All strategies and the pipeline pick up the new values automatically.
"""

# ── Tiktoken encoding ──────────────────────────────────────────────────────────
# cl100k_base is used by GPT-4 and text-embedding-3 models.
# 85-90% accurate for nomic-embed-text (BERT-based).
# Swap to "p50k_base" for older GPT-3 models if needed.
TIKTOKEN_ENCODING = "cl100k_base"

# ── Chunk size targets ─────────────────────────────────────────────────────────
# nomic-embed-text max context: 8192 tokens
# 512 tokens = optimal retrieval precision sweet spot (empirically validated)
# 1024 tokens = hard ceiling — never produce a chunk above this
CHUNK_TARGET_TOKENS = 512
CHUNK_MAX_TOKENS = 1024

# ── Overlap ───────────────────────────────────────────────────────────────────
# 10% of target = 51 tokens
# Overlap ensures context at chunk boundaries is not lost.
# e.g. "The contract signed on..." at end of chunk N appears at start of N+1.
CHUNK_OVERLAP_TOKENS = 51  # ~10% of 512

# ── Sentence window retrieval ─────────────────────────────────────────────────
# At retrieval time, return this many neighboring chunks on each side.
# Window of 2 = chunk[i-2], chunk[i-1], chunk[i], chunk[i+1], chunk[i+2]
# = ~5 × 512 = ~2560 tokens of context sent to the LLM.
SENTENCE_WINDOW_SIZE = 2

# ── Strategy routing by source_type ───────────────────────────────────────────
# Maps document source_type → chunking strategy class name.
# ChunkingRouter reads this to select the strategy without if/elif chains.
STRATEGY_MAP: dict[str, str] = {
    # Prose documents — sentence boundary + window context
    "pdf":        "SentenceWindowStrategy",
    "docx":       "SentenceWindowStrategy",
    "txt":        "SentenceWindowStrategy",
    "eml":        "SentenceWindowStrategy",
    "msg":        "SentenceWindowStrategy",
    "pptx":       "SentenceWindowStrategy",
    "rtf":        "SentenceWindowStrategy",
    "epub":       "SentenceWindowStrategy",
    "odt":        "SentenceWindowStrategy",

    # Structured text — header-aware recursive splitting
    "markdown":   "StructureAwareStrategy",
    "html":       "StructureAwareStrategy",

    # Tabular data — hard token budget (no sentence boundaries in tables)
    "xlsx":       "TokenStrategy",
    "csv":        "TokenStrategy",

    # Default fallback for unknown source types
    "unknown":    "SentenceWindowStrategy",
}
