"""
retrieval/config.py

What problem does this solve?
- The reranker model and the final top-k are retrieval decisions. They lived
  in embeddings/config.py, which made the embeddings module carry settings
  only the retrieval module reads.

Why is RETRIEVAL_TOP_K not here?
- It is the default candidate count of PgVectorStore search, which the
  embeddings module owns. Retrieval reads it from agent.embeddings.
"""

# ── Reranker model ─────────────────────────────────────────────────────────────
# bge-reranker-v2-m3: cross-encoder reranker from the same BAAI family as BGE-M3.
# Runs fully local — enterprise documents never leave the machine.
# License: MIT | Source: https://huggingface.co/BAAI/bge-reranker-v2-m3
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Final chunks returned to the LLM after reranking.
# 5 × ~512 tokens ≈ 2560 tokens — fits comfortably in any LLM context window.
RERANK_TOP_K = 5
