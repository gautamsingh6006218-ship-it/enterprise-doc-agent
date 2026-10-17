"""Embeddings module: encode chunks with BGE-M3 and own the document_chunks table."""

from agent.embeddings.bge_encoder import BGEEncoder
from agent.embeddings.config import RETRIEVAL_TOP_K
from agent.embeddings.service import EmbeddingResult, EmbeddingService
from agent.embeddings.store import PgVectorStore

__all__ = [
    "BGEEncoder",
    "RETRIEVAL_TOP_K",
    "EmbeddingResult",
    "EmbeddingService",
    "PgVectorStore",
]
