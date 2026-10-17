"""Ingestion module: load files into Documents and detect near-duplicates."""

from agent.ingestion.models import Document, DocumentChunk
from agent.ingestion.service import IngestionResult, IngestionService

__all__ = [
    "Document",
    "DocumentChunk",
    "IngestionResult",
    "IngestionService",
]
