"""
processing/models.py

What problem does this solve?
- After ingestion, a Document holds raw extracted text that is noisy,
  un-normalised, and carries no structured metadata. Downstream services
  (ChunkingService, EmbeddingService, VectorStore) need clean text and
  rich metadata to produce high-quality retrieval results.

Why a separate ProcessedDocument instead of modifying Document in-place?
- Preserves the original extracted text for audit and re-processing.
- Makes the processing pipeline idempotent: running it twice on the same
  Document always produces the same ProcessedDocument.
- Keeps ingestion models (models.py) decoupled from processing concerns.

Where does ProcessedDocument fit?
  IngestionService → Document → PreprocessingService → ProcessedDocument
  → ChunkingService → DocumentChunk (inherits cleaned_text as its text)
"""

from dataclasses import dataclass, field
from typing import Any

from agent.ingestion.models import Document


@dataclass
class ExtractedMetadata:
    """
    What problem does this solve?
    - Raw document text contains structured information (dates, emails, entities)
      scattered as plain text. Extracting them into typed fields allows the
      vector store to filter on them (e.g. "show docs from 2024", "docs in French").

    Why does this class exist?
    - Centralises all extracted structured fields in one typed object.
    - Passed into every chunk's metadata so vector store filters work at
      query time without a separate metadata DB lookup.

    Field rationale:
    - language:            ISO 639-1 code ("en", "de"). Drives multilingual
                           embedding model selection in EmbeddingService.
    - category:            Rule-based document type ("contract", "invoice").
                           Enables category-scoped retrieval ("find all policies").
    - dates:               All dates found in the document. Enables time-range
                           filtering ("show Q3 2024 reports only").
    - emails:              Extracted email addresses. Useful for CRM integration.
    - phone_numbers:       Extracted phone numbers. Useful for contact extraction.
    - urls:                Extracted URLs. Useful for link auditing.
    - section_headers:     Document section titles. Used by header-aware chunker
                           in Phase 2 to chunk on section boundaries.
    - word_count:          Used for reading time estimates in UI.
    - char_count:          Post-cleaning character count for observability.
    - reading_time_minutes: Estimated reading time (word_count / 200 wpm).
    """

    language: str = "unknown"
    category: str = "unknown"
    dates: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    phone_numbers: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    section_headers: list[str] = field(default_factory=list)
    word_count: int = 0
    char_count: int = 0
    reading_time_minutes: float = 0.0


@dataclass
class ProcessedDocument:
    """
    What problem does this solve?
    - ChunkingService and EmbeddingService need clean text, not raw extracted text.
    - Vector store needs structured metadata fields (language, dates, category)
      alongside text vectors for filtered semantic search.

    Why does this class exist?
    - Output contract of PreprocessingService. Everything downstream consumes
      this instead of the raw Document.
    - cleaned_text is what gets chunked and embedded — not document.text.
    - document (original) is preserved for audit trail and re-processing.

    Field rationale:
    - document:           Original Document from IngestionService. Preserved
                          as-is so the raw extraction is never lost.
    - cleaned_text:       Text after all cleaning + normalization stages.
                          This is what ChunkingService splits and EmbeddingService
                          vectorises. Always prefer this over document.text.
    - extracted_metadata: Structured fields extracted from cleaned_text.
                          Merged into every chunk's metadata at chunking time.
    - processing_stats:   Observability: chars reduced, stages applied, etc.
                          Helps tune cleaning aggressiveness over time.
    """

    document: Document
    cleaned_text: str
    extracted_metadata: ExtractedMetadata
    processing_stats: dict[str, Any] = field(default_factory=dict)
