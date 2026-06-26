"""
processing/pipeline.py

What problem does this solve?
- Three independent stages (cleaning, normalization, metadata extraction)
  need to run in the correct order on every document. Without an orchestrator,
  every caller would have to instantiate and sequence the stages manually.

Why a pipeline class instead of a single large function?
- Each stage is independently testable and replaceable.
- New stages (e.g. PII redaction, language translation) can be inserted
  without changing callers — only add to the relevant list.
- processing_stats records which stages ran and their impact, giving
  observability without adding logging calls everywhere.

Stage order is deliberate:
  1. FtfyCleaner        — repair mojibake first (â€™ → ') so all subsequent
                          stages operate on valid Unicode, not garbled bytes.
  2. WhitespaceCleaner  — normalise whitespace so regex patterns have clean
                          line boundaries to match against.
  3. NoiseCleaner       — remove structural boilerplate (page numbers, TOC).
  4. UnicodeNormalizer  — replace typographic variants with ASCII equivalents.
  5. HyphenNormalizer   — repair PDF line-break hyphens AFTER unicode is clean.
  6. MetadataExtractor  — extract structured data from the now-clean text.
"""

from agent.ingestion.models import Document
from agent.processing.cleaning.ftfy_cleaner import FtfyCleaner
from agent.processing.cleaning.noise_cleaner import NoiseCleaner
from agent.processing.cleaning.whitespace_cleaner import WhitespaceCleaner
from agent.processing.metadata.extractor import MetadataExtractor
from agent.processing.models import ProcessedDocument
from agent.processing.normalization.hyphen_normalizer import HyphenNormalizer
from agent.processing.normalization.unicode_normalizer import UnicodeNormalizer


class PreprocessingPipeline:
    """
    What problem does this solve?
    - Orchestrates all preprocessing stages in the correct order and
      produces a ProcessedDocument ready for ChunkingService.

    Why does this class exist?
    - Single place to configure which cleaners and normalizers are active.
    - Dependency injection: tests or deployments can inject custom stages
      without subclassing or patching.

    Why accept cleaners/normalizers/extractor as constructor args?
    - Allows tests to inject mocks or minimal stage lists for focused testing.
    - Allows production deployments to add stages (e.g. PII redactor) by
      passing an extended list — no code changes needed.
    """

    def __init__(
        self,
        cleaners=None,
        normalizers=None,
        extractor: MetadataExtractor | None = None,
    ) -> None:
        """
        Why default lists defined in __init__ instead of class body?
        - Avoids mutable default argument pitfall in Python.
        - Instantiates fresh objects per pipeline instance,
          preventing shared state between multiple pipeline instances in tests.
        """
        self._cleaners = cleaners or [
            FtfyCleaner(),         # first: repair mojibake / encoding corruption
            WhitespaceCleaner(),   # second: normalise whitespace so regex patterns work
            NoiseCleaner(),        # third: remove structural noise (page numbers, watermarks)
        ]
        self._normalizers = normalizers or [
            UnicodeNormalizer(),   # first: ASCII equivalents for consistent tokenisation
            HyphenNormalizer(),    # second: repair line-break hyphens after unicode is clean
        ]
        self._extractor = extractor or MetadataExtractor()

    def run(self, document: Document) -> ProcessedDocument:
        """
        What problem does this solve?
        - Transforms a raw Document into a ProcessedDocument with clean text
          and structured metadata, ready for ChunkingService.

        Why record char counts before and after each stage?
        - processing_stats exposes how much each stage reduces text size.
          Useful for tuning: if NoiseCleaner removes 40% of text from a
          policy doc, the patterns may be too aggressive for that source_type.

        Why pass source_type to cleaners?
        - Format-specific rules (PDF page numbers, email headers) only apply
          to the correct document type.

        Args:
        - document: Raw Document from IngestionService.

        Returns ProcessedDocument with cleaned_text and extracted_metadata.
        """
        text = document.text
        original_char_count = len(text)
        stage_stats: dict[str, int] = {"original": original_char_count}

        # ── Stage 1: Cleaning ──────────────────────────────────────────────
        for cleaner in self._cleaners:
            text = cleaner.clean(text, source_type=document.source_type)
            stage_stats[type(cleaner).__name__] = len(text)

        # ── Stage 2: Normalisation ─────────────────────────────────────────
        for normalizer in self._normalizers:
            text = normalizer.normalize(text)
            stage_stats[type(normalizer).__name__] = len(text)

        # ── Stage 3: Metadata extraction ──────────────────────────────────
        extracted_metadata = self._extractor.extract(text, document)

        # ── Processing stats for observability ────────────────────────────
        final_char_count = len(text)
        chars_removed = original_char_count - final_char_count
        reduction_pct = (
            round(chars_removed / original_char_count * 100, 2)
            if original_char_count > 0 else 0.0
        )

        processing_stats = {
            **stage_stats,
            "final": final_char_count,
            "chars_removed": chars_removed,
            "reduction_pct": reduction_pct,
            "stages_applied": [type(s).__name__ for s in self._cleaners + self._normalizers],
        }

        return ProcessedDocument(
            document=document,
            cleaned_text=text,
            extracted_metadata=extracted_metadata,
            processing_stats=processing_stats,
        )
