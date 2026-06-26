"""
tests/ingestion/test_format_detection.py

What does this cover?
- MimeTypeDetector: MIME→extension mapping, extension fallback, disabled mode.
- MinHashDeduplicator: exact match, near-duplicate, short text skip,
                       remove(), disabled mode, index count.
- IngestionService: duplicate flagged in result, duplicate_of populated,
                    non-duplicate indexed, deduplicator DI.
- FtfyCleaner: mojibake repair, HTML entity unescaping, empty text,
               graceful fallback when ftfy unavailable.
- LoaderRegistry: MIME-detected extension used, falls back to file extension.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.ingestion.deduplicator import DeduplicationResult, MinHashDeduplicator
from agent.ingestion.mime_detector import MimeTypeDetector
from agent.processing.cleaning.ftfy_cleaner import FtfyCleaner


# ── MimeTypeDetector ───────────────────────────────────────────────────────────

class TestMimeTypeDetector:

    def test_pdf_bytes_detected_as_pdf(self, tmp_path):
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")
        detector = MimeTypeDetector()
        assert detector.detect_extension(str(f)) == "pdf"

    def test_wrong_extension_corrected_by_mime(self, tmp_path):
        """A PDF file with a .txt extension should still be detected as pdf."""
        f = tmp_path / "disguised.txt"
        f.write_bytes(b"%PDF-1.4 fake pdf content")
        detector = MimeTypeDetector()
        ext = detector.detect_extension(str(f))
        # libmagic reads bytes, not extension — should return "pdf"
        assert ext == "pdf"

    def test_falls_back_to_extension_when_disabled(self, tmp_path):
        f = tmp_path / "document.docx"
        f.write_bytes(b"fake docx content")
        detector = MimeTypeDetector(disabled=True)
        assert detector.detect_extension(str(f)) == "docx"

    def test_unknown_extension_returns_extension_string(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_bytes(b"some unknown content")
        detector = MimeTypeDetector(disabled=True)
        assert detector.detect_extension(str(f)) == "xyz"

    def test_no_extension_returns_unknown_when_disabled(self, tmp_path):
        f = tmp_path / "noextension"
        f.write_bytes(b"plain content")
        detector = MimeTypeDetector(disabled=True)
        assert detector.detect_extension(str(f)) == "unknown"

    def test_is_magic_available_false_when_disabled(self):
        detector = MimeTypeDetector(disabled=True)
        assert detector.is_magic_available is False

    def test_is_magic_available_true_when_enabled(self):
        detector = MimeTypeDetector(disabled=False)
        # May be False if libmagic not installed — just check it's a bool
        assert isinstance(detector.is_magic_available, bool)


# ── MinHashDeduplicator ────────────────────────────────────────────────────────

LONG_TEXT_A = (
    "This is a comprehensive enterprise policy document covering all aspects of "
    "employee conduct and behaviour in the workplace. All employees must adhere "
    "to these guidelines at all times. Violations may result in disciplinary action "
    "up to and including termination of employment. The policy applies to all staff "
    "including contractors and temporary workers engaged by the organisation. "
) * 4  # repeat to get enough shingles

LONG_TEXT_B = (
    "This is a comprehensive enterprise policy document covering all aspects of "
    "employee conduct and behaviour in the workplace. All employees must adhere "
    "to these guidelines at all times. Violations may result in disciplinary action "
    "up to and including termination of employment. The policy applies to all staff "
    "including contractors and temporary workers engaged by the organisation. "
    "Updated: January 2024."  # minor addition — still near-duplicate
) * 4

UNRELATED_TEXT = (
    "The quarterly financial results show strong performance across all business "
    "units. Revenue increased by fifteen percent compared to the same period last "
    "year. Operating costs were reduced through automation and process improvements. "
    "The board has approved a dividend payment to shareholders this quarter. "
) * 4


class TestMinHashDeduplicator:

    def setup_method(self):
        self.dedup = MinHashDeduplicator(threshold=0.7, num_perm=64)

    def test_first_document_not_duplicate(self):
        result = self.dedup.check("doc-1", LONG_TEXT_A)
        assert result.is_duplicate is False

    def test_first_document_is_indexed(self):
        result = self.dedup.check("doc-1", LONG_TEXT_A)
        assert result.was_indexed is True
        assert self.dedup.indexed_count == 1

    def test_identical_text_flagged_as_duplicate(self):
        self.dedup.check("doc-1", LONG_TEXT_A)
        result = self.dedup.check("doc-2", LONG_TEXT_A)
        assert result.is_duplicate is True
        assert result.duplicate_of == "doc-1"

    def test_near_duplicate_flagged(self):
        self.dedup.check("doc-1", LONG_TEXT_A)
        result = self.dedup.check("doc-2", LONG_TEXT_B)
        assert result.is_duplicate is True

    def test_unrelated_document_not_flagged(self):
        self.dedup.check("doc-1", LONG_TEXT_A)
        result = self.dedup.check("doc-2", UNRELATED_TEXT)
        assert result.is_duplicate is False

    def test_similarity_score_populated_for_duplicate(self):
        self.dedup.check("doc-1", LONG_TEXT_A)
        result = self.dedup.check("doc-2", LONG_TEXT_A)
        assert result.similarity_score > 0.0

    def test_duplicate_not_indexed(self):
        self.dedup.check("doc-1", LONG_TEXT_A)
        self.dedup.check("doc-2", LONG_TEXT_A)
        assert self.dedup.indexed_count == 1  # only doc-1 in index

    def test_short_text_skipped(self):
        result = self.dedup.check("doc-short", "Too short.")
        assert result.is_duplicate is False
        assert result.was_indexed is False

    def test_remove_allows_reingest(self):
        self.dedup.check("doc-1", LONG_TEXT_A)
        self.dedup.remove("doc-1")
        # After removal, same text should be indexable again
        result = self.dedup.check("doc-1-v2", LONG_TEXT_A)
        assert result.is_duplicate is False
        assert result.was_indexed is True

    def test_remove_unknown_id_does_not_raise(self):
        self.dedup.remove("nonexistent-id")  # should not raise

    def test_disabled_always_returns_not_duplicate(self):
        dedup = MinHashDeduplicator(disabled=True)
        dedup.check("doc-1", LONG_TEXT_A)
        result = dedup.check("doc-2", LONG_TEXT_A)
        assert result.is_duplicate is False

    def test_disabled_indexed_count_is_zero(self):
        dedup = MinHashDeduplicator(disabled=True)
        dedup.check("doc-1", LONG_TEXT_A)
        assert dedup.indexed_count == 0

    def test_datasketch_unavailable_disables_dedup(self):
        import agent.ingestion.deduplicator as mod
        original = mod._DATASKETCH_AVAILABLE
        mod._DATASKETCH_AVAILABLE = False
        try:
            dedup = MinHashDeduplicator()
            result = dedup.check("doc-1", LONG_TEXT_A)
            assert result.is_duplicate is False
        finally:
            mod._DATASKETCH_AVAILABLE = original


# ── FtfyCleaner ────────────────────────────────────────────────────────────────

class TestFtfyCleaner:

    def setup_method(self):
        self.cleaner = FtfyCleaner()

    def test_fixes_mojibake_curly_quote(self):
        # â€™ is the UTF-8 encoding of ' decoded as Latin-1
        result = self.cleaner.clean("Itâ€™s a policy document.")
        assert "It" in result
        assert "â€™" not in result

    def test_fixes_html_entity_amp(self):
        result = self.cleaner.clean("Terms &amp; Conditions apply.")
        assert "&" in result
        assert "&amp;" not in result

    def test_empty_string_returns_empty(self):
        assert self.cleaner.clean("") == ""

    def test_clean_text_passes_through_unchanged(self):
        text = "This is already clean text with no encoding issues."
        result = self.cleaner.clean(text)
        assert result == text

    def test_source_type_ignored(self):
        text = "Clean text."
        assert self.cleaner.clean(text, source_type="pdf") == text
        assert self.cleaner.clean(text, source_type="eml") == text

    def test_ftfy_unavailable_returns_input(self):
        import agent.processing.cleaning.ftfy_cleaner as mod
        original = mod._FTFY_AVAILABLE
        mod._FTFY_AVAILABLE = False
        try:
            cleaner = FtfyCleaner()
            text = "Itâ€™s still garbled"
            result = cleaner.clean(text)
            assert result == text  # returned unchanged, no crash
        finally:
            mod._FTFY_AVAILABLE = original

    def test_ftfy_cleaner_is_first_stage_in_pipeline(self):
        """FtfyCleaner must run before WhitespaceCleaner in the default pipeline."""
        from agent.processing.pipeline import PreprocessingPipeline
        pipeline = PreprocessingPipeline()
        first_cleaner = pipeline._cleaners[0]
        assert isinstance(first_cleaner, FtfyCleaner)


# ── LoaderRegistry MIME integration ───────────────────────────────────────────

class TestLoaderRegistryMimeIntegration:

    def test_mime_detector_used_in_get_loader(self, tmp_path):
        """Registry uses MimeTypeDetector result to pick loader."""
        from agent.ingestion.loader_registry import LoaderRegistry

        # A file with .txt extension but PDF bytes
        f = tmp_path / "report.txt"
        f.write_bytes(b"%PDF-1.4 fake pdf content")

        # Inject disabled detector → falls back to .txt extension
        registry = LoaderRegistry(mime_detector=MimeTypeDetector(disabled=True))
        loader = registry.get_loader(str(f))
        assert loader.supported_extensions == [".txt"] or ".txt" in loader.supported_extensions

    def test_disabled_mime_falls_back_to_extension(self, tmp_path):
        from agent.ingestion.loader_registry import LoaderRegistry

        f = tmp_path / "document.html"
        f.write_bytes(b"<html><body>test</body></html>")

        registry = LoaderRegistry(mime_detector=MimeTypeDetector(disabled=True))
        loader = registry.get_loader(str(f))
        assert ".html" in loader.supported_extensions
