"""
tests/processing/test_preprocessing_service.py

What does this cover?
- WhitespaceCleaner: control chars, trailing spaces, excess blank lines.
- NoiseCleaner: page numbers, watermarks, TOC entries, email headers.
- UnicodeNormalizer: curly quotes, dashes, bullets, zero-width chars.
- HyphenNormalizer: PDF line-break hyphen repair.
- MetadataExtractor: dates, emails, phones, URLs, headers, language, category.
- PreprocessingPipeline: stage sequencing, stats, empty text handling.
- PreprocessingService: result wrapping, error handling, dependency injection.
"""

from pathlib import Path

import pytest

from agent.ingestion.models import Document
from agent.processing.cleaning.noise_cleaner import NoiseCleaner
from agent.processing.cleaning.whitespace_cleaner import WhitespaceCleaner
from agent.processing.metadata.extractor import MetadataExtractor
from agent.processing.models import ProcessedDocument
from agent.processing.normalization.hyphen_normalizer import HyphenNormalizer
from agent.processing.normalization.unicode_normalizer import UnicodeNormalizer
from agent.processing.pipeline import PreprocessingPipeline
from agent.services.preprocessing_service import PreprocessingResult, PreprocessingService


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_document(text: str, source_type: str = "txt", title: str = "test") -> Document:
    """Creates a minimal Document for testing pipeline stages."""
    return Document(
        id="test-id",
        source_type=source_type,
        source_path="/test/path",
        title=title,
        text=text,
    )


# ── WhitespaceCleaner ──────────────────────────────────────────────────────────

class TestWhitespaceCleaner:

    def setup_method(self):
        self.cleaner = WhitespaceCleaner()

    def test_removes_null_bytes(self):
        result = self.cleaner.clean("hello\x00world")
        assert "\x00" not in result
        assert "helloworld" in result

    def test_removes_control_characters(self):
        result = self.cleaner.clean("text\x07with\x1bcontrol")
        assert "\x07" not in result
        assert "\x1b" not in result

    def test_normalises_windows_line_endings(self):
        result = self.cleaner.clean("line1\r\nline2\r\nline3")
        assert "\r" not in result
        assert "line1\nline2\nline3" in result

    def test_replaces_tabs_with_spaces(self):
        result = self.cleaner.clean("col1\tcol2\tcol3")
        assert "\t" not in result
        assert "col1 col2 col3" in result

    def test_strips_trailing_spaces(self):
        result = self.cleaner.clean("line with spaces   \nanother line  ")
        for line in result.split("\n"):
            assert not line.endswith(" "), f"Trailing space in: '{line}'"

    def test_collapses_excess_blank_lines(self):
        text = "paragraph one\n\n\n\n\nparagraph two"
        result = self.cleaner.clean(text)
        assert "\n\n\n" not in result
        assert "paragraph one" in result
        assert "paragraph two" in result

    def test_preserves_double_newline_paragraph_boundary(self):
        text = "first paragraph\n\nsecond paragraph"
        result = self.cleaner.clean(text)
        assert "\n\n" in result


# ── NoiseCleaner ───────────────────────────────────────────────────────────────

class TestNoiseCleaner:

    def setup_method(self):
        self.cleaner = NoiseCleaner()

    def test_removes_page_number_explicit(self):
        text = "Real content\nPage 3 of 15\nMore content"
        result = self.cleaner.clean(text, source_type="pdf")
        assert "Page 3 of 15" not in result
        assert "Real content" in result

    def test_removes_page_number_decorated(self):
        text = "Some text\n— 7 —\nMore text"
        result = self.cleaner.clean(text, source_type="pdf")
        assert "— 7 —" not in result

    def test_removes_watermark_confidential(self):
        text = "Section 1\nCONFIDENTIAL\nThis is the content."
        result = self.cleaner.clean(text)
        assert "CONFIDENTIAL" not in result
        assert "This is the content." in result

    def test_keeps_inline_confidential(self):
        """'CONFIDENTIAL' mid-sentence must NOT be removed."""
        text = "This document is CONFIDENTIAL and proprietary."
        result = self.cleaner.clean(text)
        assert "CONFIDENTIAL" in result

    def test_removes_toc_entry_dots(self):
        text = "Introduction\nChapter 1 Overview .......... 5\nChapter text"
        result = self.cleaner.clean(text, source_type="pdf")
        assert ".......... 5" not in result

    def test_removes_email_headers(self):
        text = "From: alice@example.com\nTo: bob@example.com\nHello Bob"
        result = self.cleaner.clean(text, source_type="eml")
        assert "From: alice@example.com" not in result
        assert "Hello Bob" in result

    def test_does_not_remove_email_headers_from_pdf(self):
        """Email header patterns must NOT fire on non-email source types."""
        text = "From this date forward, the contract is valid.\nTo all parties."
        result = self.cleaner.clean(text, source_type="pdf")
        assert "From this date forward" in result


# ── UnicodeNormalizer ──────────────────────────────────────────────────────────

class TestUnicodeNormalizer:

    def setup_method(self):
        self.normalizer = UnicodeNormalizer()

    def test_normalises_curly_single_quotes(self):
        result = self.normalizer.normalize("it’s a test")
        assert "'" in result
        assert "’" not in result

    def test_normalises_curly_double_quotes(self):
        result = self.normalizer.normalize("“hello”")
        assert '"hello"' in result

    def test_normalises_em_dash(self):
        result = self.normalizer.normalize("word—another")
        assert "-" in result
        assert "—" not in result

    def test_normalises_bullet_point(self):
        result = self.normalizer.normalize("• Item one")
        assert "- Item one" in result

    def test_removes_zero_width_space(self):
        result = self.normalizer.normalize("hello​world")
        assert "​" not in result
        assert "helloworld" in result

    def test_removes_bom(self):
        result = self.normalizer.normalize("﻿Document start")
        assert "﻿" not in result
        assert "Document start" in result

    def test_expands_ellipsis(self):
        result = self.normalizer.normalize("and so on…")
        assert "..." in result
        assert "…" not in result

    def test_preserves_valid_unicode_content(self):
        """French/German characters must not be stripped."""
        text = "Bonjour, café, naïve, über"
        result = self.normalizer.normalize(text)
        assert "café" in result
        assert "über" in result


# ── HyphenNormalizer ───────────────────────────────────────────────────────────

class TestHyphenNormalizer:

    def setup_method(self):
        self.normalizer = HyphenNormalizer()

    def test_joins_broken_word(self):
        result = self.normalizer.normalize("infor-\nmation system")
        assert "information" in result

    def test_keeps_hyphen_for_compound_prefix(self):
        result = self.normalizer.normalize("self-\nsufficient approach")
        assert "self-sufficient" in result

    def test_keeps_hyphen_for_multi_prefix(self):
        result = self.normalizer.normalize("multi-\nlingual support")
        assert "multi-lingual" in result

    def test_does_not_join_uppercase_continuation(self):
        """Uppercase after newline = new sentence, not line-break continuation."""
        text = "See chapter-\nThree for details"
        result = self.normalizer.normalize(text)
        # Should NOT join since "Three" starts with uppercase
        assert "chapter-\nThree" in result or "chapter-" in result


# ── MetadataExtractor ──────────────────────────────────────────────────────────

class TestMetadataExtractor:

    def setup_method(self):
        self.extractor = MetadataExtractor()
        self.doc = make_document("placeholder")

    def test_extracts_iso_date(self):
        doc = make_document("Contract dated 2024-03-15 is valid.")
        result = self.extractor.extract(doc.text, doc)
        assert "2024-03-15" in result.dates

    def test_extracts_written_date(self):
        doc = make_document("Signed on January 15, 2024 by both parties.")
        result = self.extractor.extract(doc.text, doc)
        assert any("January" in d or "2024" in d for d in result.dates)

    def test_extracts_email(self):
        doc = make_document("Contact john.doe@example.com for details.")
        result = self.extractor.extract(doc.text, doc)
        assert "john.doe@example.com" in result.emails

    def test_extracts_url(self):
        doc = make_document("Visit https://docs.company.com/api for the spec.")
        result = self.extractor.extract(doc.text, doc)
        assert any("docs.company.com" in u for u in result.urls)

    def test_extracts_markdown_headers(self):
        doc = make_document("## Introduction\n\nContent here.\n\n## Conclusion")
        result = self.extractor.extract(doc.text, doc)
        assert "Introduction" in result.section_headers
        assert "Conclusion" in result.section_headers

    def test_word_count_correct(self):
        doc = make_document("one two three four five")
        result = self.extractor.extract(doc.text, doc)
        assert result.word_count == 5

    def test_detects_english(self):
        doc = make_document(
            "This is a standard enterprise policy document for all employees. "
            "It covers compliance requirements and expected behaviour at work."
        )
        result = self.extractor.extract(doc.text, doc)
        assert result.language == "en"

    def test_detects_policy_category(self):
        doc = make_document(
            "This policy shall apply to all employees. Compliance is required. "
            "Employees must not violate the guidelines and procedures."
        )
        result = self.extractor.extract(doc.text, doc)
        assert result.category == "policy"

    def test_detects_email_category_from_source_type(self):
        doc = make_document("Hello, please review attached.", source_type="eml")
        result = self.extractor.extract(doc.text, doc)
        assert result.category == "email"

    def test_deduplicates_emails(self):
        text = "Email: alice@test.com\nAlso: alice@test.com\nOr: alice@test.com"
        doc = make_document(text)
        result = self.extractor.extract(doc.text, doc)
        assert result.emails.count("alice@test.com") == 1


# ── PreprocessingPipeline ──────────────────────────────────────────────────────

class TestPreprocessingPipeline:

    def setup_method(self):
        self.pipeline = PreprocessingPipeline()

    def test_returns_processed_document(self):
        doc = make_document("Hello world. This is a test document for processing.")
        result = self.pipeline.run(doc)
        assert isinstance(result, ProcessedDocument)

    def test_cleaned_text_is_populated(self):
        doc = make_document("Some text with\r\nWindows line endings.")
        result = self.pipeline.run(doc)
        assert len(result.cleaned_text) > 0
        assert "\r" not in result.cleaned_text

    def test_processing_stats_present(self):
        doc = make_document("Text with content for stats.")
        result = self.pipeline.run(doc)
        assert "original" in result.processing_stats
        assert "final" in result.processing_stats
        assert "reduction_pct" in result.processing_stats
        assert "stages_applied" in result.processing_stats

    def test_original_document_preserved(self):
        original_text = "Original unmodified text."
        doc = make_document(original_text)
        result = self.pipeline.run(doc)
        assert result.document.text == original_text

    def test_handles_empty_text(self):
        doc = make_document("")
        result = self.pipeline.run(doc)
        assert result.cleaned_text == ""

    def test_all_stages_recorded_in_stats(self):
        doc = make_document("Sample text.")
        result = self.pipeline.run(doc)
        stages = result.processing_stats["stages_applied"]
        assert "WhitespaceCleaner" in stages
        assert "NoiseCleaner" in stages
        assert "UnicodeNormalizer" in stages
        assert "HyphenNormalizer" in stages

    def test_unicode_normalised_in_output(self):
        doc = make_document("He said “hello” and left.")
        result = self.pipeline.run(doc)
        assert "“" not in result.cleaned_text
        assert '"hello"' in result.cleaned_text


# ── PreprocessingService ───────────────────────────────────────────────────────

class TestPreprocessingService:

    def setup_method(self):
        self.service = PreprocessingService()

    def test_success_result(self):
        doc = make_document("Valid document text for processing.")
        result = self.service.process(doc)
        assert result.success is True
        assert result.processed_document is not None
        assert result.error is None

    def test_processed_document_has_cleaned_text(self):
        doc = make_document("Text with\r\nWindows endings and\ttabs.")
        result = self.service.process(doc)
        assert "\r" not in result.processed_document.cleaned_text
        assert "\t" not in result.processed_document.cleaned_text

    def test_error_on_pipeline_failure(self):
        """Inject a broken pipeline to verify error wrapping works."""
        class BrokenPipeline:
            def run(self, document):
                raise RuntimeError("simulated pipeline failure")

        service = PreprocessingService(pipeline=BrokenPipeline())
        doc = make_document("some text")
        result = service.process(doc)
        assert result.success is False
        assert result.processed_document is None
        assert "simulated pipeline failure" in result.error

    def test_result_preserves_tenant_id(self):
        doc = make_document("Some content.")
        doc.tenant_id = "acme-corp"
        result = self.service.process(doc)
        assert result.processed_document.document.tenant_id == "acme-corp"

    def test_result_preserves_rbac_fields(self):
        doc = make_document("Some content.")
        doc.owner_id = "user-99"
        doc.access_roles = ["legal", "hr"]
        doc.visibility = "restricted"
        result = self.service.process(doc)
        pd = result.processed_document
        assert pd.document.owner_id == "user-99"
        assert pd.document.access_roles == ["legal", "hr"]
        assert pd.document.visibility == "restricted"

    def test_extracted_metadata_populated(self):
        doc = make_document(
            "This policy shall apply to all. Contact hr@company.com for details. "
            "Effective 2024-01-01."
        )
        result = self.service.process(doc)
        meta = result.processed_document.extracted_metadata
        assert meta.word_count > 0
        assert meta.char_count > 0
        assert "hr@company.com" in meta.emails
        assert "2024-01-01" in meta.dates
