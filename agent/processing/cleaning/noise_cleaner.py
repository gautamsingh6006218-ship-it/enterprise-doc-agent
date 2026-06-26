"""
processing/cleaning/noise_cleaner.py

What problem does this solve?
- Enterprise documents contain structural boilerplate that has zero semantic
  value for RAG: page numbers, repeated headers/footers, watermarks, and
  table-of-contents entries. Including these in chunks degrades embedding
  quality and wastes vector store capacity.

Why regex instead of ML-based noise detection?
- Regex is deterministic, fast (microseconds per document), and has no
  model size or inference cost. These noise patterns are structurally
  predictable — they don't need semantic understanding to detect.
- ML-based approaches are reserved for ambiguous cases (Phase 2).

Why source_type-aware cleaning?
- Page number patterns only appear in PDFs (not in plain text or email).
- Email headers (From:, Subject:) only appear in .eml/.msg documents.
- Applying the wrong rules removes valid content from the wrong format.

Conservatism principle:
- When in doubt, do NOT remove. A false negative (keeping noise) is
  preferable to a false positive (removing real content). Each pattern
  is anchored to line boundaries (^, $) to prevent mid-text removal.
"""

import re

from agent.processing.cleaning.base import BaseTextCleaner


class NoiseCleaner(BaseTextCleaner):
    """
    What problem does this solve?
    - Removes structural boilerplate from documents so chunks contain only
      semantically meaningful content.

    Why does this class exist?
    - WhitespaceCleaner handles whitespace artefacts. NoiseCleaner handles
      content-level noise. Separating them keeps each cleaner focused
      and independently testable.

    Patterns removed:
    - Page numbers (PDF-specific)
    - Standalone watermark lines (CONFIDENTIAL, DRAFT, etc.)
    - Table-of-contents entries (title ........ page_number)
    - Email routing headers (From:, To:, Message-ID:, etc.)
    """

    # ── Page number patterns (PDF only) ───────────────────────────────────────
    # Anchored to full lines (^ and $) to avoid removing numbers mid-sentence.

    # "Page 3 of 15", "Page 3", "page 3 of 15"
    _PAGE_NUM_EXPLICIT_RE = re.compile(
        r'(?m)^\s*[Pp]age\s+\d+(?:\s+of\s+\d+)?\s*$'
    )
    # "— 3 —", "- 3 -", "| 3 |"  (centered page numbers with decorators)
    _PAGE_NUM_DECORATED_RE = re.compile(
        r'(?m)^\s*[-–—|]\s*\d{1,4}\s*[-–—|]\s*$'
    )
    # "3" alone on a line — conservative: only 1-3 digit standalone numbers
    # that appear at the very start or end of a page block.
    _PAGE_NUM_BARE_RE = re.compile(
        r'(?m)^\s*\d{1,3}\s*$'
    )

    # ── Watermark patterns ────────────────────────────────────────────────────
    # Only matches when the watermark word appears ALONE on a line.
    # "This document is CONFIDENTIAL" is NOT removed — only a bare line.
    _WATERMARK_RE = re.compile(
        r'(?mi)^\s*(?:'
        r'CONFIDENTIAL'
        r'|DRAFT'
        r'|COPY'
        r'|SAMPLE'
        r'|PROPRIETARY'
        r'|INTERNAL\s+USE\s+ONLY'
        r'|DO\s+NOT\s+DISTRIBUTE'
        r'|NOT\s+FOR\s+DISTRIBUTION'
        r')\s*$'
    )

    # ── Table of contents entries ─────────────────────────────────────────────
    # Matches lines like: "Introduction .............. 5"
    # or "Chapter 2  The Architecture          12"
    # Requires 4+ dots OR 6+ spaces as the leader to avoid false positives.
    _TOC_DOTS_RE = re.compile(r'(?m)^.{3,80}\.{4,}\s*\d+\s*$')
    _TOC_SPACES_RE = re.compile(r'(?m)^.{3,80}\s{6,}\d+\s*$')

    # ── Email routing headers ─────────────────────────────────────────────────
    # Removes technical headers that add no semantic value to RAG.
    # "From:", "To:", "Cc:", "Bcc:", "Message-ID:", "X-Mailer:", etc.
    _EMAIL_HEADER_RE = re.compile(
        r'(?m)^\s*(?:'
        r'From|To|Cc|Bcc|Reply-To|Message-ID|X-[\w-]+'
        r'|Delivered-To|Received|MIME-Version|Content-Type'
        r'|Content-Transfer-Encoding|Return-Path'
        r')\s*:.*$'
    )

    def clean(self, text: str, source_type: str = "unknown") -> str:
        """
        What problem does this solve?
        - Removes structurally identifiable noise from documents, leaving
          only semantically meaningful content for chunking.

        Why are these inputs required?
        - text:        The text to clean (after WhitespaceCleaner).
        - source_type: Controls which patterns are applied.
                       Page number removal only runs on "pdf" to avoid
                       removing valid standalone numbers from other formats.

        Why not strip email headers for all source_types?
        - "From:" appears legitimately in contract text ("From this date...").
          Restricting to email source_types prevents false removals.

        Why keep blank-line cleanup after each removal?
        - Removing a watermark or TOC line leaves a double blank line.
          Collapsing these keeps the output clean for the chunker.

        Returns text with structural noise removed.
        """
        # PDF-specific: page numbers appear as standalone lines between pages
        if source_type in ("pdf",):
            text = self._PAGE_NUM_EXPLICIT_RE.sub('', text)
            text = self._PAGE_NUM_DECORATED_RE.sub('', text)
            text = self._PAGE_NUM_BARE_RE.sub('', text)

        # Watermarks — safe to apply across all formats
        text = self._WATERMARK_RE.sub('', text)

        # TOC entries — safe for pdf, docx, markdown
        if source_type in ("pdf", "docx", "markdown"):
            text = self._TOC_DOTS_RE.sub('', text)
            text = self._TOC_SPACES_RE.sub('', text)

        # Email routing headers — only for email formats
        if source_type in ("eml", "msg", "email"):
            text = self._EMAIL_HEADER_RE.sub('', text)

        # Collapse the double blank lines left behind by removed lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()
