"""
processing/cleaning/whitespace_cleaner.py

What problem does this solve?
- PDF extractors and OCR engines produce text with excessive blank lines,
  trailing spaces, tabs mixed with spaces, and non-printable control
  characters. These pollute chunks and waste embedding token budget.

Why run this before NoiseCleaner?
- Normalising whitespace first makes regex patterns in NoiseCleaner simpler
  and more reliable. A page-number pattern is easier to match on a clean
  line than on one surrounded by tabs and trailing spaces.

What this cleaner does NOT touch:
- Single newlines between paragraphs — preserved as structure signals.
- Leading indentation in code blocks — only affects prose documents.
- Content characters — only whitespace variants are modified.
"""

import re
import unicodedata

from agent.processing.cleaning.base import BaseTextCleaner


class WhitespaceCleaner(BaseTextCleaner):
    """
    What problem does this solve?
    - Removes all whitespace artefacts that add zero semantic value but
      consume token budget and confuse chunk boundary detection.

    Why does this class exist?
    - First stage in the cleaning pipeline. Produces a predictable
      whitespace structure that all subsequent cleaners can rely on.

    Cleaning stages (applied in order):
    1. Strip null bytes and non-printable control characters.
    2. Normalise line endings to \\n.
    3. Replace tabs with single spaces.
    4. Strip trailing whitespace from each line.
    5. Collapse 3+ consecutive blank lines → 2 blank lines.
    6. Strip leading/trailing whitespace from the full text.
    """

    # Matches control characters except \n (line feed) and \r (carriage return).
    # \x00-\x08: NUL through BS  |  \x0b-\x0c: VT, FF
    # \x0e-\x1f: SO through US   |  \x7f: DEL
    _CONTROL_CHARS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    # Three or more consecutive blank lines (lines containing only whitespace).
    _EXCESS_BLANK_LINES_RE = re.compile(r'\n{3,}')

    # Trailing whitespace (spaces/tabs) at end of each line.
    _TRAILING_WHITESPACE_RE = re.compile(r'[ \t]+$', re.MULTILINE)

    def clean(self, text: str, source_type: str = "unknown") -> str:
        """
        What problem does this solve?
        - Produces text with predictable, clean whitespace structure
          before any content-level cleaning rules are applied.

        Why remove control characters first?
        - Control characters can break regex patterns in later stages
          (NoiseCleaner, MetadataExtractor) if not removed first.

        Why normalise \\r\\n to \\n?
        - Windows line endings (\r\n) from DOCX/email files would otherwise
          make single-newline patterns miss or double-count line breaks.

        Why cap at 2 consecutive blank lines (not 1)?
        - Two blank lines is the common convention for section separation.
          Collapsing to 1 would merge visually distinct sections.
          Anything beyond 2 is artefact from PDF page breaks or OCR gaps.

        Args:
        - text:        Raw text to clean.
        - source_type: Not used here — whitespace rules apply to all formats.

        Returns cleaned text with normalised whitespace.
        """
        # Step 1: remove non-printable control characters (NUL, DEL, BEL, etc.)
        text = self._CONTROL_CHARS_RE.sub('', text)

        # Step 2: normalise Windows (\r\n) and old Mac (\r) line endings to \n
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Step 3: replace tabs with a single space — tabs in prose are artefacts,
        # not intentional indentation worth preserving
        text = text.replace('\t', ' ')

        # Step 4: strip trailing spaces/tabs from every line
        text = self._TRAILING_WHITESPACE_RE.sub('', text)

        # Step 5: collapse 3+ blank lines → 2 blank lines
        # 2 blank lines (\n\n\n in the string) = one visible empty line between blocks
        text = self._EXCESS_BLANK_LINES_RE.sub('\n\n', text)

        # Step 6: remove leading/trailing whitespace from the full document
        return text.strip()
