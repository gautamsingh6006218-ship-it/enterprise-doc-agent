"""
processing/normalization/unicode_normalizer.py

What problem does this solve?
- Enterprise documents (DOCX, PDF, HTML) contain many Unicode variants of
  the same character: curly quotes, em dashes, fancy bullets, non-breaking
  spaces, zero-width characters, and BOM markers. These cause two problems:
  1. Embedding models tokenise them inconsistently — "it's" (curly) and
     "it's" (straight) may produce different token sequences and vectors.
  2. Regex patterns in MetadataExtractor fail on fancy variants (e.g. a
     date regex written for ASCII dashes misses em-dash separated dates).

Why character mapping instead of unicodedata.normalize()?
- unicodedata.normalize(NFKC) handles some cases but misses visually
  identical characters like curly quotes and smart apostrophes that have
  different Unicode code points.
- An explicit mapping gives full control and is easy to extend.

Why not strip all non-ASCII?
- Enterprise documents are multilingual. French, German, Spanish characters
  (é, ü, ñ, ç) are valid content. Only replace specific problematic
  Unicode variants with their ASCII equivalents — not all non-ASCII.
"""

from agent.processing.normalization.base import BaseNormalizer


class UnicodeNormalizer(BaseNormalizer):
    """
    What problem does this solve?
    - Replaces Unicode typographic variants with their plain ASCII equivalents
      so embedding models and downstream regex patterns work consistently.

    Why does this class exist?
    - Isolates all Unicode normalisation in one place. Adding a new
      character mapping only touches this file.

    Character categories normalised:
    - Quotation marks (curly/smart → straight)
    - Dashes (em dash, en dash → hyphen-minus)
    - Bullets (•, ▪, ►, ‣ → -)
    - Whitespace variants (non-breaking space, thin space → regular space)
    - Zero-width characters (removed entirely — they are invisible noise)
    - BOM (byte order mark — removed, causes issues at string start)
    """

    # Mapping: Unicode char → ASCII replacement.
    # Order does not matter — replacements are applied via str.translate().
    _CHAR_MAP: dict[int, str] = {
        # Quotation marks
        ord('‘'): "'",   # LEFT SINGLE QUOTATION MARK  '
        ord('’'): "'",   # RIGHT SINGLE QUOTATION MARK '
        ord('‚'): "'",   # SINGLE LOW-9 QUOTATION MARK ‚
        ord('‛'): "'",   # SINGLE HIGH-REVERSED QUOTATION MARK ‛
        ord('“'): '"',   # LEFT DOUBLE QUOTATION MARK  "
        ord('”'): '"',   # RIGHT DOUBLE QUOTATION MARK "
        ord('„'): '"',   # DOUBLE LOW-9 QUOTATION MARK „
        ord('‟'): '"',   # DOUBLE HIGH-REVERSED QUOTATION MARK ‟
        ord('«'): '"',   # LEFT-POINTING DOUBLE ANGLE «
        ord('»'): '"',   # RIGHT-POINTING DOUBLE ANGLE »

        # Dashes — all normalised to hyphen-minus for regex consistency
        ord('–'): '-',   # EN DASH –
        ord('—'): '-',   # EM DASH —
        ord('―'): '-',   # HORIZONTAL BAR ―
        ord('−'): '-',   # MINUS SIGN −

        # Bullets — normalised to hyphen so list items are uniformly formatted
        ord('•'): '-',   # BULLET •
        ord('▪'): '-',   # BLACK SMALL SQUARE ▪
        ord('▫'): '-',   # WHITE SMALL SQUARE ▫
        ord('►'): '-',   # BLACK RIGHT-POINTING POINTER ►
        ord('●'): '-',   # BLACK CIRCLE ●
        ord('⁃'): '-',   # HYPHEN BULLET ⁃
        ord('‣'): '-',   # TRIANGULAR BULLET ‣

        # Whitespace variants — all to regular ASCII space
        ord(' '): ' ',   # NO-BREAK SPACE
        ord(' '): ' ',   # NARROW NO-BREAK SPACE
        ord(' '): ' ',   # THIN SPACE
        ord(' '): ' ',   # PUNCTUATION SPACE
        ord(' '): ' ',   # FIGURE SPACE
        ord(' '): ' ',   # EM SPACE
        ord(' '): ' ',   # EN SPACE

        # Zero-width characters — removed entirely (invisible noise)
        ord('​'): '',    # ZERO WIDTH SPACE
        ord('‌'): '',    # ZERO WIDTH NON-JOINER
        ord('‍'): '',    # ZERO WIDTH JOINER
        ord('\u200E'): '',    # LEFT-TO-RIGHT MARK
        ord('\u200F'): '',    # RIGHT-TO-LEFT MARK
        ord('﻿'): '',    # ZERO WIDTH NO-BREAK SPACE (BOM)

        # Ellipsis — expanded to three dots so tokeniser handles it uniformly
        ord('…'): '...',  # HORIZONTAL ELLIPSIS …
    }

    # Pre-built translation table for performance.
    # str.translate() with a pre-built table is O(n) and faster than
    # iterative str.replace() calls in a loop.
    _TRANSLATION_TABLE = str.maketrans(_CHAR_MAP)

    def normalize(self, text: str) -> str:
        """
        What problem does this solve?
        - Replaces all typographic Unicode variants with plain ASCII equivalents
          so text is consistent for tokenisation, regex, and embedding.

        Why str.translate() instead of multiple str.replace() calls?
        - str.translate() scans the string once (O(n)) regardless of how many
          substitutions are defined. N separate str.replace() calls scan N times.
          For 35 character mappings, translate() is ~35x faster.

        Returns text with all mapped characters replaced.
        """
        return text.translate(self._TRANSLATION_TABLE)
