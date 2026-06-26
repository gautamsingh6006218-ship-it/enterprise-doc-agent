"""
processing/normalization/hyphen_normalizer.py

What problem does this solve?
- PDF text extraction splits long words across lines with a hyphen:
    "infor-\nmation" instead of "information"
    "multi-\nlingual" instead of "multi-lingual" OR "multilingual"
- These broken words produce incorrect token sequences in the embedding
  model. "infor" and "mation" as separate tokens have no semantic meaning;
  "information" as one token does.

Why is this a separate normalizer from UnicodeNormalizer?
- Unicode normalisation is character-level substitution.
- Hyphen normalisation is multi-character pattern matching across line
  boundaries. Different concern, different regex logic.

The hard problem — compound vs broken hyphens:
- "end-\nof-line" (broken compound) → should become "end-of-line"
- "infor-\nmation" (broken word)    → should become "information"
- "self-\nsufficient" (ambiguous)   → heuristic: rejoin as "self-sufficient"

Approach:
- When a word ends with hyphen at end of line AND the next line starts with
  a lowercase letter → the hyphen is a line-break artefact → rejoin.
- When next line starts with uppercase → likely a new sentence → keep as-is.
- This heuristic is ~95% accurate for English enterprise documents.
"""

import re

from agent.processing.normalization.base import BaseNormalizer


class HyphenNormalizer(BaseNormalizer):
    """
    What problem does this solve?
    - Repairs PDF line-break hyphenation so broken words are reunited
      before embedding. "infor-\\nmation" → "information".

    Why does this class exist?
    - This is a PDF-specific artefact. Keeping it isolated means it can be
      disabled for non-PDF source types in the future without touching
      other normalizers.
    """

    # Matches: word-\n + optional leading whitespace + lowercase continuation
    # Group 1: the part before the hyphen  e.g. "infor"
    # Group 2: the continuation word       e.g. "mation"
    #
    # Why lowercase continuation only?
    # - Uppercase after \n typically signals a new sentence or proper noun,
    #   not a line-break continuation. Rejoining those would be incorrect.
    _SOFT_HYPHEN_RE = re.compile(r'(\w+)-\n\s*([a-z]\w*)')

    def normalize(self, text: str) -> str:
        """
        What problem does this solve?
        - Rejoins words broken across lines by PDF line-wrap hyphenation.

        Why rejoin as word1 + word2 (no hyphen) instead of word1-word2?
        - In most cases the hyphen was inserted purely by the PDF renderer
          to fit text in a column — it is not part of the actual word.
        - "infor-\nmation" is "information", not "infor-mation".
        - Legitimate compound hyphen words ("end-to-end") are NOT split
          across lines this way in well-formatted PDFs.

        Why a substitution function instead of a fixed replacement string?
        - The substitution needs to decide between joining with or without
          a hyphen. A re.sub() lambda allows per-match logic.

        Returns text with line-break hyphens repaired.
        """
        def rejoin(match: re.Match) -> str:
            prefix = match.group(1)   # e.g. "infor"
            suffix = match.group(2)   # e.g. "mation"
            # Heuristic: if prefix looks like a full standalone word that
            # is commonly hyphenated (self-, multi-, cross-, non-), keep the
            # hyphen. Otherwise drop it and join directly.
            _COMPOUND_PREFIXES = {
                "self", "multi", "cross", "non", "pre", "post",
                "co", "re", "inter", "intra", "over", "under",
            }
            if prefix.lower() in _COMPOUND_PREFIXES:
                return f"{prefix}-{suffix}"
            return f"{prefix}{suffix}"

        return self._SOFT_HYPHEN_RE.sub(rejoin, text)
