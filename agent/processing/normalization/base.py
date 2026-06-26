"""
processing/normalization/base.py

What problem does this solve?
- The pipeline needs to chain multiple normalizers uniformly.
  Without a contract, each normalizer has its own method name and signature.

Why a separate ABC from BaseTextCleaner?
- Cleaners remove noise (structural boilerplate). Normalizers transform
  valid content into a canonical form. Semantically different concerns,
  kept separate so each can evolve independently.
- Separation also makes it clear in the pipeline where noise ends and
  normalisation begins — useful for debugging quality issues.

How to add a new normalizer:
1. Subclass BaseNormalizer.
2. Implement normalize().
3. Add instance to PreprocessingPipeline._normalizers list.
"""

from abc import ABC, abstractmethod


class BaseNormalizer(ABC):
    """
    What problem does this solve?
    - Gives PreprocessingPipeline a single type to iterate over for
      the normalisation stage, regardless of how many normalizers exist.

    Why does this class exist?
    - Enforces that every normalizer receives a str and returns a str.
      Consistent interface means the pipeline loop never changes.
    """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        What problem does this solve?
        - Transforms text into a canonical form by replacing or rewriting
          characters/patterns that vary across documents but mean the same thing.

        Why no source_type parameter (unlike BaseTextCleaner)?
        - Normalisation rules (Unicode characters, hyphenation) apply
          universally across all document formats. There is no format-specific
          logic needed here — that distinction belongs in the cleaning stage.

        Why return str instead of modifying in-place?
        - Same reason as BaseTextCleaner.clean(): composability and testability.
          Each normalizer is a pure function on text.
        """
        ...
