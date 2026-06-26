"""
processing/cleaning/base.py

What problem does this solve?
- Without a shared contract, each cleaner defines its own method signature.
  The pipeline cannot call them uniformly.

Why ABC instead of Protocol?
- Raises TypeError at class definition if clean() is not implemented.
  Catches missing implementations at instantiation, not at runtime.

Why pass source_type to clean()?
- PDF documents have page numbers; HTML documents do not.
- Email documents have From/To headers to strip; contracts do not.
- source_type allows cleaners to apply format-specific rules without
  needing separate cleaner classes per format.

How to add a new cleaner:
1. Subclass BaseTextCleaner.
2. Implement clean().
3. Add instance to PreprocessingPipeline._cleaners list.
   Nothing else needs to change.
"""

from abc import ABC, abstractmethod


class BaseTextCleaner(ABC):
    """
    What problem does this solve?
    - Gives PreprocessingPipeline a single type to iterate over,
      regardless of how many cleaners exist.

    Why does this class exist?
    - Enforces that every cleaner receives (text, source_type) and
      returns a cleaned str. Consistent interface across all cleaners.
    """

    @abstractmethod
    def clean(self, text: str, source_type: str = "unknown") -> str:
        """
        What problem does this solve?
        - Takes noisy text and returns a cleaner version.
          Each cleaner handles one specific category of noise.

        Why are these inputs required?
        - text:        The text to clean. Never modified in-place — always
                       returns a new string to keep the pipeline composable.
        - source_type: Format hint ("pdf", "docx", "html", "email").
                       Allows cleaners to skip rules irrelevant to the format.
                       e.g. page number removal only applies to "pdf".

        Why return str instead of modifying in-place?
        - Composability: pipeline chains clean() calls without side effects.
        - Testability: each cleaner can be tested in isolation with a plain string.
        """
        ...
