"""
ingestion/loaders/base.py
-------------------------
Abstract base class (contract) that every document loader must implement.

Why an ABC instead of a Protocol?
- ABC enforces the contract at class definition time — a loader that forgets
  to implement `load()` raises TypeError on instantiation, not at call time.
- Better IDE support: type checkers and autocomplete work correctly with ABC.
- `can_load()` is provided as a concrete helper so individual loaders don't
  have to repeat the extension-check logic.

To add a new format (e.g. PPTX):
    1. Create `agent/ingestion/loaders/pptx.py`
    2. Subclass BaseDocumentLoader
    3. Implement `supported_extensions` and `load()`
    4. Register in LoaderRegistry — no other files need to change.
"""

from abc import ABC, abstractmethod

from agent.ingestion.models import Document


class BaseDocumentLoader(ABC):
    """
    Contract that all document loaders must satisfy.

    Each loader is responsible for exactly one concern: reading a specific
    file format and returning a normalised Document object. Validation,
    chunking, and embedding are handled by separate services downstream.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """
        Returns the list of file extensions this loader can handle.

        Extensions must be lowercase and include the leading dot.
        Example: ['.pdf'] or ['.txt', '.md', '.markdown']
        """
        ...

    @abstractmethod
    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        Load the file at `file_path` and return a normalised Document.

        Args:
            file_path:  Absolute or relative path to the source file.
            tenant_id:  Tenant identifier for multi-tenant isolation.

        Returns:
            A Document instance with id, source_type, text, and metadata
            populated. The text field must always be plain UTF-8 string.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the file extension is not supported by
                               this loader.
        """
        ...

    def can_load(self, file_path: str) -> bool:
        """
        Quick check whether this loader supports the given file.

        Used by LoaderRegistry to route files without instantiating loaders.

        Args:
            file_path: Path to the file (only the extension is inspected).

        Returns:
            True if the file extension matches one of supported_extensions.
        """
        from pathlib import Path
        return Path(file_path).suffix.lower() in self.supported_extensions
