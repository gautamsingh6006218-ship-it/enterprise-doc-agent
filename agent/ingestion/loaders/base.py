"""
ingestion/loaders/base.py

What problem does this solve?
- Without a shared contract, each loader defines its own method names and
  signatures. LoaderRegistry cannot call them uniformly.

Why does this file exist?
- Defines the ABC (Abstract Base Class) every loader must implement.
- Enforces the contract at class definition time — a loader missing `load()`
  raises TypeError on instantiation, not silently at runtime.
- `can_load()` is a shared helper so loaders never repeat extension-check logic.

Why ABC instead of Protocol?
- ABC raises at instantiation if a method is missing. Protocol raises at call
  time. For a pipeline, early failure is always better.
- Better IDE autocomplete and static analysis support.

How to add a new format (e.g. PPTX):
1. Create agent/ingestion/loaders/pptx.py
2. Subclass BaseDocumentLoader
3. Implement supported_extensions and load()
4. Register in LoaderRegistry — nothing else changes.
"""

from abc import ABC, abstractmethod

from agent.ingestion.models import Document


class BaseDocumentLoader(ABC):
    """
    What problem does this solve?
    - Gives LoaderRegistry a single type to program against regardless of
      how many loaders exist or what formats they handle.

    Why does this class exist?
    - Every loader (PDF, DOCX, TXT, HTML) must expose the same two things:
      what extensions it handles and how to load a file.
    - Shared `can_load()` helper so callers never need to inspect extensions
      themselves.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """
        What problem does this solve?
        - LoaderRegistry needs to know which extensions map to which loader
          without instantiating or calling the loader.

        Why list[str] instead of a single str?
        - One loader can handle multiple related formats (e.g. TxtLoader
          handles .txt, .md, .markdown). A list avoids creating duplicate
          loader classes for near-identical formats.

        Required format: lowercase with leading dot — e.g. [".pdf"], [".txt", ".md"]
        """
        ...

    @abstractmethod
    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        What problem does this solve?
        - Converts a file on disk into a normalised Document object that every
          downstream service (chunker, embedder) can consume without knowing
          the source format.

        Why are these inputs required?
        - file_path:  The source file. Without it there is nothing to load.
        - tenant_id:  Passed into Document for multi-tenant data isolation.
                      The loader sets it so every Document carries its tenant
                      from the moment it is created.

        Why Document instead of raw str or dict?
        - raw str loses all metadata (source path, hash, tenant, created_at).
        - dict is untyped — callers would need to know field names by memory.
        - Document is a typed contract: every field is named and its purpose
          is documented in models.py.

        Raises:
        - FileNotFoundError  if the file does not exist.
        - ValueError         if the file extension is not supported by this loader.
        """
        ...

    def can_load(self, file_path: str) -> bool:
        """
        What problem does this solve?
        - LoaderRegistry needs a cheap way to check compatibility before
          calling load() — avoids opening the file just to reject it.

        Why a concrete method here instead of abstract?
        - The logic (check suffix against supported_extensions) is identical
          for every loader. Putting it here means loaders never have to
          implement it themselves.

        Why bool instead of raising?
        - Used as a routing predicate, not an assertion. Callers decide what
          to do when False — some log a warning, some skip, some raise.
        """
        from pathlib import Path
        return Path(file_path).suffix.lower() in self.supported_extensions
