"""Registry module: owns the documents table (one pipeline record per document)."""

from agent.registry.models import STATUSES, DocumentRecord
from agent.registry.service import RegistryResult, RegistryService
from agent.registry.store import DocumentRegistryStore

__all__ = [
    "STATUSES",
    "DocumentRecord",
    "RegistryResult",
    "RegistryService",
    "DocumentRegistryStore",
]
