from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbeddingResult:
    chunk_id: str
    document_id: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)