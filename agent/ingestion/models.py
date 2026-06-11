from dataclasses import dataclass, field
from typing import Any

@dataclass
class Document:
    id: str
    source_type: str
    source_path: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


