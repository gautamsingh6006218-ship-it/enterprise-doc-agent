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