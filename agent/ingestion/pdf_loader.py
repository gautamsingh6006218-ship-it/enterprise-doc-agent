from pathlib import Path
from uuid import uuid4

import fitz

from agent.ingestion.models import Document

def load_pdf(file_path: str) -> Document:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a pdf file, got : {path.suffix}")
    
    pdf = fitz.open(path)

    pages_text: list[str] = []

    for page in pdf:
        text = page.get_text()
        pages_text.append(text)
    
    full_text = "\n\n".join(pages_text).strip()

    metadata = {
        "page_count": pdf.page_count,
        "file_name": path.name,
    }
    pdf.close()

    return Document(
        id=str(uuid4()),
        source_type="pdf",
        source_path=str(path),
        title=path.stem,
        text=full_text,
        metadata=metadata
    )