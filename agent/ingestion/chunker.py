from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agent.ingestion.models import Document, DocumentChunk


def chunk_document(document: Document,chunk_size: int= 1000,chunk_overlap:int = 200) -> list[DocumentChunk]:

    if not document.text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n","\n",". "," ",""],
    )
    texts = splitter.split_text(document.text)
    chunks: list[DocumentChunk] = []

    for chunk_index, text in enumerate(texts):
        metadata = dict(document.metadata)
        metadata["source_type"] = document.source_type
        metadata["source_path"] = document.source_path
        metadata["title"] = document.title
        metadata["chunk_size"] = chunk_size
        metadata["chunk_overlap"] = chunk_overlap

        chunk = DocumentChunk(
            id = str(uuid4()),
            document_id=document.id,
            text=text,
            chunk_index=chunk_index,
            metadata=metadata
        )

        chunks.append(chunk)
    return chunks



