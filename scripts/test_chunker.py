import sys
from agent.ingestion.pdf_loader import load_pdf
from agent.ingestion.chunker import chunk_document

def verify_chunk_pdf():
    args = sys.argv[1:]

    if len(args) != 1:
        raise ValueError("Provide only one argument")
    pdf_path = args[0]
    document = load_pdf(pdf_path)

    chunks = chunk_document(document)
    
    return document, chunks
    
if __name__ == "__main__":
    try:
        document, chunks = verify_chunk_pdf()
        print("Document chunked successfully")
        print(f"Document {document.title}")
        print(f"Document text length: {len(document.text)}")
        print(f"Total chunks {len(chunks)}")

        if not chunks:
             print("No chunk")
             sys.exit(0)
        first_chunk = chunks[0]
        
        print(f"First chunk ID: {first_chunk.id}")
        print(f"First chunk document ID: {first_chunk.document_id}")
        print(f"First chunk index: {first_chunk.chunk_index}")
        print(f"First chunk text length: {len(first_chunk.text)}")
        print(f"First chunk metadata: {first_chunk.metadata}")
        print(f"First chunk preview: {first_chunk.text[:500]}")
    except (ValueError, FileNotFoundError) as e:
        print(e)
        sys.exit(1)