import sys

from agent.embeddings.ollama_provider import OllamaEmbeddingProvider
from agent.ingestion.chunker import chunk_document
from agent.ingestion.pdf_loader import load_pdf


def verify_ollama_embeddings():
    args = sys.argv[1:]

    if len(args) != 1:
        raise ValueError("Provide only one pdf path")
    
    pdf_path = args[0]

    document = load_pdf(pdf_path)
    chunks = chunk_document(document)

    provider = OllamaEmbeddingProvider()
    embedding_results = provider.embed_chunks(chunks)

    return document, chunks, embedding_results


if __name__ == "__main__":
    try:
        document, chunks, embedding_results = verify_ollama_embeddings()

        print("Ollama embedding test successful")
        print(f"Document title: {document.title}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Total embeddings: {len(embedding_results)}")

        if not embedding_results:
            print("No embedding created")
            sys.exit(0)
        
        first_embedding = embedding_results[0]

        print(f"First embedding chunk id : {first_embedding.chunk_id}")
        print(f"First embedding doocument id: {first_embedding.document_id}")
        print(f"First embedding dimension: {len(first_embedding.embedding)}")
        print(f"First 5 embedding values: {first_embedding.embedding[:5]}")
        print(f"First embedding metadata: {first_embedding.metadata}")
    except (ValueError, FileNotFoundError) as e: 
        print(e)
        sys.exit(1)