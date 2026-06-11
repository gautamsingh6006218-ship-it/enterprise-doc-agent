from langchain_ollama import OllamaEmbeddings

from agent.embeddings.models import EmbeddingResult
from agent.ingestion.models import DocumentChunk


class OllamaEmbeddingProvider:
    def __init__(
            self,
            model: str = "nomic-embed-text",
            base_url: str= "http://localhost:11434",
                 ) -> None:
            self.model = model
            self.base_url = base_url
            self.client = OllamaEmbeddings(
                  model=model,
                  base_url=base_url,
            )
            
    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddingResult]:
          if not chunks:
                return []
          texts = [chunk.text for chunk in chunks]
          vectors = self.client.embed_documents(texts)

          results: list[EmbeddingResult] = []

          for chunk, vector in zip(chunks, vectors):
                result = EmbeddingResult(
                      chunk_id=chunk.id,
                      document_id=chunk.document_id,
                      embedding=vector,
                      metadata=dict(chunk.metadata),
                ) 
                results.append(result)

          return results
        