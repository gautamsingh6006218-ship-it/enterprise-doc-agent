"""
api/dependencies.py

What problem does this solve?
- FastAPI routes need service instances (PipelineService, RetrievalService, etc.)
  injected via Depends(). Without this file, each route would either create
  its own service (wasting memory, loading BGE-M3 multiple times) or import
  module-level singletons (untestable without monkeypatching).

Why module-level _service vars with getter functions?
- Services are created once at first Depends() call and reused across requests
  (effectively singletons). Tests override them via app.dependency_overrides
  without touching module state — the standard FastAPI testing pattern.

Why lazy initialisation (create on first call, not at import time)?
- BGE-M3 and the reranker each take several seconds to load. Importing this
  module should not trigger model loading — only the first request should.
- Tests that override the dependency never trigger model loading at all.

Environment variables used:
  PGVECTOR_URL      — PostgreSQL connection string (required for DB-backed services)
  JWT_SECRET        — see auth.py
"""

import os
from functools import lru_cache

from agent.pipeline.orchestrator import DocumentPipeline
from agent.services.chunking_service import ChunkingService
from agent.services.embedding_service import EmbeddingService
from agent.services.ingestion_service import IngestionService
from agent.services.llm_service import LLMService
from agent.services.pipeline_service import PipelineService
from agent.services.preprocessing_service import PreprocessingService
from agent.services.registry_service import RegistryService
from agent.services.retrieval_service import RetrievalService


@lru_cache(maxsize=1)
def get_pipeline_service() -> PipelineService:
    """
    What problem does this solve?
    - Wires the full 4-stage DocumentPipeline and wraps it in PipelineService.
      All services use production defaults (real model, real DB).

    Why lru_cache?
    - BGE-M3 is ~1GB on disk. Loading it once and caching the PipelineService
      means every request shares the same model instance. Without caching,
      each request would load the model from disk.

    Why no DB connection here?
    - EmbeddingService.store requires a psycopg2 connection. PipelineService
      can operate with store=None (no vector storage) for environments where
      the DB isn't wired up yet. The full DB-backed version is created when
      PGVECTOR_URL is set.
    """
    ingestion = IngestionService()
    preprocessing = PreprocessingService()
    chunking = ChunkingService()

    # EmbeddingService is wired with encoder+store only when PGVECTOR_URL is set
    pgvector_url = os.getenv("PGVECTOR_URL")
    if pgvector_url:
        from agent.embeddings.bge_encoder import BGEEncoder
        from agent.embeddings.store import PgVectorStore
        import psycopg2
        conn = psycopg2.connect(pgvector_url)
        encoder = BGEEncoder()
        store = PgVectorStore(connection=conn)
        embedding = EmbeddingService(encoder=encoder, store=store)
    else:
        embedding = EmbeddingService()  # returns error if called — acceptable for dev

    pipeline = DocumentPipeline(
        ingestion_service=ingestion,
        preprocessing_service=preprocessing,
        chunking_service=chunking,
        embedding_service=embedding,
    )

    registry_store = _get_registry_store()
    return PipelineService(pipeline=pipeline, registry_store=registry_store)


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    """
    What problem does this solve?
    - Wires BGE-M3 encoder + PgVectorStore + BGE reranker into RetrievalPipeline
      and wraps it in RetrievalService.

    Why requires PGVECTOR_URL?
    - Retrieval without a vector store returns no results.
      A RetrievalService with no pipeline raises at construction — this is
      intentional: retrieval with no DB is a misconfiguration, not a graceful
      degradation.
    """
    from agent.embeddings.bge_encoder import BGEEncoder
    from agent.embeddings.store import PgVectorStore
    from agent.retrieval.pipeline import RetrievalPipeline
    from agent.retrieval.reranker import BGEReranker
    import psycopg2

    pgvector_url = os.getenv("PGVECTOR_URL", "")
    conn = psycopg2.connect(pgvector_url)
    encoder = BGEEncoder()
    store = PgVectorStore(connection=conn)
    reranker = BGEReranker()
    retrieval_pipeline = RetrievalPipeline(
        encoder=encoder, store=store, reranker=reranker
    )
    return RetrievalService(pipeline=retrieval_pipeline)


@lru_cache(maxsize=1)
def get_registry_service() -> RegistryService:
    from agent.registry.store import DocumentRegistryStore
    store = _get_registry_store()
    if store is None:
        raise RuntimeError("PGVECTOR_URL not set — registry service unavailable")
    return RegistryService(store=store)


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """
    Returns an LLMService backed by a local Ollama model.

    Environment variables:
      OLLAMA_LLM_MODEL  — model name (default: llama3.2)
      OLLAMA_BASE_URL   — Ollama host (default: http://localhost:11434)
    """
    from agent.llm.ollama_provider import OllamaProvider
    provider = OllamaProvider()
    return LLMService(provider=provider)


def _get_registry_store():
    pgvector_url = os.getenv("PGVECTOR_URL")
    if not pgvector_url:
        return None
    from agent.registry.store import DocumentRegistryStore
    import psycopg2
    conn = psycopg2.connect(pgvector_url)
    return DocumentRegistryStore(connection=conn)
