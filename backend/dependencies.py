"""
backend/dependencies.py

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

from fastapi import Depends

from agent.chunking import ChunkingService
from agent.documents import DocumentService
from agent.embeddings import BGEEncoder, EmbeddingService, PgVectorStore
from agent.ingestion import IngestionService
from agent.llm import LLMService, OllamaProvider
from agent.pipeline import DocumentPipeline, PipelineService
from agent.processing import PreprocessingService
from agent.qa import QAService
from agent.registry import DocumentRegistryStore, RegistryService
from agent.retrieval import BGEReranker, RetrievalPipeline, RetrievalService


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
    store = _get_registry_store()
    if store is None:
        raise RuntimeError("PGVECTOR_URL not set — registry service unavailable")
    return RegistryService(store=store)


@lru_cache(maxsize=1)
def get_vector_store() -> PgVectorStore:
    """Returns the PgVectorStore singleton for chunk deletion operations."""
    pgvector_url = os.getenv("PGVECTOR_URL", "")
    import psycopg2
    conn = psycopg2.connect(pgvector_url)
    return PgVectorStore(connection=conn)


def get_document_service(
    registry_service: RegistryService = Depends(get_registry_service),
    vector_store: PgVectorStore = Depends(get_vector_store),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
) -> DocumentService:
    """
    Per-request DocumentService over the cached registry, vector store and pipeline.

    Why Depends parameters instead of lru_cache?
    - Tests override get_registry_service, get_vector_store and
      get_pipeline_service. Declaring them as sub-dependencies makes those
      overrides reach DocumentService too.
    """
    return DocumentService(
        registry_service=registry_service,
        chunk_store=vector_store,
        pipeline_service=pipeline_service,
    )


def build_document_service() -> DocumentService:
    """DocumentService for callers outside a request, such as the file watcher."""
    return get_document_service(get_registry_service(), get_vector_store(), get_pipeline_service())


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """
    Returns an LLMService backed by a local Ollama model.

    Environment variables:
      OLLAMA_LLM_MODEL  — model name (default: llama3.2)
      OLLAMA_BASE_URL   — Ollama host (default: http://localhost:11434)
    """
    provider = OllamaProvider()
    return LLMService(provider=provider)


def get_qa_service(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> QAService:
    """Per-request QAService; sub-dependencies let tests override retrieval and LLM."""
    return QAService(retrieval_service=retrieval_service, llm_service=llm_service)


def _get_registry_store():
    pgvector_url = os.getenv("PGVECTOR_URL")
    if not pgvector_url:
        return None
    import psycopg2
    conn = psycopg2.connect(pgvector_url)
    return DocumentRegistryStore(connection=conn)
