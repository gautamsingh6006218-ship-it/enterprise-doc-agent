"""
api/app.py

What problem does this solve?
- FastAPI application assembly: registers routers, configures CORS,
  and exposes the app object that uvicorn serves.

Why a create_app() factory instead of a module-level app?
- Tests call create_app() to get a fresh app instance with overridden
  dependencies. A module-level app would carry state between tests.
- Enables multiple app instances in the same process (useful for testing
  different configurations without monkey-patching globals).

Run with:
  uvicorn agent.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.api.routes import documents, ingest, query, sync


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise Document AI Agent",
        description=(
            "Ingest documents from files, Confluence, SharePoint, Jira, and Wiki. "
            "Query via hybrid dense+BM25 retrieval with bge-reranker-v2-m3 reranking."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — tighten origins in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(documents.router)
    app.include_router(sync.router)

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok", "version": "1.0.0"}

    return app


# Module-level instance for uvicorn
app = create_app()
