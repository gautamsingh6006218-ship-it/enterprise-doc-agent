"""
api/app.py

FastAPI application assembly: registers routers, configures CORS, starts
background scheduler and file watcher, and exposes the app object for uvicorn.

Run with:
  uvicorn agent.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # loads .env before any service reads os.getenv()

from agent.api.routes import documents, ingest, query, sync
from agent.api.scheduler import start_scheduler, stop_scheduler
from agent.watchers.file_watcher import start_file_watcher, stop_file_watcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch scheduler and file watcher
    start_scheduler()
    start_file_watcher()
    yield
    # Shutdown: stop both cleanly
    stop_scheduler()
    stop_file_watcher()


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
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
