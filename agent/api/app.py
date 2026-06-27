"""
api/app.py

FastAPI application assembly: registers routers, configures CORS, starts
background scheduler and file watcher, and exposes the app object for uvicorn.

All API routes are mounted under /api so the built React app and the dev
Vite proxy both hit the same paths without rewriting.

Run with:
  uvicorn agent.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()  # loads .env before any service reads os.getenv()

from agent.api.routes import documents, ingest, query, sync
from agent.api.scheduler import start_scheduler, stop_scheduler
from agent.watchers.file_watcher import start_file_watcher, stop_file_watcher

# Path to the compiled React app (produced by `npm run build` inside frontend/)
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    start_file_watcher()
    yield
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
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API routes (all under /api) ────────────────────────────────────────────
    app.include_router(ingest.router,    prefix="/api")
    app.include_router(query.router,     prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(sync.router,      prefix="/api")

    @app.get("/api/health", tags=["health"])
    def health():
        return {"status": "ok", "version": "1.0.0"}

    @app.post("/api/auth/token", tags=["auth"])
    def get_dev_token(
        tenant_id: str = "acme",
        user_id: str = "user1",
        role: str = "admin",
    ):
        """Dev-mode token generator. Returns a 24-hour JWT. Disable in production."""
        from agent.api.auth import make_test_token
        token = make_test_token(tenant_id=tenant_id, user_id=user_id, roles=[role])
        return {"token": token, "tenant_id": tenant_id, "user_id": user_id}

    # ── React static build (only if frontend/dist exists) ─────────────────────
    if _FRONTEND_DIST.is_dir():
        # /assets — hashed JS/CSS bundles
        app.mount(
            "/assets",
            StaticFiles(directory=_FRONTEND_DIST / "assets"),
            name="assets",
        )

        # Root-level static files (favicon.svg, icons.svg, etc.)
        for static_file in _FRONTEND_DIST.iterdir():
            if static_file.is_file() and static_file.name != "index.html":
                _name = static_file.name

                @app.get(f"/{_name}", include_in_schema=False)
                def _static(f: Path = static_file):
                    return FileResponse(str(f))

        # SPA fallback: any non-API path returns index.html so React Router works
        @app.get("/{full_path:path}", include_in_schema=False)
        def spa_fallback(full_path: str):
            return FileResponse(str(_FRONTEND_DIST / "index.html"))

    return app


# Module-level instance for uvicorn
app = create_app()
