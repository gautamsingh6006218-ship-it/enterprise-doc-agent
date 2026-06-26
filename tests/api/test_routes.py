"""
tests/api/test_routes.py

What does this cover?
- Auth:             missing token → 401, expired token → 401, missing
                    tenant_id claim → 403, valid token → RBACContext injected.
- POST /ingest:     golden path (file uploaded, pipeline called, 200 response),
                    empty file → 400, exact duplicate skipped via registry,
                    pipeline failure propagated in response.
- POST /query:      golden path (chunks returned), empty query → 422,
                    retrieval failure → 500.
- GET /documents:   list returns records, status filter forwarded,
                    registry error → 500.
- GET /documents/{id}: found → 200, not found → 404, wrong tenant → 403.
- DELETE /documents/{id}: deleted=True → 200, not found → 404,
                          wrong tenant → 403.
- GET /health:      200 with status=ok (no auth required).
- POST /sync:       unknown source → 404, unconfigured source → 503.
"""

import io
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import jwt
import pytest
from fastapi.testclient import TestClient

from agent.api.app import create_app
from agent.api.auth import _JWT_SECRET, get_rbac_context
from agent.api.dependencies import (
    get_pipeline_service,
    get_registry_service,
    get_retrieval_service,
    get_vector_store,
)
from agent.pipeline.models import PipelineResult
from agent.registry.models import DocumentRecord
from agent.retrieval.models import RBACContext, RetrievedContext, SearchResult
from agent.services.pipeline_service import PipelineService
from agent.services.registry_service import RegistryResult, RegistryService
from agent.services.retrieval_service import RetrievalResult, RetrievalService


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_token(tenant_id="t1", user_id="u1", roles=None, expired=False):
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "roles": roles or [],
        "exp": int(time.time()) + (3600 if not expired else -3600),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm="HS256")


def _make_token_no_tenant():
    payload = {"sub": "u1", "exp": int(time.time()) + 3600}
    return jwt.encode(payload, _JWT_SECRET, algorithm="HS256")


def _mock_rbac(tenant_id="t1", user_id="u1"):
    return RBACContext(tenant_id=tenant_id, user_id=user_id, user_roles=[])


def _sample_record(doc_id="doc-1", tenant_id="t1"):
    return DocumentRecord(
        id=doc_id, file_path="/a.pdf", status="completed",
        tenant_id=tenant_id, owner_id="u1", total_chunks=5
    )


def _mock_pipeline_service(success=True, is_duplicate=False, chunks=3, error=None):
    svc = MagicMock(spec=PipelineService)
    result = PipelineResult(
        success=success,
        file_path="/tmp/x.pdf",
        document_id="doc-new" if success else None,
        total_chunks=chunks,
        is_duplicate=is_duplicate,
        duplicate_of="doc-old" if is_duplicate else None,
        similarity_score=0.91 if is_duplicate else 0.0,
        total_duration_ms=250.0,
        failed_stage=None if success else "chunking",
        error=error,
    )
    svc.run.return_value = result
    return svc


def _mock_registry_service(
    records=None,
    found_record=None,
    hash_record=None,
    delete_result=True,
    error=None,
):
    svc = MagicMock(spec=RegistryService)

    list_result = MagicMock()
    list_result.success = error is None
    list_result.records = records or []
    list_result.error = error
    svc.list_by_tenant.return_value = list_result

    get_result = MagicMock()
    get_result.success = True
    get_result.record = found_record
    get_result.error = None
    svc.get.return_value = get_result

    hash_result = MagicMock()
    hash_result.success = True
    hash_result.record = hash_record
    svc.get_by_file_hash.return_value = hash_result

    # filename lookup always returns None (no existing doc with same name)
    filename_result = MagicMock()
    filename_result.success = True
    filename_result.record = None
    svc.get_by_original_filename.return_value = filename_result

    del_result = MagicMock()
    del_result.success = True
    del_result.deleted = delete_result
    svc.delete.return_value = del_result

    return svc


def _mock_retrieval_service(success=True, error=None):
    svc = MagicMock(spec=RetrievalService)
    if success:
        context = RetrievedContext(
            query="test query",
            chunks=[
                SearchResult(
                    chunk_id="c1", document_id="doc-1",
                    text="Relevant content.", score=0.92, metadata={}
                )
            ],
            window_texts=["Relevant content."],
            retrieval_stats={"total_candidates": 50},
        )
        result = RetrievalResult(success=True, context=context)
    else:
        result = RetrievalResult(success=False, error=error or "retrieval failed")
    svc.query.return_value = result
    return svc


@pytest.fixture()
def client_factory():
    """Returns a factory that creates a TestClient with overridden services."""
    def _make(
        pipeline_svc=None,
        registry_svc=None,
        retrieval_svc=None,
        rbac=None,
    ):
        app = create_app()
        if rbac is not None:
            app.dependency_overrides[get_rbac_context] = lambda: rbac
        if pipeline_svc is not None:
            app.dependency_overrides[get_pipeline_service] = lambda: pipeline_svc
        if registry_svc is not None:
            app.dependency_overrides[get_registry_service] = lambda: registry_svc
        if retrieval_svc is not None:
            app.dependency_overrides[get_retrieval_service] = lambda: retrieval_svc
        # Always mock vector_store so tests don't need a live DB
        app.dependency_overrides[get_vector_store] = lambda: MagicMock()
        return TestClient(app, raise_server_exceptions=False)
    return _make


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_200(self, client_factory):
        client = client_factory()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_no_auth_required(self, client_factory):
        client = client_factory()
        resp = client.get("/health")
        assert resp.status_code == 200


# ── Auth ───────────────────────────────────────────────────────────────────────

class TestAuth:

    def test_missing_token_returns_401(self, client_factory):
        client = client_factory()
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 401

    def test_expired_token_returns_401(self, client_factory):
        client = client_factory()
        token = _make_token(expired=True)
        resp = client.post(
            "/query",
            json={"query": "test"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401

    def test_invalid_token_returns_401(self, client_factory):
        client = client_factory()
        resp = client.post(
            "/query",
            json={"query": "test"},
            headers={"Authorization": "Bearer not-a-jwt"},
        )
        assert resp.status_code == 401

    def test_token_missing_tenant_id_returns_403(self, client_factory):
        client = client_factory()
        token = _make_token_no_tenant()
        resp = client.post(
            "/query",
            json={"query": "test"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_valid_token_accepted(self, client_factory):
        retrieval_svc = _mock_retrieval_service()
        client = client_factory(retrieval_svc=retrieval_svc)
        token = _make_token()
        resp = client.post(
            "/query",
            json={"query": "architecture decision"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200


# ── POST /ingest ───────────────────────────────────────────────────────────────

class TestIngest:

    def _client(self, client_factory, pipeline_svc=None, registry_svc=None):
        return client_factory(
            rbac=_mock_rbac(),
            pipeline_svc=pipeline_svc or _mock_pipeline_service(),
            registry_svc=registry_svc or _mock_registry_service(),
        )

    def test_success_200(self, client_factory):
        client = self._client(client_factory)
        resp = client.post(
            "/ingest",
            files={"file": ("report.txt", b"Enterprise content", "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_document_id_in_response(self, client_factory):
        client = self._client(client_factory)
        resp = client.post(
            "/ingest",
            files={"file": ("report.txt", b"content", "text/plain")},
        )
        assert resp.json()["document_id"] == "doc-new"

    def test_total_chunks_in_response(self, client_factory):
        client = self._client(client_factory)
        resp = client.post(
            "/ingest",
            files={"file": ("report.txt", b"content", "text/plain")},
        )
        assert resp.json()["total_chunks"] == 3

    def test_empty_file_returns_400(self, client_factory):
        client = self._client(client_factory)
        resp = client.post(
            "/ingest",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 400

    def test_pipeline_failure_in_response(self, client_factory):
        pipeline_svc = _mock_pipeline_service(success=False, error="GPU OOM")
        client = self._client(client_factory, pipeline_svc=pipeline_svc)
        resp = client.post(
            "/ingest",
            files={"file": ("report.txt", b"content", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["error"] == "GPU OOM"

    def test_exact_duplicate_skips_pipeline(self, client_factory):
        existing_record = _sample_record("doc-already")
        registry_svc = _mock_registry_service(hash_record=existing_record)
        pipeline_svc = _mock_pipeline_service()
        client = self._client(client_factory, pipeline_svc=pipeline_svc,
                              registry_svc=registry_svc)
        resp = client.post(
            "/ingest",
            files={"file": ("report.txt", b"same bytes", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_duplicate"] is True
        assert data["document_id"] == "doc-already"
        pipeline_svc.run.assert_not_called()

    def test_near_duplicate_result_propagated(self, client_factory):
        pipeline_svc = _mock_pipeline_service(success=True, is_duplicate=True)
        client = self._client(client_factory, pipeline_svc=pipeline_svc)
        resp = client.post(
            "/ingest",
            files={"file": ("report.txt", b"similar content", "text/plain")},
        )
        data = resp.json()
        assert data["is_duplicate"] is True
        assert data["duplicate_of"] == "doc-old"

    def test_visibility_form_field_accepted(self, client_factory):
        pipeline_svc = _mock_pipeline_service()
        client = self._client(client_factory, pipeline_svc=pipeline_svc)
        client.post(
            "/ingest",
            data={"visibility": "restricted"},
            files={"file": ("report.txt", b"content", "text/plain")},
        )
        kwargs = pipeline_svc.run.call_args.kwargs
        assert kwargs["visibility"] == "restricted"

    def test_access_roles_parsed(self, client_factory):
        pipeline_svc = _mock_pipeline_service()
        client = self._client(client_factory, pipeline_svc=pipeline_svc)
        client.post(
            "/ingest",
            data={"access_roles": "hr,legal"},
            files={"file": ("report.txt", b"content", "text/plain")},
        )
        kwargs = pipeline_svc.run.call_args.kwargs
        assert set(kwargs["access_roles"]) == {"hr", "legal"}


# ── POST /query ────────────────────────────────────────────────────────────────

class TestQuery:

    def _client(self, client_factory, retrieval_svc=None):
        return client_factory(
            rbac=_mock_rbac(),
            retrieval_svc=retrieval_svc or _mock_retrieval_service(),
        )

    def test_success_200(self, client_factory):
        client = self._client(client_factory)
        resp = client.post("/query", json={"query": "architecture decision"})
        assert resp.status_code == 200

    def test_chunks_returned(self, client_factory):
        client = self._client(client_factory)
        resp = client.post("/query", json={"query": "deployment guide"})
        data = resp.json()
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["chunk_id"] == "c1"
        assert data["chunks"][0]["score"] == pytest.approx(0.92)

    def test_query_echoed_in_response(self, client_factory):
        client = self._client(client_factory)
        resp = client.post("/query", json={"query": "deployment guide"})
        assert resp.json()["query"] == "test query"

    def test_empty_query_returns_422(self, client_factory):
        client = self._client(client_factory)
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422

    def test_missing_query_returns_422(self, client_factory):
        client = self._client(client_factory)
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_top_k_too_large_returns_422(self, client_factory):
        client = self._client(client_factory)
        resp = client.post("/query", json={"query": "test", "top_k": 999})
        assert resp.status_code == 422

    def test_retrieval_failure_returns_500(self, client_factory):
        retrieval_svc = _mock_retrieval_service(success=False, error="vector DB down")
        client = self._client(client_factory, retrieval_svc=retrieval_svc)
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 500


# ── GET /documents ─────────────────────────────────────────────────────────────

class TestListDocuments:

    def _client(self, client_factory, records=None, error=None):
        return client_factory(
            rbac=_mock_rbac(),
            registry_svc=_mock_registry_service(records=records, error=error),
        )

    def test_success_200(self, client_factory):
        client = self._client(client_factory, records=[_sample_record()])
        resp = client.get("/documents")
        assert resp.status_code == 200

    def test_documents_returned(self, client_factory):
        client = self._client(client_factory, records=[_sample_record("d1"), _sample_record("d2")])
        data = client.get("/documents").json()
        assert len(data["documents"]) == 2
        assert data["total"] == 2

    def test_empty_list(self, client_factory):
        client = self._client(client_factory, records=[])
        data = client.get("/documents").json()
        assert data["documents"] == []

    def test_status_filter_forwarded(self, client_factory):
        registry_svc = _mock_registry_service()
        client = client_factory(rbac=_mock_rbac(), registry_svc=registry_svc)
        client.get("/documents?status=failed_embedding")
        kwargs = registry_svc.list_by_tenant.call_args.kwargs
        assert kwargs["status"] == "failed_embedding"

    def test_pagination_forwarded(self, client_factory):
        registry_svc = _mock_registry_service()
        client = client_factory(rbac=_mock_rbac(), registry_svc=registry_svc)
        client.get("/documents?limit=10&offset=20")
        kwargs = registry_svc.list_by_tenant.call_args.kwargs
        assert kwargs["limit"] == 10
        assert kwargs["offset"] == 20

    def test_registry_error_returns_500(self, client_factory):
        client = self._client(client_factory, error="DB timeout")
        resp = client.get("/documents")
        assert resp.status_code == 500


# ── GET /documents/{id} ────────────────────────────────────────────────────────

class TestGetDocument:

    def test_found_returns_200(self, client_factory):
        registry_svc = _mock_registry_service(found_record=_sample_record())
        client = client_factory(rbac=_mock_rbac(), registry_svc=registry_svc)
        resp = client.get("/documents/doc-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "doc-1"

    def test_not_found_returns_404(self, client_factory):
        registry_svc = _mock_registry_service(found_record=None)
        client = client_factory(rbac=_mock_rbac(), registry_svc=registry_svc)
        resp = client.get("/documents/nonexistent")
        assert resp.status_code == 404

    def test_wrong_tenant_returns_403(self, client_factory):
        # Record belongs to tenant-other, but caller's RBAC is tenant t1
        record = _sample_record(tenant_id="tenant-other")
        registry_svc = _mock_registry_service(found_record=record)
        client = client_factory(rbac=_mock_rbac(tenant_id="t1"), registry_svc=registry_svc)
        resp = client.get("/documents/doc-1")
        assert resp.status_code == 403


# ── DELETE /documents/{id} ─────────────────────────────────────────────────────

class TestDeleteDocument:

    def test_deleted_returns_200(self, client_factory):
        registry_svc = _mock_registry_service(
            found_record=_sample_record(), delete_result=True
        )
        client = client_factory(rbac=_mock_rbac(), registry_svc=registry_svc)
        resp = client.delete("/documents/doc-1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert resp.json()["document_id"] == "doc-1"

    def test_not_found_returns_404(self, client_factory):
        registry_svc = _mock_registry_service(found_record=None)
        client = client_factory(rbac=_mock_rbac(), registry_svc=registry_svc)
        resp = client.delete("/documents/nonexistent")
        assert resp.status_code == 404

    def test_wrong_tenant_returns_403(self, client_factory):
        record = _sample_record(tenant_id="other-tenant")
        registry_svc = _mock_registry_service(found_record=record)
        client = client_factory(rbac=_mock_rbac(tenant_id="t1"), registry_svc=registry_svc)
        resp = client.delete("/documents/doc-1")
        assert resp.status_code == 403


# ── POST /sync ─────────────────────────────────────────────────────────────────

class TestSync:

    def test_unknown_source_returns_404(self, client_factory):
        client = client_factory(
            rbac=_mock_rbac(),
            pipeline_svc=_mock_pipeline_service(),
            registry_svc=_mock_registry_service(),
        )
        resp = client.post("/sync/notion", json={})
        assert resp.status_code == 404

    def test_unconfigured_confluence_returns_503(self, client_factory, monkeypatch):
        monkeypatch.delenv("CONFLUENCE_URL", raising=False)
        monkeypatch.delenv("CONFLUENCE_TOKEN", raising=False)
        client = client_factory(
            rbac=_mock_rbac(),
            pipeline_svc=_mock_pipeline_service(),
            registry_svc=_mock_registry_service(),
        )
        resp = client.post("/sync/confluence", json={})
        assert resp.status_code == 503

    def test_unconfigured_jira_returns_503(self, client_factory, monkeypatch):
        monkeypatch.delenv("JIRA_URL", raising=False)
        monkeypatch.delenv("JIRA_TOKEN", raising=False)
        client = client_factory(
            rbac=_mock_rbac(),
            pipeline_svc=_mock_pipeline_service(),
            registry_svc=_mock_registry_service(),
        )
        resp = client.post("/sync/jira", json={})
        assert resp.status_code == 503

    def test_unconfigured_wiki_returns_503(self, client_factory, monkeypatch):
        monkeypatch.delenv("WIKI_HOST", raising=False)
        client = client_factory(
            rbac=_mock_rbac(),
            pipeline_svc=_mock_pipeline_service(),
            registry_svc=_mock_registry_service(),
        )
        resp = client.post("/sync/wiki", json={})
        assert resp.status_code == 503

    def test_case_insensitive_source(self, client_factory, monkeypatch):
        monkeypatch.delenv("CONFLUENCE_URL", raising=False)
        client = client_factory(
            rbac=_mock_rbac(),
            pipeline_svc=_mock_pipeline_service(),
            registry_svc=_mock_registry_service(),
        )
        resp = client.post("/sync/CONFLUENCE", json={})
        assert resp.status_code == 503  # configured but unconfigured, not 404
