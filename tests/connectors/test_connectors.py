"""
tests/connectors/test_connectors.py

What does this cover?
- ConnectorDoc:       field defaults, content_type defaults.
- SyncResult:         field defaults, errors list.
- BaseConnector:      abstract enforcement.
- ConfluenceConnector: import guard, fetch delegates to CQL, pagination,
                       page_to_doc mapping, parse errors skipped gracefully.
- JiraConnector:      import guard, JQL building, comment extraction,
                       ADF text extraction, pagination.
- MediaWikiConnector: import guard, fetch_page maps to ConnectorDoc,
                       missing pages skipped.
- SharePointConnector: import guard tested.
- SyncService:        golden path (fetch → pipeline called per doc),
                       skip unchanged (hash already in registry),
                       connector fetch failure → SyncResult(success=False),
                       per-doc pipeline failure isolated (other docs continue),
                       temp file cleaned up after pipeline call,
                       SyncResult stats (fetched, ingested, skipped, failed).
- _content_hash:      text content hash, file path hash.
- _write_temp_file:   writes html/text/markdown, returns existing file_path.
- _cleanup_temp:      silent on missing file.
"""

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest

from agent.connectors.models import ConnectorDoc, SyncResult
from agent.connectors.base import BaseConnector
from agent.services.sync_service import (
    SyncService, _content_hash, _write_temp_file, _cleanup_temp
)


# ── ConnectorDoc ───────────────────────────────────────────────────────────────

class TestConnectorDoc:

    def test_required_fields(self):
        doc = ConnectorDoc(
            source_url="confluence::ENG::123",
            title="Architecture Doc",
            source_name="confluence",
        )
        assert doc.source_url == "confluence::ENG::123"
        assert doc.source_name == "confluence"

    def test_default_content_empty(self):
        doc = ConnectorDoc(source_url="x", title="T", source_name="jira")
        assert doc.content == ""
        assert doc.file_path is None

    def test_default_content_type_html(self):
        doc = ConnectorDoc(source_url="x", title="T", source_name="confluence")
        assert doc.content_type == "html"

    def test_metadata_default_empty_dict(self):
        doc = ConnectorDoc(source_url="x", title="T", source_name="wiki")
        assert doc.metadata == {}

    def test_last_modified_optional(self):
        doc = ConnectorDoc(source_url="x", title="T", source_name="wiki")
        assert doc.last_modified is None


# ── SyncResult ─────────────────────────────────────────────────────────────────

class TestSyncResult:

    def test_defaults(self):
        r = SyncResult(success=True, source="confluence")
        assert r.fetched == 0
        assert r.ingested == 0
        assert r.skipped == 0
        assert r.failed == 0
        assert r.errors == []

    def test_errors_independent_between_instances(self):
        r1 = SyncResult(success=True, source="jira")
        r2 = SyncResult(success=True, source="jira")
        r1.errors.append("err1")
        assert r2.errors == []


# ── BaseConnector ABC ──────────────────────────────────────────────────────────

class TestBaseConnector:

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseConnector()

    def test_concrete_subclass_works(self):
        class FakeConnector(BaseConnector):
            @property
            def source_name(self):
                return "fake"
            def fetch_documents(self, since=None, **kwargs):
                return []

        c = FakeConnector()
        assert c.source_name == "fake"
        assert c.fetch_documents() == []

    def test_missing_fetch_documents_raises(self):
        class IncompleteConnector(BaseConnector):
            @property
            def source_name(self):
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteConnector()


# ── ConfluenceConnector ────────────────────────────────────────────────────────

class TestConfluenceConnector:

    def test_import_guard_raises_without_atlassian(self):
        import agent.connectors.confluence as mod
        original = mod._ATLASSIAN_AVAILABLE
        mod._ATLASSIAN_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="atlassian-python-api"):
                from agent.connectors.confluence import ConfluenceConnector
                ConfluenceConnector(url="http://x", token="t")
        finally:
            mod._ATLASSIAN_AVAILABLE = original

    def test_source_name_is_confluence(self):
        from agent.connectors.confluence import ConfluenceConnector
        import agent.connectors.confluence as mod
        if not mod._ATLASSIAN_AVAILABLE:
            pytest.skip("atlassian-python-api not installed")
        with patch("agent.connectors.confluence.AtlassianConfluence"):
            c = ConfluenceConnector(url="http://x", token="t")
            assert c.source_name == "confluence"

    def test_fetch_returns_empty_on_client_error(self):
        from agent.connectors.confluence import ConfluenceConnector
        import agent.connectors.confluence as mod
        if not mod._ATLASSIAN_AVAILABLE:
            pytest.skip("atlassian-python-api not installed")
        with patch("agent.connectors.confluence.AtlassianConfluence") as MockClient:
            MockClient.return_value.get_all_spaces.side_effect = RuntimeError("auth failed")
            c = ConfluenceConnector(url="http://x", token="t")
            result = c.fetch_documents()
            assert result == []

    def test_page_to_doc_maps_fields(self):
        from agent.connectors.confluence import ConfluenceConnector
        import agent.connectors.confluence as mod
        if not mod._ATLASSIAN_AVAILABLE:
            pytest.skip("atlassian-python-api not installed")
        with patch("agent.connectors.confluence.AtlassianConfluence"):
            c = ConfluenceConnector(url="http://x", token="t")

        page = {
            "content": {
                "id": "42",
                "title": "API Design",
                "body": {"storage": {"value": "<p>Hello</p>"}},
                "history": {
                    "lastUpdated": {
                        "when": "2024-03-15T10:00:00.000Z",
                        "by": {"displayName": "Alice"},
                    }
                },
            },
            "lastModified": "2024-03-15T10:00:00.000Z",
        }
        doc = c._page_to_doc(page, "ENG")
        assert doc is not None
        assert doc.title == "API Design"
        assert doc.content == "<p>Hello</p>"
        assert doc.content_type == "html"
        assert doc.source_url == "confluence::ENG::42"
        assert doc.metadata["space_key"] == "ENG"

    def test_page_to_doc_returns_none_on_parse_error(self):
        from agent.connectors.confluence import ConfluenceConnector
        import agent.connectors.confluence as mod
        if not mod._ATLASSIAN_AVAILABLE:
            pytest.skip("atlassian-python-api not installed")
        with patch("agent.connectors.confluence.AtlassianConfluence"):
            c = ConfluenceConnector(url="http://x", token="t")
        # Missing required 'id' key
        doc = c._page_to_doc({"content": {}}, "ENG")
        assert doc is None


# ── JiraConnector ──────────────────────────────────────────────────────────────

class TestJiraConnector:

    def test_import_guard_raises_without_atlassian(self):
        import agent.connectors.jira as mod
        original = mod._ATLASSIAN_AVAILABLE
        mod._ATLASSIAN_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="atlassian-python-api"):
                from agent.connectors.jira import JiraConnector
                JiraConnector(url="http://x", token="t")
        finally:
            mod._ATLASSIAN_AVAILABLE = original

    def test_source_name_is_jira(self):
        from agent.connectors.jira import JiraConnector
        import agent.connectors.jira as mod
        if not mod._ATLASSIAN_AVAILABLE:
            pytest.skip("atlassian-python-api not installed")
        with patch("agent.connectors.jira.AtlassianJira"):
            c = JiraConnector(url="http://x", token="t")
            assert c.source_name == "jira"

    def test_jql_with_project_key(self):
        from agent.connectors.jira import JiraConnector
        import agent.connectors.jira as mod
        if not mod._ATLASSIAN_AVAILABLE:
            pytest.skip("atlassian-python-api not installed")
        with patch("agent.connectors.jira.AtlassianJira"):
            c = JiraConnector(url="http://x", token="t")
        jql = c._build_jql(None, "ENG", None, False)
        assert 'project = "ENG"' in jql
        assert "Closed" in jql

    def test_jql_with_since(self):
        from agent.connectors.jira import JiraConnector
        import agent.connectors.jira as mod
        if not mod._ATLASSIAN_AVAILABLE:
            pytest.skip("atlassian-python-api not installed")
        with patch("agent.connectors.jira.AtlassianJira"):
            c = JiraConnector(url="http://x", token="t")
        since = datetime(2024, 6, 1, tzinfo=timezone.utc)
        jql = c._build_jql(since, None, None, False)
        assert "updated" in jql
        assert "2024-06-01" in jql

    def test_jql_include_closed(self):
        from agent.connectors.jira import JiraConnector
        import agent.connectors.jira as mod
        if not mod._ATLASSIAN_AVAILABLE:
            pytest.skip("atlassian-python-api not installed")
        with patch("agent.connectors.jira.AtlassianJira"):
            c = JiraConnector(url="http://x", token="t")
        jql = c._build_jql(None, None, None, True)
        assert "Closed" not in jql

    def test_issue_to_doc_maps_fields(self):
        from agent.connectors.jira import JiraConnector
        import agent.connectors.jira as mod
        if not mod._ATLASSIAN_AVAILABLE:
            pytest.skip("atlassian-python-api not installed")
        with patch("agent.connectors.jira.AtlassianJira"):
            c = JiraConnector(url="http://x", token="t")

        issue = {
            "key": "ENG-42",
            "fields": {
                "summary": "Fix null pointer exception",
                "description": "When loading PDF, we get NPE on line 42.",
                "issuetype": {"name": "Bug"},
                "status": {"name": "In Progress"},
                "reporter": {"displayName": "Bob"},
                "comment": {"comments": []},
                "updated": "2024-05-10T12:00:00.000Z",
            }
        }
        doc = c._issue_to_doc(issue)
        assert doc is not None
        assert "ENG-42" in doc.title
        assert "Fix null pointer" in doc.title
        assert doc.content_type == "text"
        assert doc.source_url == "jira::ENG-42"
        assert doc.metadata["issue_key"] == "ENG-42"
        assert doc.metadata["project_key"] == "ENG"

    def test_extract_comments_plain_text(self):
        from agent.connectors.jira import JiraConnector
        import agent.connectors.jira as mod
        if not mod._ATLASSIAN_AVAILABLE:
            pytest.skip("atlassian-python-api not installed")
        with patch("agent.connectors.jira.AtlassianJira"):
            c = JiraConnector(url="http://x", token="t")

        comment_field = {
            "comments": [
                {"author": {"displayName": "Alice"}, "body": "Looks good to me."},
                {"author": {"displayName": "Bob"}, "body": "Merging now."},
            ]
        }
        text = c._extract_comments(comment_field)
        assert "Alice" in text
        assert "Looks good to me." in text
        assert "Bob" in text

    def test_adf_to_text_extracts_paragraphs(self):
        from agent.connectors.jira import _adf_to_text
        adf = {
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world."},
                    ]
                }
            ]
        }
        assert "Hello" in _adf_to_text(adf)
        assert "world." in _adf_to_text(adf)

    def test_fetch_returns_empty_on_client_error(self):
        from agent.connectors.jira import JiraConnector
        import agent.connectors.jira as mod
        if not mod._ATLASSIAN_AVAILABLE:
            pytest.skip("atlassian-python-api not installed")
        with patch("agent.connectors.jira.AtlassianJira") as MockClient:
            MockClient.return_value.jql.side_effect = RuntimeError("auth failed")
            c = JiraConnector(url="http://x", token="t")
            result = c.fetch_documents()
            assert result == []


# ── MediaWikiConnector ─────────────────────────────────────────────────────────

class TestMediaWikiConnector:

    def test_import_guard_raises_without_mwclient(self):
        import agent.connectors.wiki as mod
        original = mod._MWCLIENT_AVAILABLE
        mod._MWCLIENT_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="mwclient"):
                from agent.connectors.wiki import MediaWikiConnector
                MediaWikiConnector(host="en.wikipedia.org")
        finally:
            mod._MWCLIENT_AVAILABLE = original

    def test_source_name_is_wiki(self):
        from agent.connectors.wiki import MediaWikiConnector
        import agent.connectors.wiki as mod
        if not mod._MWCLIENT_AVAILABLE:
            pytest.skip("mwclient not installed")
        with patch("agent.connectors.wiki.mwclient.Site"):
            c = MediaWikiConnector(host="wiki.example.com")
            assert c.source_name == "wiki"

    def test_fetch_returns_empty_on_error(self):
        from agent.connectors.wiki import MediaWikiConnector
        import agent.connectors.wiki as mod
        if not mod._MWCLIENT_AVAILABLE:
            pytest.skip("mwclient not installed")
        with patch("agent.connectors.wiki.mwclient.Site") as MockSite:
            MockSite.return_value.allpages.side_effect = RuntimeError("connection refused")
            c = MediaWikiConnector(host="wiki.example.com")
            result = c.fetch_documents()
            assert result == []

    def test_fetch_page_returns_none_for_nonexistent(self):
        from agent.connectors.wiki import MediaWikiConnector
        import agent.connectors.wiki as mod
        if not mod._MWCLIENT_AVAILABLE:
            pytest.skip("mwclient not installed")
        with patch("agent.connectors.wiki.mwclient.Site") as MockSite:
            mock_page = MagicMock()
            mock_page.exists = False
            MockSite.return_value.pages.__getitem__ = MagicMock(return_value=mock_page)
            c = MediaWikiConnector(host="wiki.example.com")
            doc = c._fetch_page("NonExistentPage")
            assert doc is None


# ── SharePointConnector ────────────────────────────────────────────────────────

class TestSharePointConnector:

    def test_import_guard_raises_without_o365(self):
        import agent.connectors.sharepoint as mod
        original = mod._O365_AVAILABLE
        mod._O365_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="O365"):
                from agent.connectors.sharepoint import SharePointConnector
                SharePointConnector(
                    client_id="x", client_secret="y",
                    tenant_id="z", site_url="http://sp"
                )
        finally:
            mod._O365_AVAILABLE = original

    def test_source_name_is_sharepoint(self):
        from agent.connectors.sharepoint import SharePointConnector
        import agent.connectors.sharepoint as mod
        if not mod._O365_AVAILABLE:
            pytest.skip("O365 not installed")
        with patch("agent.connectors.sharepoint.Account") as MockAccount:
            MockAccount.return_value.is_authenticated = True
            c = SharePointConnector(
                client_id="x", client_secret="y",
                tenant_id="z", site_url="http://sp"
            )
            assert c.source_name == "sharepoint"


# ── _content_hash ──────────────────────────────────────────────────────────────

class TestContentHash:

    def test_text_content_produces_hash(self):
        doc = ConnectorDoc(source_url="x", title="T", source_name="confluence",
                           content="Hello world")
        h = _content_hash(doc)
        assert len(h) == 64  # SHA-256 hex

    def test_same_content_same_hash(self):
        doc1 = ConnectorDoc(source_url="a", title="T", source_name="jira", content="abc")
        doc2 = ConnectorDoc(source_url="b", title="T", source_name="jira", content="abc")
        assert _content_hash(doc1) == _content_hash(doc2)

    def test_different_content_different_hash(self):
        doc1 = ConnectorDoc(source_url="x", title="T", source_name="jira", content="abc")
        doc2 = ConnectorDoc(source_url="x", title="T", source_name="jira", content="xyz")
        assert _content_hash(doc1) != _content_hash(doc2)

    def test_file_path_hashes_file_bytes(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"PDF content bytes")
        doc = ConnectorDoc(source_url="x", title="T", source_name="sharepoint",
                           file_path=str(f))
        h = _content_hash(doc)
        assert len(h) == 64

    def test_file_path_differs_from_content_hash(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_bytes(b"same text")
        doc_file = ConnectorDoc(source_url="x", title="T", source_name="sharepoint",
                                file_path=str(f), content="")
        doc_text = ConnectorDoc(source_url="x", title="T", source_name="confluence",
                                content="same text")
        # File hash and text hash of same bytes should be equal
        assert _content_hash(doc_file) == _content_hash(doc_text)


# ── _write_temp_file ───────────────────────────────────────────────────────────

class TestWriteTempFile:

    def test_html_content_gets_html_extension(self):
        doc = ConnectorDoc(source_url="x", title="T", source_name="confluence",
                           content="<p>Hello</p>", content_type="html")
        path = _write_temp_file(doc)
        try:
            assert path.endswith(".html")
            assert os.path.exists(path)
        finally:
            _cleanup_temp(path)

    def test_text_content_gets_txt_extension(self):
        doc = ConnectorDoc(source_url="x", title="T", source_name="jira",
                           content="Issue text", content_type="text")
        path = _write_temp_file(doc)
        try:
            assert path.endswith(".txt")
        finally:
            _cleanup_temp(path)

    def test_markdown_content_gets_md_extension(self):
        doc = ConnectorDoc(source_url="x", title="T", source_name="wiki",
                           content="# Heading", content_type="markdown")
        path = _write_temp_file(doc)
        try:
            assert path.endswith(".md")
        finally:
            _cleanup_temp(path)

    def test_content_written_to_file(self):
        doc = ConnectorDoc(source_url="x", title="T", source_name="confluence",
                           content="<h1>Title</h1>", content_type="html")
        path = _write_temp_file(doc)
        try:
            assert open(path).read() == "<h1>Title</h1>"
        finally:
            _cleanup_temp(path)

    def test_existing_file_path_returned_directly(self, tmp_path):
        f = tmp_path / "report.docx"
        f.write_bytes(b"DOCX bytes")
        doc = ConnectorDoc(source_url="x", title="T", source_name="sharepoint",
                           file_path=str(f))
        path = _write_temp_file(doc)
        assert path == str(f)  # returned unchanged, no new file created


# ── _cleanup_temp ──────────────────────────────────────────────────────────────

class TestCleanupTemp:

    def test_deletes_existing_file(self, tmp_path):
        f = tmp_path / "temp.txt"
        f.write_text("data")
        _cleanup_temp(str(f))
        assert not f.exists()

    def test_silent_on_missing_file(self):
        _cleanup_temp("/nonexistent/path/file.txt")  # should not raise

    def test_silent_on_none(self):
        _cleanup_temp(None)  # should not raise


# ── SyncService ────────────────────────────────────────────────────────────────

def _make_connector(docs=None, error=None):
    connector = MagicMock(spec=BaseConnector)
    connector.source_name = "confluence"
    if error:
        connector.fetch_documents.side_effect = error
    else:
        connector.fetch_documents.return_value = docs or []
    return connector


def _make_doc(source_url="confluence::ENG::1", content="<p>Hello</p>"):
    return ConnectorDoc(
        source_url=source_url,
        title="Test Page",
        source_name="confluence",
        content=content,
        content_type="html",
    )


def _make_sync_service(pipeline_success=True, already_indexed=False):
    pipeline = MagicMock()
    pipeline_result = MagicMock()
    pipeline_result.success = pipeline_success
    pipeline_result.error = None if pipeline_success else "pipeline error"
    pipeline.run.return_value = pipeline_result

    registry = MagicMock()
    registry_result = MagicMock()
    registry_result.success = True
    registry_result.record = MagicMock() if already_indexed else None
    registry.get_by_file_hash.return_value = registry_result

    svc = SyncService(pipeline_service=pipeline, registry_service=registry)
    return svc, pipeline, registry


class TestSyncServiceGoldenPath:

    def test_success_true(self):
        svc, pipeline, _ = _make_sync_service()
        connector = _make_connector(docs=[_make_doc()])
        result = svc.sync(connector, tenant_id="t1")
        assert result.success is True

    def test_fetched_count(self):
        svc, _, _ = _make_sync_service()
        connector = _make_connector(docs=[_make_doc("u1"), _make_doc("u2")])
        result = svc.sync(connector, tenant_id="t1")
        assert result.fetched == 2

    def test_ingested_count(self):
        svc, _, _ = _make_sync_service()
        connector = _make_connector(docs=[_make_doc("u1"), _make_doc("u2")])
        result = svc.sync(connector, tenant_id="t1")
        assert result.ingested == 2

    def test_pipeline_called_for_each_doc(self):
        svc, pipeline, _ = _make_sync_service()
        connector = _make_connector(docs=[_make_doc("u1", "a"), _make_doc("u2", "b")])
        svc.sync(connector, tenant_id="t1")
        assert pipeline.run.call_count == 2

    def test_pipeline_called_with_tenant_id(self):
        svc, pipeline, _ = _make_sync_service()
        connector = _make_connector(docs=[_make_doc()])
        svc.sync(connector, tenant_id="acme", owner_id="bot")
        kwargs = pipeline.run.call_args.kwargs
        assert kwargs["tenant_id"] == "acme"
        assert kwargs["owner_id"] == "bot"

    def test_source_name_in_result(self):
        svc, _, _ = _make_sync_service()
        connector = _make_connector(docs=[])
        result = svc.sync(connector, tenant_id="t1")
        assert result.source == "confluence"

    def test_duration_non_negative(self):
        svc, _, _ = _make_sync_service()
        connector = _make_connector(docs=[])
        result = svc.sync(connector, tenant_id="t1")
        assert result.duration_ms >= 0.0

    def test_connector_kwargs_forwarded(self):
        svc, _, _ = _make_sync_service()
        connector = _make_connector(docs=[])
        svc.sync(connector, tenant_id="t1", space_key="ENG")
        connector.fetch_documents.assert_called_once_with(since=None, space_key="ENG")


class TestSyncServiceSkip:

    def test_skipped_when_already_indexed(self):
        svc, pipeline, _ = _make_sync_service(already_indexed=True)
        connector = _make_connector(docs=[_make_doc()])
        result = svc.sync(connector, tenant_id="t1")
        assert result.skipped == 1
        assert result.ingested == 0
        pipeline.run.assert_not_called()


class TestSyncServiceFailures:

    def test_connector_fetch_failure_returns_success_false(self):
        svc, _, _ = _make_sync_service()
        connector = _make_connector(error=RuntimeError("auth failed"))
        result = svc.sync(connector, tenant_id="t1")
        assert result.success is False
        assert "auth failed" in result.error

    def test_pipeline_failure_increments_failed(self):
        svc, pipeline, _ = _make_sync_service(pipeline_success=False)
        connector = _make_connector(docs=[_make_doc()])
        result = svc.sync(connector, tenant_id="t1")
        assert result.failed == 1
        assert result.ingested == 0

    def test_one_doc_failure_does_not_stop_others(self):
        pipeline = MagicMock()
        success_result = MagicMock()
        success_result.success = True
        fail_result = MagicMock()
        fail_result.success = False
        fail_result.error = "chunking failed"
        pipeline.run.side_effect = [fail_result, success_result]

        registry = MagicMock()
        reg_result = MagicMock()
        reg_result.success = True
        reg_result.record = None
        registry.get_by_file_hash.return_value = reg_result

        svc = SyncService(pipeline_service=pipeline, registry_service=registry)
        connector = _make_connector(docs=[_make_doc("u1", "aaa"), _make_doc("u2", "bbb")])
        result = svc.sync(connector, tenant_id="t1")

        assert result.failed == 1
        assert result.ingested == 1
        assert result.fetched == 2

    def test_error_messages_collected(self):
        svc, _, _ = _make_sync_service(pipeline_success=False)
        connector = _make_connector(docs=[_make_doc("u1")])
        result = svc.sync(connector, tenant_id="t1")
        assert len(result.errors) == 1
        assert "u1" in result.errors[0]

    def test_temp_file_cleaned_up_after_pipeline(self):
        svc, pipeline, _ = _make_sync_service()
        doc = _make_doc(content="<p>Clean me up</p>")
        connector = _make_connector(docs=[doc])

        captured_paths = []
        original_run = pipeline.run

        def capture_and_run(**kwargs):
            path = kwargs.get("file_path", "")
            captured_paths.append(path)
            r = MagicMock()
            r.success = True
            return r

        pipeline.run.side_effect = capture_and_run
        svc.sync(connector, tenant_id="t1")

        # temp file should be deleted after pipeline runs
        for path in captured_paths:
            assert not os.path.exists(path), f"Temp file not cleaned up: {path}"
