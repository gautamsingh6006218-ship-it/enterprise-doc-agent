"""
connectors/confluence.py

What problem does this solve?
- Enterprise teams store most of their knowledge in Confluence spaces (runbooks,
  architecture docs, onboarding guides, policy pages). Without a Confluence
  connector, operators must manually export pages to PDF/HTML — a process that
  breaks incremental sync and misses page updates.

Why atlassian-python-api over LangChain's ConfluenceLoader?
- atlassian-python-api gives direct access to the raw REST API response
  including last_modified timestamps and author info. LangChain's loader
  abstracts these away, which prevents incremental sync (we can't tell
  which pages changed since the last run).
- Fewer dependencies: we already have our own pipeline; we don't need
  LangChain's loader abstraction on top.

Why return HTML content_type?
- Confluence pages are stored and returned as HTML. Our existing HtmlLoader
  (used by LoaderRegistry) already strips scripts, extracts clean text, and
  handles title extraction from <title> tags. No new cleaning code needed.

Authentication:
- Cloud Confluence: url + email + api_token (personal access token from
  https://id.atlassian.com/manage-profile/security/api-tokens)
- Server/Data Center: url + username + password (or PAT)
"""

import logging
from datetime import datetime, timezone

from agent.connectors.base import BaseConnector
from agent.connectors.models import ConnectorDoc

logger = logging.getLogger(__name__)

try:
    from atlassian import Confluence as AtlassianConfluence
    _ATLASSIAN_AVAILABLE = True
except ImportError:
    _ATLASSIAN_AVAILABLE = False


class ConfluenceConnector(BaseConnector):
    """
    What problem does this solve?
    - Fetches Confluence pages as HTML ConnectorDocs so SyncService can
      push them through the existing ingestion pipeline without any changes
      to the pipeline itself.

    Why paginate with limit=50?
    - Confluence API caps at 100 results per request. 50 is conservative:
      it reduces the risk of timeouts on slow instances with large pages
      (pages with many macros/attachments take longer to serialise).
      For spaces with thousands of pages, the connector paginates automatically.

    Why include space_key in metadata?
    - Enables RBAC at the space level: different Confluence spaces often map
      to different teams (Engineering vs HR vs Legal). Storing space_key in
      chunk metadata lets the retrieval layer filter by space.
    """

    _PAGE_LIMIT = 50

    def __init__(
        self,
        url: str,
        token: str,
        username: str = "",
        cloud: bool = True,
    ) -> None:
        """
        Args:
        - url:      Confluence base URL, e.g. "https://myorg.atlassian.net/wiki"
        - token:    API token (cloud) or password (server).
        - username: Email address (cloud) or username (server).
        - cloud:    True for Atlassian Cloud, False for Server/Data Center.
        """
        if not _ATLASSIAN_AVAILABLE:
            raise ImportError(
                "atlassian-python-api is required for ConfluenceConnector. "
                "Install it with: pip install atlassian-python-api"
            )
        self._client = AtlassianConfluence(
            url=url,
            username=username,
            password=token,
            cloud=cloud,
        )

    @property
    def source_name(self) -> str:
        return "confluence"

    def fetch_documents(
        self,
        since: datetime | None = None,
        space_key: str | None = None,
        **kwargs,
    ) -> list[ConnectorDoc]:
        """
        What problem does this solve?
        - Fetches all pages from one or all Confluence spaces and returns
          them as ConnectorDoc objects for SyncService to ingest.

        Args:
        - since:      Only return pages modified after this datetime (UTC).
                      None = full sync.
        - space_key:  Limit to a single space (e.g. "ENG", "HR"). None = all spaces.

        Why CQL instead of the pages API?
        - CQL (Confluence Query Language) supports lastModified filtering
          natively. The pages REST endpoint has no date filter — we'd have
          to fetch everything and filter client-side (expensive for large wikis).
        """
        try:
            spaces = (
                [{"key": space_key}]
                if space_key
                else self._get_all_spaces()
            )
            docs = []
            for space in spaces:
                docs.extend(self._fetch_space(space["key"], since))
            return docs
        except Exception as e:
            logger.error("Confluence fetch failed: %s", e)
            return []

    # ── Private ────────────────────────────────────────────────────────────────

    def _get_all_spaces(self) -> list[dict]:
        spaces, start = [], 0
        while True:
            batch = self._client.get_all_spaces(start=start, limit=50)
            results = batch.get("results", [])
            spaces.extend(results)
            if len(results) < 50:
                break
            start += 50
        return spaces

    def _fetch_space(
        self, space_key: str, since: datetime | None
    ) -> list[ConnectorDoc]:
        cql = f'space = "{space_key}" AND type = page'
        if since:
            ts = since.strftime("%Y-%m-%d %H:%M")
            cql += f' AND lastModified >= "{ts}"'

        docs, start = [], 0
        while True:
            results = self._client.cql(
                cql,
                start=start,
                limit=self._PAGE_LIMIT,
                expand="body.storage,version,history.lastUpdated",
            )
            pages = results.get("results", [])
            for page in pages:
                doc = self._page_to_doc(page, space_key)
                if doc:
                    docs.append(doc)
            if len(pages) < self._PAGE_LIMIT:
                break
            start += self._PAGE_LIMIT

        return docs

    def _page_to_doc(self, page: dict, space_key: str) -> ConnectorDoc | None:
        try:
            page_id = page["content"]["id"]
            title = page["content"]["title"]
            html = page["content"].get("body", {}).get("storage", {}).get("value", "")

            last_updated = page.get("lastModified") or page.get(
                "content", {}
            ).get("history", {}).get("lastUpdated", {}).get("when")

            last_modified = None
            if last_updated:
                try:
                    last_modified = datetime.fromisoformat(
                        last_updated.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            author = (
                page.get("content", {})
                .get("history", {})
                .get("lastUpdated", {})
                .get("by", {})
                .get("displayName", "")
            )

            return ConnectorDoc(
                source_url=f"confluence::{space_key}::{page_id}",
                title=title,
                source_name=self.source_name,
                content=html,
                content_type="html",
                author=author,
                last_modified=last_modified,
                metadata={
                    "space_key": space_key,
                    "page_id": page_id,
                    "confluence_title": title,
                },
            )
        except Exception as e:
            logger.warning("Skipping Confluence page (parse error): %s", e)
            return None
