"""
connectors/wiki.py

What problem does this solve?
- MediaWiki instances (Wikipedia-style internal wikis) are common in large
  enterprises and open-source organisations as knowledge bases, technical
  documentation stores, and process registries.
- Without a connector, wiki pages must be manually exported — a process
  that doesn't scale to wikis with thousands of articles.

Why mwclient over direct MediaWiki REST API?
- mwclient handles authentication (bot credentials, OAuth), session management,
  and pagination automatically. The raw REST API requires manual handling of
  continuation tokens, which is error-prone.
- mwclient's Page object gives direct access to wikitext + parsed HTML,
  last revision timestamp (for incremental sync), and page metadata.

Why use parsed HTML (not wikitext)?
- Wikitext contains templates ({{infobox}}, {{cite web}}) that produce
  garbage if treated as plain text. The MediaWiki API renders wikitext to
  HTML which our existing HtmlLoader already handles cleanly.

Authentication:
- Public wikis:  no credentials needed (anonymous read access)
- Private wikis: bot_username + bot_password (create at Special:BotPasswords)
"""

import logging
from datetime import datetime, timezone

from agent.connectors.base import BaseConnector
from agent.connectors.models import ConnectorDoc

logger = logging.getLogger(__name__)

try:
    import mwclient
    _MWCLIENT_AVAILABLE = True
except ImportError:
    _MWCLIENT_AVAILABLE = False


class MediaWikiConnector(BaseConnector):
    """
    What problem does this solve?
    - Fetches MediaWiki pages as HTML ConnectorDocs for SyncService.

    Why filter by namespace?
    - MediaWiki namespaces separate article content (namespace 0) from talk
      pages (namespace 1), user pages (namespace 2), templates (namespace 10),
      etc. For RAG indexing, only article content (namespace 0) and
      optionally project pages (namespace 4) are useful. Talk and user
      pages contain discussion noise not suitable for enterprise search.

    Why cap at max_pages?
    - Public wikis like Wikipedia have millions of pages. Without a cap,
      a misconfigured sync could attempt to ingest the entire wiki.
      max_pages provides a safety rail. For internal wikis (hundreds to
      thousands of pages), set it higher.
    """

    def __init__(
        self,
        host: str,
        path: str = "/w/",
        scheme: str = "https",
        bot_username: str = "",
        bot_password: str = "",
    ) -> None:
        """
        Args:
        - host:         Wiki hostname, e.g. "en.wikipedia.org" or
                        "wiki.mycompany.com"
        - path:         MediaWiki API path (default "/w/" for standard installs).
        - scheme:       "https" (default) or "http".
        - bot_username: Bot account username for private wikis.
        - bot_password: Bot account password for private wikis.
        """
        if not _MWCLIENT_AVAILABLE:
            raise ImportError(
                "mwclient is required for MediaWikiConnector. "
                "Install it with: pip install mwclient"
            )
        self._site = mwclient.Site(host, path=path, scheme=scheme)
        if bot_username and bot_password:
            self._site.login(bot_username, bot_password)

    @property
    def source_name(self) -> str:
        return "wiki"

    def fetch_documents(
        self,
        since: datetime | None = None,
        namespace: int = 0,
        category: str | None = None,
        max_pages: int = 5000,
        **kwargs,
    ) -> list[ConnectorDoc]:
        """
        What problem does this solve?
        - Fetches MediaWiki pages and returns them as ConnectorDoc objects.

        Args:
        - since:     Only return pages with revisions after this datetime.
                     Uses the MediaWiki recentchanges API for incremental sync.
                     None = full sync of all pages in the namespace.
        - namespace: MediaWiki namespace to index (0 = articles, default).
        - category:  Limit to pages in a specific category. None = all pages.
        - max_pages: Safety cap on total pages fetched per sync.

        Why two fetch paths (since vs full scan)?
        - The recentchanges API is efficient for incremental sync — it returns
          only changed pages. For full sync, we must enumerate allpages.
          Trying to apply a date filter to allpages requires fetching every
          page's revision history — much slower than recentchanges.
        """
        try:
            if since:
                return self._fetch_recent(since, namespace, max_pages)
            if category:
                return self._fetch_category(category, namespace, max_pages)
            return self._fetch_all_pages(namespace, max_pages)
        except Exception as e:
            logger.error("MediaWiki fetch failed: %s", e)
            return []

    # ── Private ────────────────────────────────────────────────────────────────

    def _fetch_recent(
        self, since: datetime, namespace: int, max_pages: int
    ) -> list[ConnectorDoc]:
        """Use recentchanges API — efficient for incremental sync."""
        docs = []
        seen: set[str] = set()
        since_ts = since.strftime("%Y-%m-%dT%H:%M:%SZ")

        changes = self._site.recentchanges(
            start=since_ts,
            namespace=namespace,
            type="edit|new",
            dir="newer",
            limit=500,
        )
        for change in changes:
            title = change.get("title", "")
            if title in seen or len(docs) >= max_pages:
                continue
            seen.add(title)
            doc = self._fetch_page(title)
            if doc:
                docs.append(doc)
        return docs

    def _fetch_category(
        self, category: str, namespace: int, max_pages: int
    ) -> list[ConnectorDoc]:
        docs = []
        cat = self._site.categories[category]
        for page in cat:
            if len(docs) >= max_pages:
                break
            if page.namespace != namespace:
                continue
            doc = self._fetch_page(page.name)
            if doc:
                docs.append(doc)
        return docs

    def _fetch_all_pages(self, namespace: int, max_pages: int) -> list[ConnectorDoc]:
        docs = []
        for page in self._site.allpages(namespace=namespace):
            if len(docs) >= max_pages:
                break
            doc = self._fetch_page(page.name)
            if doc:
                docs.append(doc)
        return docs

    def _fetch_page(self, title: str) -> ConnectorDoc | None:
        try:
            page = self._site.pages[title]
            if not page.exists:
                return None

            # Get rendered HTML from MediaWiki API
            html = self._site.api(
                "parse",
                page=title,
                prop="text",
                disablelimitreport=True,
            ).get("parse", {}).get("text", {}).get("*", "")

            # Last revision timestamp for dedup
            last_modified = None
            rev = page.revision
            if rev and hasattr(rev, "timestamp"):
                try:
                    ts = rev.timestamp
                    if isinstance(ts, (list, tuple)):
                        ts = ts[0] if ts else None
                    if ts:
                        last_modified = datetime.fromisoformat(
                            str(ts).replace("Z", "+00:00")
                        )
                except Exception:
                    pass

            return ConnectorDoc(
                source_url=f"wiki::{self._site.host}::{title}",
                title=title,
                source_name=self.source_name,
                content=html,
                content_type="html",
                last_modified=last_modified,
                metadata={
                    "wiki_host": self._site.host,
                    "page_title": title,
                    "namespace": page.namespace,
                },
            )
        except Exception as e:
            logger.warning("Skipping wiki page '%s': %s", title, e)
            return None
