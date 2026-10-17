"""
connectors/factory.py

What problem does this solve?
- POST /sync/{source} and the background scheduler both build connectors from
  environment variables. The builder lived in the sync route file, so the
  scheduler imported a private function from a route file.

Why raise domain errors instead of HTTPException?
- The connectors module must not know about HTTP. The route maps
  UnknownSourceError → 404 and ConnectorNotConfiguredError → 503; the
  scheduler logs both and skips the source.

Environment variables for each source:
  Confluence:  CONFLUENCE_URL, CONFLUENCE_TOKEN, CONFLUENCE_USERNAME
  SharePoint:  SP_CLIENT_ID, SP_CLIENT_SECRET, SP_TENANT_ID, SP_SITE_URL
  Jira:        JIRA_URL, JIRA_TOKEN, JIRA_USERNAME
  Wiki:        WIKI_HOST, WIKI_BOT_USER, WIKI_BOT_PASSWORD
"""

import os

from agent.connectors.base import BaseConnector

KNOWN_SOURCES = {"confluence", "sharepoint", "jira", "wiki"}


class ConnectorConfigError(Exception):
    """A connector cannot be built for the requested source."""


class UnknownSourceError(ConnectorConfigError):
    """The source name is not one of KNOWN_SOURCES."""


class ConnectorNotConfiguredError(ConnectorConfigError):
    """The source is known but its environment variables are not set."""


def build_connector(source: str) -> BaseConnector:
    """
    Build the configured connector for the given source.

    Raises UnknownSourceError for an unknown source and
    ConnectorNotConfiguredError if required environment variables are not set.
    """
    if source not in KNOWN_SOURCES:
        raise UnknownSourceError(
            f"Unknown source '{source}'. Known: {sorted(KNOWN_SOURCES)}"
        )

    if source == "confluence":
        url = os.getenv("CONFLUENCE_URL")
        token = os.getenv("CONFLUENCE_TOKEN")
        if not url or not token:
            raise ConnectorNotConfiguredError(
                "Confluence not configured. Set CONFLUENCE_URL and CONFLUENCE_TOKEN."
            )
        from agent.connectors.confluence import ConfluenceConnector
        return ConfluenceConnector(
            url=url,
            token=token,
            username=os.getenv("CONFLUENCE_USERNAME", ""),
        )

    if source == "sharepoint":
        client_id = os.getenv("SP_CLIENT_ID")
        client_secret = os.getenv("SP_CLIENT_SECRET")
        sp_tenant_id = os.getenv("SP_TENANT_ID")
        site_url = os.getenv("SP_SITE_URL")
        if not all([client_id, client_secret, sp_tenant_id, site_url]):
            raise ConnectorNotConfiguredError(
                "SharePoint not configured. Set SP_CLIENT_ID, SP_CLIENT_SECRET, SP_TENANT_ID, SP_SITE_URL."
            )
        from agent.connectors.sharepoint import SharePointConnector
        return SharePointConnector(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=sp_tenant_id,
            site_url=site_url,
        )

    if source == "jira":
        url = os.getenv("JIRA_URL")
        token = os.getenv("JIRA_TOKEN")
        if not url or not token:
            raise ConnectorNotConfiguredError(
                "Jira not configured. Set JIRA_URL and JIRA_TOKEN."
            )
        from agent.connectors.jira import JiraConnector
        return JiraConnector(
            url=url,
            token=token,
            username=os.getenv("JIRA_USERNAME", ""),
        )

    host = os.getenv("WIKI_HOST")
    if not host:
        raise ConnectorNotConfiguredError("Wiki not configured. Set WIKI_HOST.")
    from agent.connectors.wiki import MediaWikiConnector
    return MediaWikiConnector(
        host=host,
        bot_username=os.getenv("WIKI_BOT_USER", ""),
        bot_password=os.getenv("WIKI_BOT_PASSWORD", ""),
    )
