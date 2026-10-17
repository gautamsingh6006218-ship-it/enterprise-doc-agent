"""Connectors module: fetch documents from Confluence, SharePoint, Jira and MediaWiki."""

from agent.connectors.base import BaseConnector
from agent.connectors.factory import (
    KNOWN_SOURCES,
    ConnectorConfigError,
    ConnectorNotConfiguredError,
    UnknownSourceError,
    build_connector,
)
from agent.connectors.models import ConnectorDoc, SyncResult

__all__ = [
    "BaseConnector",
    "KNOWN_SOURCES",
    "ConnectorConfigError",
    "ConnectorNotConfiguredError",
    "UnknownSourceError",
    "build_connector",
    "ConnectorDoc",
    "SyncResult",
]
