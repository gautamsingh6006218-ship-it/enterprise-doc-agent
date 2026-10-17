"""
tests/connectors/test_factory.py

What does this cover?
- build_connector(): unknown source → UnknownSourceError,
                     known source without env vars → ConnectorNotConfiguredError,
                     both errors are ConnectorConfigError (what the scheduler catches),
                     configured source → connector with env values.
"""

from unittest.mock import patch

import pytest

from agent.connectors.factory import (
    KNOWN_SOURCES,
    ConnectorConfigError,
    ConnectorNotConfiguredError,
    UnknownSourceError,
    build_connector,
)

_ENV_VARS = [
    "CONFLUENCE_URL", "CONFLUENCE_TOKEN", "CONFLUENCE_USERNAME",
    "SP_CLIENT_ID", "SP_CLIENT_SECRET", "SP_TENANT_ID", "SP_SITE_URL",
    "JIRA_URL", "JIRA_TOKEN", "JIRA_USERNAME",
    "WIKI_HOST", "WIKI_BOT_USER", "WIKI_BOT_PASSWORD",
]


@pytest.fixture(autouse=True)
def _clear_connector_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestBuildConnector:

    def test_unknown_source_raises(self):
        with pytest.raises(UnknownSourceError, match="Unknown source 'notion'"):
            build_connector("notion")

    @pytest.mark.parametrize("source", sorted(KNOWN_SOURCES))
    def test_unconfigured_source_raises(self, source):
        with pytest.raises(ConnectorNotConfiguredError, match="not configured"):
            build_connector(source)

    def test_errors_share_base_class(self):
        assert issubclass(UnknownSourceError, ConnectorConfigError)
        assert issubclass(ConnectorNotConfiguredError, ConnectorConfigError)

    def test_configured_jira_uses_env(self, monkeypatch):
        monkeypatch.setenv("JIRA_URL", "https://jira.example.com")
        monkeypatch.setenv("JIRA_TOKEN", "tok")
        monkeypatch.setenv("JIRA_USERNAME", "bot")
        with patch("agent.connectors.jira.JiraConnector") as jira_cls:
            connector = build_connector("jira")
        jira_cls.assert_called_once_with(
            url="https://jira.example.com", token="tok", username="bot"
        )
        assert connector is jira_cls.return_value
