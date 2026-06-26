"""
connectors/jira.py

What problem does this solve?
- Jira issues are a primary knowledge source for engineering teams: bug reports,
  feature specs, incident post-mortems, architecture decisions. Without a Jira
  connector, this institutional knowledge is invisible to the RAG system.

Why index Jira issues instead of just attachments?
- Most Jira knowledge lives in description fields and comment threads, not
  attached files. A post-mortem is often written directly in the issue
  description. Comments contain decision rationale that doesn't exist elsewhere.

Why format as Markdown?
- Jira descriptions use Atlassian Document Format (ADF) or wiki markup.
  atlassian-python-api returns the rendered HTML representation which our
  existing HtmlLoader can process. For plain-text fields (summary, comments),
  we build a simple Markdown structure so the content is readable as-is.

Authentication:
- Cloud Jira: url + email + api_token
- Server/Data Center: url + username + password (or PAT via token field)

Why include project_key and issue_type in metadata?
- Enables RBAC filtering: "only show issues from the SECURITY project to the
  security team". Stored in chunk metadata so retrieval RBAC can filter by
  project just like it filters by tenant_id.
"""

import logging
from datetime import datetime, timezone

from agent.connectors.base import BaseConnector
from agent.connectors.models import ConnectorDoc

logger = logging.getLogger(__name__)

try:
    from atlassian import Jira as AtlassianJira
    _ATLASSIAN_AVAILABLE = True
except ImportError:
    _ATLASSIAN_AVAILABLE = False

_DEFAULT_FIELDS = "summary,description,comment,assignee,reporter,issuetype,status,updated,created"


class JiraConnector(BaseConnector):
    """
    What problem does this solve?
    - Fetches Jira issues (descriptions + comment threads) as Markdown
      ConnectorDocs so SyncService can push them through the pipeline.

    Why JQL for filtering?
    - JQL (Jira Query Language) supports all filter combinations natively:
      project filter, issue type filter, updated-since filter, and
      status filter. The REST pagination API doesn't support date filtering —
      JQL is the only efficient way to do incremental sync.

    Why exclude closed/resolved issues by default?
    - Closed issues from years ago add noise without value.
      Active and recently-resolved issues are the relevant knowledge.
      Callers can override by passing include_closed=True.
    """

    _PAGE_SIZE = 50

    def __init__(
        self,
        url: str,
        token: str,
        username: str = "",
        cloud: bool = True,
    ) -> None:
        """
        Args:
        - url:      Jira base URL, e.g. "https://myorg.atlassian.net"
        - token:    API token (cloud) or password (server).
        - username: Email address (cloud) or username (server).
        - cloud:    True for Atlassian Cloud, False for Server/Data Center.
        """
        if not _ATLASSIAN_AVAILABLE:
            raise ImportError(
                "atlassian-python-api is required for JiraConnector. "
                "Install it with: pip install atlassian-python-api"
            )
        self._client = AtlassianJira(
            url=url,
            username=username,
            password=token,
            cloud=cloud,
        )

    @property
    def source_name(self) -> str:
        return "jira"

    def fetch_documents(
        self,
        since: datetime | None = None,
        project_key: str | None = None,
        issue_types: list[str] | None = None,
        include_closed: bool = False,
        **kwargs,
    ) -> list[ConnectorDoc]:
        """
        What problem does this solve?
        - Fetches Jira issues matching the given filters and returns them
          as ConnectorDoc objects ready for ingestion.

        Args:
        - since:          JQL updated >= filter for incremental sync.
        - project_key:    Limit to a single project (e.g. "ENG", "INFRA").
                          None = all projects.
        - issue_types:    Limit to specific types: ["Bug", "Story", "Epic"].
                          None = all types.
        - include_closed: Include issues with status Closed/Done.
                          False by default to reduce noise.
        """
        try:
            jql = self._build_jql(since, project_key, issue_types, include_closed)
            return self._fetch_all(jql)
        except Exception as e:
            logger.error("Jira fetch failed: %s", e)
            return []

    # ── Private ────────────────────────────────────────────────────────────────

    def _build_jql(
        self,
        since: datetime | None,
        project_key: str | None,
        issue_types: list[str] | None,
        include_closed: bool,
    ) -> str:
        clauses = []
        if project_key:
            clauses.append(f'project = "{project_key}"')
        if issue_types:
            types_str = ", ".join(f'"{t}"' for t in issue_types)
            clauses.append(f"issuetype in ({types_str})")
        if not include_closed:
            clauses.append('status not in ("Closed", "Done", "Resolved")')
        if since:
            ts = since.strftime("%Y-%m-%d %H:%M")
            clauses.append(f'updated >= "{ts}"')
        return " AND ".join(clauses) if clauses else "ORDER BY updated DESC"

    def _fetch_all(self, jql: str) -> list[ConnectorDoc]:
        docs, start = [], 0
        while True:
            response = self._client.jql(
                jql,
                start=start,
                limit=self._PAGE_SIZE,
                fields=_DEFAULT_FIELDS,
            )
            issues = response.get("issues", [])
            for issue in issues:
                doc = self._issue_to_doc(issue)
                if doc:
                    docs.append(doc)
            if len(issues) < self._PAGE_SIZE:
                break
            start += self._PAGE_SIZE
        return docs

    def _issue_to_doc(self, issue: dict) -> ConnectorDoc | None:
        try:
            fields = issue.get("fields", {})
            key = issue["key"]
            summary = fields.get("summary", key)
            description = fields.get("description") or ""
            issue_type = fields.get("issuetype", {}).get("name", "Issue")
            status = fields.get("status", {}).get("name", "")
            reporter = fields.get("reporter", {}).get("displayName", "")
            updated = fields.get("updated", "")

            comments = self._extract_comments(fields.get("comment", {}))

            # Build structured Markdown so chunk text is human-readable
            md_parts = [
                f"# {summary}",
                f"**Type:** {issue_type}  **Status:** {status}  **Key:** {key}",
                "",
            ]
            if description:
                md_parts += ["## Description", description, ""]
            if comments:
                md_parts += ["## Comments", comments]

            content = "\n".join(md_parts)

            last_modified = None
            if updated:
                try:
                    last_modified = datetime.fromisoformat(
                        updated.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            return ConnectorDoc(
                source_url=f"jira::{key}",
                title=f"{key}: {summary}",
                source_name=self.source_name,
                content=content,
                content_type="text",
                author=reporter,
                last_modified=last_modified,
                metadata={
                    "issue_key": key,
                    "issue_type": issue_type,
                    "status": status,
                    "project_key": key.split("-")[0],
                },
            )
        except Exception as e:
            logger.warning("Skipping Jira issue (parse error): %s", e)
            return None

    @staticmethod
    def _extract_comments(comment_field: dict) -> str:
        comments = comment_field.get("comments", [])
        parts = []
        for c in comments:
            author = c.get("author", {}).get("displayName", "Unknown")
            body = c.get("body") or ""
            if isinstance(body, dict):
                # ADF format — extract plain text from paragraphs
                body = _adf_to_text(body)
            parts.append(f"**{author}:** {body}")
        return "\n\n".join(parts)


def _adf_to_text(adf: dict) -> str:
    """
    Minimal Atlassian Document Format → plain text extractor.
    Only extracts paragraph text nodes — sufficient for comment indexing.
    Full ADF rendering is outside scope; atlassian-python-api handles
    the common cases we need.
    """
    parts = []
    for node in adf.get("content", []):
        if node.get("type") == "paragraph":
            for child in node.get("content", []):
                if child.get("type") == "text":
                    parts.append(child.get("text", ""))
    return " ".join(parts)
