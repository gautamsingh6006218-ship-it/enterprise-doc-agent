"""
connectors/base.py

What problem does this solve?
- SyncService needs to call fetch_documents() on any connector without knowing
  which enterprise source it talks to. Without a shared interface, SyncService
  would need a conditional branch for every source type — brittle and hard to
  extend when a new source (Notion, Zendesk, Slack) is added.

Why ABC instead of a Protocol?
- ABC gives a clear error at class definition time if a subclass forgets to
  implement fetch_documents() — faster feedback than a Protocol which only
  errors at call time.
- source_name as an abstract property ensures every connector declares its
  name — used in ConnectorDoc.source_name and SyncResult.source.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from agent.connectors.models import ConnectorDoc


class BaseConnector(ABC):
    """
    What problem does this solve?
    - Common interface for all enterprise source connectors.
      SyncService only calls fetch_documents() — it never imports
      ConfluenceConnector, SharePointConnector, etc. directly.

    Why fetch_documents() instead of a generator?
    - A list is simpler to mock in tests and to pass to SyncService.
      For sources with thousands of documents, pagination is handled
      inside the connector (it yields pages internally) and returns a
      complete list. If memory becomes a concern at scale, this can be
      changed to a generator without changing SyncService's interface.

    Why since as a fetch_documents parameter?
    - Incremental sync: only fetch documents modified after a given
      datetime. Without this, every sync re-processes the entire source.
      Passing it to fetch_documents() lets each connector use the native
      API filter (Confluence lastModified, Jira JQL updated>=, SharePoint
      lastModifiedDateTime) — more efficient than fetching all and filtering.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        Short identifier for this source.
        Used in ConnectorDoc.source_name and SyncResult.source.
        Examples: "confluence", "sharepoint", "jira", "wiki"
        """

    @abstractmethod
    def fetch_documents(
        self,
        since: datetime | None = None,
        **kwargs,
    ) -> list[ConnectorDoc]:
        """
        What problem does this solve?
        - Fetches all (or incrementally updated) documents from the source
          and returns them as ConnectorDoc objects ready for SyncService.

        Args:
        - since:   Only return documents modified after this datetime.
                   None = full sync (fetch everything).
        - **kwargs: Source-specific filters (space_key, project_key,
                   library_name, namespace, etc.). Defined per subclass.

        Returns a list[ConnectorDoc]. Never raises — returns [] on error
        and logs internally (consistent with the Result pattern used
        throughout the pipeline services).
        """
