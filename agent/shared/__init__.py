"""Shared kernel: value objects that every module may depend on."""

from agent.shared.models import RBACContext, SearchResult

__all__ = ["RBACContext", "SearchResult"]
