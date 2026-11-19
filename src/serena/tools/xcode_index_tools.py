"""Optional Serena tools backed by the local Xcode IndexStore graph."""

from __future__ import annotations

import json

from serena.tools.tools_base import Tool, ToolMarkerOptional, ToolMarkerSymbolicRead
from serena.xcode_index import XcodeIndexService


class XcodeIndexStatusTool(Tool, ToolMarkerOptional, ToolMarkerSymbolicRead):
    """Report the local Xcode IndexStore graph status without starting MCP-specific services."""

    def apply(self) -> str:
        """Return graph counts, source store metadata and snapshot freshness."""
        with XcodeIndexService.from_project(self.project).graph() as graph:
            return json.dumps(graph.status().__dict__, ensure_ascii=False, indent=2, sort_keys=True)


class XcodeIndexRefreshTool(Tool, ToolMarkerOptional, ToolMarkerSymbolicRead):
    """Refresh Serena's persistent graph from a compiler-produced Xcode IndexStore."""

    def apply(self, store_path: str | None = None, prefix_maps: list[str] | None = None) -> str:
        """Read the selected DataStore and return the refreshed graph counts.

        :param store_path: optional explicit ``Index.noindex/DataStore`` path;
            otherwise use project settings or DerivedData discovery.
        :param prefix_maps: optional additional ``remote=local`` path mappings.
        """
        service = XcodeIndexService.from_project(self.project)
        store = service.resolve_store_path(store_path)
        with service.graph() as graph:
            result = graph.refresh_from_store(
                store,
                service.project_root,
                prefix_mappings=service.prefix_mappings(tuple(prefix_maps or ())),
                library_path=service.library_path(),
            )
            status = graph.status()
        return json.dumps(
            {
                "store_path": result.store_path,
                "units": result.unit_count,
                "records": result.record_count,
                "occurrences": len(result.occurrences),
                "calls": result.call_count,
                "edges": status.edges,
                "database_path": status.database_path,
                "fingerprint": result.fingerprint,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


class XcodeIndexCallersTool(Tool, ToolMarkerOptional, ToolMarkerSymbolicRead):
    """Return compiler-evidenced callers for an exact symbol name or USR."""

    def apply(self, query: str, file_path: str | None = None, limit: int = 200) -> str:
        """Return direct callers and their source call-site evidence.

        :param query: exact symbol name, ``usr:<USR>`` or a raw compiler USR.
        :param file_path: optional source file used to disambiguate names.
        :param limit: maximum number of evidence rows.
        """
        with XcodeIndexService.from_project(self.project).graph() as graph:
            return json.dumps(graph.callers(query, file_path=file_path, limit=limit), ensure_ascii=False, indent=2, sort_keys=True)


class XcodeIndexCalleesTool(Tool, ToolMarkerOptional, ToolMarkerSymbolicRead):
    """Return compiler-evidenced callees for an exact symbol name or USR."""

    def apply(self, query: str, file_path: str | None = None, limit: int = 200) -> str:
        """Return direct callees and their source call-site evidence."""
        with XcodeIndexService.from_project(self.project).graph() as graph:
            return json.dumps(graph.callees(query, file_path=file_path, limit=limit), ensure_ascii=False, indent=2, sort_keys=True)


class XcodeIndexImpactTool(Tool, ToolMarkerOptional, ToolMarkerSymbolicRead):
    """Walk compiler call edges to estimate bounded project impact."""

    def apply(
        self,
        query: str,
        direction: str = "callers",
        depth: int = 3,
        file_path: str | None = None,
        limit: int = 1000,
    ) -> str:
        """Return a bounded callers/callees traversal.

        :param query: exact symbol name or compiler USR.
        :param direction: ``callers`` for impact or ``callees`` for dependencies.
        :param depth: maximum traversal depth.
        :param file_path: optional source file used to disambiguate names.
        :param limit: maximum number of returned nodes.
        """
        with XcodeIndexService.from_project(self.project).graph() as graph:
            return json.dumps(
                graph.impact(query, direction=direction, depth=depth, file_path=file_path, limit=limit),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )


class XcodeIndexTraceTool(Tool, ToolMarkerOptional, ToolMarkerSymbolicRead):
    """Find bounded direct-call paths between compiler symbols."""

    def apply(
        self,
        start: str,
        end: str,
        max_depth: int = 8,
        start_file: str | None = None,
        end_file: str | None = None,
        max_paths: int = 100,
    ) -> str:
        """Return call paths from ``start`` to ``end`` with evidence."""
        with XcodeIndexService.from_project(self.project).graph() as graph:
            return json.dumps(
                graph.trace(
                    start,
                    end,
                    max_depth=max_depth,
                    start_file=start_file,
                    end_file=end_file,
                    max_paths=max_paths,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )


class LiveCallHierarchyTool(Tool, ToolMarkerOptional, ToolMarkerSymbolicRead):
    """Query SourceKit-LSP or clangd's live call hierarchy at a source position."""

    def apply(self, relative_file_path: str, line: int, column: int, direction: str = "incoming") -> str:
        """Return live LSP call hierarchy entries for the symbol at a position.

        :param relative_file_path: project-relative source file path.
        :param line: zero-based LSP line number.
        :param column: zero-based LSP character offset.
        :param direction: ``incoming`` or ``outgoing``.
        """
        manager = self.project.language_server_manager or self.project.create_language_server_manager()
        language_server = manager.get_language_server(relative_file_path)
        result = language_server.request_call_hierarchy(relative_file_path, line, column, direction=direction)
        return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)


__all__ = [
    "LiveCallHierarchyTool",
    "XcodeIndexCalleesTool",
    "XcodeIndexCallersTool",
    "XcodeIndexImpactTool",
    "XcodeIndexRefreshTool",
    "XcodeIndexStatusTool",
    "XcodeIndexTraceTool",
]
