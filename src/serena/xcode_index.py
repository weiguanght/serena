"""Serena integration for compiler-produced Xcode indexes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from serena.config.serena_config import ProjectConfig, ProjectConfigAutoGenerationMode, SerenaConfig
from solidlsp.indexstore import IndexStoreError, discover_index_store_path
from solidlsp.indexstore_graph import IndexStoreGraph
from solidlsp.ls_config import LanguageServerId

if TYPE_CHECKING:
    from serena.project import Project


class XcodeIndexService:
    """Resolve project configuration and expose the local IndexStore graph."""

    # Keep the generated graph under Serena's ignored cache directory.  A
    # project may still choose any explicit path through ``index_graph_path``.
    DEFAULT_DATABASE_NAME = os.path.join("cache", "xcode-index.sqlite3")

    def __init__(self, project_root: str, data_path: str, swift_settings: dict[str, Any] | None = None):
        self.project_root = str(Path(project_root).expanduser().resolve())
        self.data_path = str(Path(data_path).expanduser().resolve())
        self.swift_settings = swift_settings or {}

    @classmethod
    def from_project(cls, project: "Project") -> "XcodeIndexService":
        """Create a service from an already-loaded Serena project."""
        settings = dict(project.serena_config.ls_specific_settings)
        if project.project_config.ls_specific_settings and project.is_trusted():
            settings.update(project.project_config.ls_specific_settings)
        return cls(
            project.project_root,
            project.path_to_serena_data_folder(),
            cls._language_settings(settings, LanguageServerId.SWIFT),
        )

    @classmethod
    def from_path(cls, project_root: str | os.PathLike[str], serena_config: SerenaConfig) -> "XcodeIndexService":
        """Create a service without starting a language server or requiring MCP."""
        root = str(Path(project_root).expanduser().resolve())
        settings = dict(serena_config.ls_specific_settings)
        project_yml = serena_config.get_project_yml_location(root)
        if os.path.isfile(project_yml) and serena_config.is_trusted_project_path(root):
            project_config = ProjectConfig.load(
                root,
                serena_config=serena_config,
                autogen=ProjectConfigAutoGenerationMode.NONE,
            )
            settings.update(project_config.ls_specific_settings)
        return cls(
            root,
            serena_config.get_project_serena_folder(root),
            cls._language_settings(settings, LanguageServerId.SWIFT),
        )

    @staticmethod
    def _language_settings(settings: dict[Any, Any], language: LanguageServerId) -> dict[str, Any]:
        value = settings.get(language, settings.get(language.value, {}))
        return dict(value) if isinstance(value, dict) else {}

    @property
    def database_path(self) -> str:
        """Return the configured persistent graph path."""
        configured = self.swift_settings.get("index_graph_path")
        if not isinstance(configured, str) or not configured:
            return str(Path(self.data_path) / self.DEFAULT_DATABASE_NAME)
        path = Path(os.path.expandvars(os.path.expanduser(configured)))
        if not path.is_absolute():
            path = Path(self.data_path) / path
        return str(path.resolve())

    def graph(self, database_path: str | None = None) -> IndexStoreGraph:
        """Open the persistent graph, optionally at an explicit path."""
        resolved_path = database_path or self.database_path
        cache_path = Path(self.data_path) / "cache"
        if database_path is None and Path(resolved_path).is_relative_to(cache_path):
            self._ensure_default_database_is_ignored()
        return IndexStoreGraph(resolved_path)

    def _ensure_default_database_is_ignored(self) -> None:
        """Ensure Serena's generated cache directory is ignored by Git."""
        data_path = Path(self.data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        gitignore_path = data_path / ".gitignore"
        rule = "/cache"
        if gitignore_path.is_file():
            content = gitignore_path.read_text(encoding="utf-8")
            existing_rules = content.splitlines()
            if rule in existing_rules:
                return
            separator = "" if not content or content.endswith(("\n", "\r")) else "\n"
            with gitignore_path.open("a", encoding="utf-8") as stream:
                stream.write(f"{separator}{rule}\n")
            return
        gitignore_path.write_text(f"{rule}\n", encoding="utf-8")

    def resolve_store_path(self, explicit_path: str | None = None) -> str:
        """Resolve an explicit, configured or discovered Xcode DataStore."""
        configured = explicit_path or self.swift_settings.get("index_store_path")
        if isinstance(configured, str) and configured:
            path = Path(os.path.expandvars(os.path.expanduser(configured)))
            if not path.is_absolute():
                path = Path(self.project_root) / path
            if not path.exists():
                raise IndexStoreError(f"IndexStore path does not exist: {path}")
            return str(path.resolve())
        discovered = discover_index_store_path(self.project_root)
        if discovered is None:
            raise IndexStoreError(
                "No matching Xcode IndexStore was found. Build the desired scheme in Xcode, "
                "then pass --store or configure swift.index_store_path."
            )
        return discovered

    def prefix_mappings(self, extra_mappings: tuple[str, ...] = ()) -> dict[str, str] | list[str] | tuple[str, ...] | None:
        """Return configured prefix mappings plus command-line additions."""
        configured = self.swift_settings.get("index_prefix_map")
        if not extra_mappings:
            return configured
        if configured is None:
            return extra_mappings
        if isinstance(configured, dict):
            result = [f"{source}={destination}" for source, destination in configured.items()]
        elif isinstance(configured, list | tuple):
            result = list(configured)
        else:
            raise ValueError("swift.index_prefix_map must be a mapping or list of 'remote=local' strings")
        result.extend(extra_mappings)
        return result

    def library_path(self) -> str | None:
        """Return an optional explicit libIndexStore path."""
        value = self.swift_settings.get("indexstore_library_path")
        if not isinstance(value, str) or not value:
            return None
        path = Path(os.path.expandvars(os.path.expanduser(value)))
        if not path.is_absolute():
            path = Path(self.project_root) / path
        return str(path.resolve())


__all__ = ["XcodeIndexService"]
