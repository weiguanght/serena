"""Persistent queries over an Xcode IndexStore snapshot.

The graph is deliberately evidence-oriented.  A symbol is identified by its
compiler USR, while occurrences and call edges retain their source location,
compilation unit and target.  This makes the snapshot useful for whole-project
impact analysis without claiming that it replaces live SourceKit-LSP/clangd
queries for unsaved edits.
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from solidlsp.indexstore import (
    ROLE_REL_CALLEDBY,
    IndexStoreError,
    IndexStoreReader,
    IndexStoreReadResult,
    index_store_fingerprint,
)

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class GraphSymbol:
    """A compiler symbol returned by a graph query."""

    usr: str
    name: str
    language: int
    kind: int
    definition_file: str | None = None
    definition_line: int | None = None
    definition_column: int | None = None


@dataclass(frozen=True)
class GraphEvidence:
    """One call-edge occurrence retained as query evidence."""

    caller_usr: str
    callee_usr: str
    file_path: str
    line: int
    column: int
    unit_name: str
    target: str
    roles: int
    dynamic: bool
    relation_roles: int


@dataclass(frozen=True)
class GraphStatus:
    """Summary of the current persisted snapshot."""

    database_path: str
    store_path: str | None
    root_path: str | None
    fingerprint: str | None
    is_current: bool | None
    symbols: int
    occurrences: int
    edges: int
    units: int
    refreshed_at: str | None


class IndexStoreGraph:
    """Store and query an IndexStore snapshot in SQLite."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(Path(database_path).expanduser().resolve())
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def __enter__(self) -> "IndexStoreGraph":
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the SQLite connection."""
        self._connection.close()

    def _create_schema(self) -> None:
        """Create or upgrade the small local schema."""
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS graph_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS graph_units (
                unit_name TEXT NOT NULL,
                main_file TEXT NOT NULL,
                target TEXT NOT NULL,
                module_name TEXT NOT NULL,
                output_file TEXT NOT NULL,
                working_directory TEXT NOT NULL,
                sysroot_path TEXT NOT NULL,
                provider_identifier TEXT NOT NULL,
                provider_version TEXT NOT NULL,
                modification_time REAL NOT NULL,
                is_debug_compilation INTEGER NOT NULL,
                is_module_unit INTEGER NOT NULL,
                is_system_unit INTEGER NOT NULL,
                PRIMARY KEY (unit_name, target)
            );
            CREATE TABLE IF NOT EXISTS graph_symbols (
                usr TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                language INTEGER NOT NULL,
                kind INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS graph_occurrences (
                occurrence_id INTEGER PRIMARY KEY,
                usr TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line INTEGER NOT NULL,
                column_number INTEGER NOT NULL,
                roles INTEGER NOT NULL,
                unit_name TEXT NOT NULL,
                target TEXT NOT NULL,
                dynamic INTEGER NOT NULL,
                FOREIGN KEY (usr) REFERENCES graph_symbols (usr)
            );
            CREATE TABLE IF NOT EXISTS graph_definitions (
                occurrence_id INTEGER PRIMARY KEY,
                usr TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line INTEGER NOT NULL,
                column_number INTEGER NOT NULL,
                unit_name TEXT NOT NULL,
                target TEXT NOT NULL,
                FOREIGN KEY (occurrence_id) REFERENCES graph_occurrences (occurrence_id),
                FOREIGN KEY (usr) REFERENCES graph_symbols (usr)
            );
            CREATE TABLE IF NOT EXISTS graph_edges (
                edge_id INTEGER PRIMARY KEY,
                caller_usr TEXT NOT NULL,
                callee_usr TEXT NOT NULL,
                occurrence_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                line INTEGER NOT NULL,
                column_number INTEGER NOT NULL,
                unit_name TEXT NOT NULL,
                target TEXT NOT NULL,
                roles INTEGER NOT NULL,
                dynamic INTEGER NOT NULL,
                relation_roles INTEGER NOT NULL,
                FOREIGN KEY (occurrence_id) REFERENCES graph_occurrences (occurrence_id),
                FOREIGN KEY (caller_usr) REFERENCES graph_symbols (usr),
                FOREIGN KEY (callee_usr) REFERENCES graph_symbols (usr)
            );
            CREATE INDEX IF NOT EXISTS graph_symbols_name ON graph_symbols (name);
            CREATE INDEX IF NOT EXISTS graph_occurrences_usr ON graph_occurrences (usr);
            CREATE INDEX IF NOT EXISTS graph_occurrences_file ON graph_occurrences (file_path);
            CREATE INDEX IF NOT EXISTS graph_definitions_usr ON graph_definitions (usr);
            CREATE INDEX IF NOT EXISTS graph_edges_caller ON graph_edges (caller_usr);
            CREATE INDEX IF NOT EXISTS graph_edges_callee ON graph_edges (callee_usr);
            """
        )
        self._migrate_units_schema()
        self._set_meta("schema_version", str(SCHEMA_VERSION))
        self._connection.commit()

    def _migrate_units_schema(self) -> None:
        """Upgrade databases created before target-aware unit identity."""
        columns = self._connection.execute("PRAGMA table_info(graph_units)").fetchall()
        if not columns:
            return

        # Version 1 used ``unit_name`` as the sole primary key.  Xcode can
        # produce the same unit name for different targets, so rebuild that
        # small table with a composite key while keeping all other tables and
        # their evidence rows intact.
        primary_key_columns = [str(row[1]) for row in columns if int(row[5]) > 0]
        if primary_key_columns != ["unit_name"]:
            return

        legacy_table = "graph_units_v1"
        self._connection.execute(f"DROP TABLE IF EXISTS {legacy_table}")
        self._connection.execute(f"ALTER TABLE graph_units RENAME TO {legacy_table}")
        self._connection.execute(
            """
            CREATE TABLE graph_units (
                unit_name TEXT NOT NULL,
                main_file TEXT NOT NULL,
                target TEXT NOT NULL,
                module_name TEXT NOT NULL,
                output_file TEXT NOT NULL,
                working_directory TEXT NOT NULL,
                sysroot_path TEXT NOT NULL,
                provider_identifier TEXT NOT NULL,
                provider_version TEXT NOT NULL,
                modification_time REAL NOT NULL,
                is_debug_compilation INTEGER NOT NULL,
                is_module_unit INTEGER NOT NULL,
                is_system_unit INTEGER NOT NULL,
                PRIMARY KEY (unit_name, target)
            )
            """
        )
        self._connection.execute(
            """
            INSERT INTO graph_units(
                unit_name, main_file, target, module_name, output_file,
                working_directory, sysroot_path, provider_identifier,
                provider_version, modification_time, is_debug_compilation,
                is_module_unit, is_system_unit
            )
            SELECT unit_name, main_file, target, module_name, output_file,
                   working_directory, sysroot_path, provider_identifier,
                   provider_version, modification_time, is_debug_compilation,
                   is_module_unit, is_system_unit
            FROM graph_units_v1
            """
        )
        self._connection.execute(f"DROP TABLE {legacy_table}")

    def _set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO graph_meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _get_meta(self, key: str) -> str | None:
        row = self._connection.execute("SELECT value FROM graph_meta WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row is not None else None

    def refresh_from_store(
        self,
        store_path: str | Path,
        root_path: str | Path,
        *,
        prefix_mappings: dict[str, str] | list[str] | tuple[str, ...] | None = None,
        library_path: str | Path | None = None,
    ) -> IndexStoreReadResult:
        """Read an IndexStore and replace the persisted snapshot."""
        result = IndexStoreReader(
            store_path,
            root_path,
            prefix_mappings=prefix_mappings,
            library_path=library_path,
        ).read()
        self.replace_snapshot(result)
        return result

    def replace_snapshot(self, result: IndexStoreReadResult) -> None:
        """Atomically replace the graph with a reader result."""
        connection = self._connection
        with connection:
            connection.execute("DELETE FROM graph_edges")
            connection.execute("DELETE FROM graph_definitions")
            connection.execute("DELETE FROM graph_occurrences")
            connection.execute("DELETE FROM graph_symbols")
            connection.execute("DELETE FROM graph_units")

            for unit in result.units:
                connection.execute(
                    """INSERT INTO graph_units(
                        unit_name, main_file, target, module_name, output_file,
                        working_directory, sysroot_path, provider_identifier,
                        provider_version, modification_time, is_debug_compilation,
                        is_module_unit, is_system_unit
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        unit.name,
                        unit.main_file,
                        unit.target,
                        unit.module_name,
                        unit.output_file,
                        unit.working_directory,
                        unit.sysroot_path,
                        unit.provider_identifier,
                        unit.provider_version,
                        unit.modification_time,
                        int(unit.is_debug_compilation),
                        int(unit.is_module_unit),
                        int(unit.is_system_unit),
                    ),
                )

            symbols: dict[str, tuple[str, int, int]] = {}
            for occurrence in result.occurrences:
                existing = symbols.get(occurrence.usr)
                if existing is None or (not existing[0] and occurrence.name):
                    symbols[occurrence.usr] = (occurrence.name, occurrence.language, occurrence.kind)
            connection.executemany(
                "INSERT INTO graph_symbols(usr, name, language, kind) VALUES (?, ?, ?, ?)",
                ((usr, name, language, kind) for usr, (name, language, kind) in symbols.items()),
            )

            for occurrence in result.occurrences:
                cursor = connection.execute(
                    """INSERT INTO graph_occurrences(
                        usr, file_path, line, column_number, roles, unit_name,
                        target, dynamic
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        occurrence.usr,
                        occurrence.file_path,
                        occurrence.line,
                        occurrence.column,
                        occurrence.roles,
                        occurrence.unit_name,
                        occurrence.target,
                        int(occurrence.is_dynamic),
                    ),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("SQLite did not return an occurrence id")
                occurrence_id = int(cursor.lastrowid)

                if occurrence.is_definition:
                    connection.execute(
                        """INSERT INTO graph_definitions(
                            occurrence_id, usr, file_path, line, column_number,
                            unit_name, target
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            occurrence_id,
                            occurrence.usr,
                            occurrence.file_path,
                            occurrence.line,
                            occurrence.column,
                            occurrence.unit_name,
                            occurrence.target,
                        ),
                    )

                if occurrence.is_call and occurrence.caller_relations:
                    for caller_usr, relation_roles in occurrence.caller_relations:
                        if not relation_roles & ROLE_REL_CALLEDBY or caller_usr not in symbols:
                            # A relation can point at a symbol whose defining
                            # occurrence is outside the project.  Keep the
                            # occurrence but do not fabricate an in-project node.
                            continue
                        connection.execute(
                            """INSERT INTO graph_edges(
                                caller_usr, callee_usr, occurrence_id, file_path,
                                line, column_number, unit_name, target, roles,
                                dynamic, relation_roles
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                caller_usr,
                                occurrence.usr,
                                occurrence_id,
                                occurrence.file_path,
                                occurrence.line,
                                occurrence.column,
                                occurrence.unit_name,
                                occurrence.target,
                                occurrence.roles,
                                int(occurrence.is_dynamic),
                                relation_roles,
                            ),
                        )

            self._set_meta("store_path", result.store_path)
            self._set_meta("root_path", result.root_path)
            self._set_meta("fingerprint", result.fingerprint or index_store_fingerprint(result.store_path))
            self._set_meta("refreshed_at", datetime.now(UTC).isoformat())
            self._set_meta("unit_count", str(result.unit_count))
            self._set_meta("record_count", str(result.record_count))

    def status(self) -> GraphStatus:
        """Return counts and freshness information for the snapshot."""
        store_path = self._get_meta("store_path")
        expected_fingerprint = self._get_meta("fingerprint")
        current_fingerprint = index_store_fingerprint(store_path) if store_path else None
        is_current: bool | None
        if expected_fingerprint is None or current_fingerprint is None:
            is_current = None
        else:
            is_current = expected_fingerprint == current_fingerprint
        return GraphStatus(
            database_path=self.database_path,
            store_path=store_path,
            root_path=self._get_meta("root_path"),
            fingerprint=expected_fingerprint,
            is_current=is_current,
            symbols=self._count("graph_symbols"),
            occurrences=self._count("graph_occurrences"),
            edges=self._count("graph_edges"),
            units=self._count("graph_units"),
            refreshed_at=self._get_meta("refreshed_at"),
        )

    def reindex_if_stale(
        self,
        store_path: str | Path,
        root_path: str | Path,
        *,
        prefix_mappings: dict[str, str] | list[str] | tuple[str, ...] | None = None,
        library_path: str | Path | None = None,
    ) -> IndexStoreReadResult | None:
        """Refresh the graph only when the store fingerprint changed."""
        fingerprint = index_store_fingerprint(store_path)
        status = self.status()
        if status.store_path == str(Path(store_path).expanduser().resolve()) and status.fingerprint == fingerprint:
            return None
        return self.refresh_from_store(
            store_path,
            root_path,
            prefix_mappings=prefix_mappings,
            library_path=library_path,
        )

    def _count(self, table: str) -> int:
        row = self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0])

    def resolve_symbols(self, query: str, *, file_path: str | None = None) -> list[GraphSymbol]:
        """Resolve a USR or exact symbol name to compiler symbols."""
        if query.startswith("usr:"):
            query = query[4:]
        if query.startswith(("c:", "s:", "$s")):
            rows = self._connection.execute("SELECT * FROM graph_symbols WHERE usr = ?", (query,)).fetchall()
        else:
            rows = self._connection.execute("SELECT * FROM graph_symbols WHERE name = ? ORDER BY usr", (query,)).fetchall()
        symbols = [self._symbol_from_row(row) for row in rows]
        if file_path is None:
            return symbols
        root_path = self._get_meta("root_path")
        candidate = Path(file_path).expanduser()
        if not candidate.is_absolute() and root_path:
            candidate = Path(root_path) / candidate
        normalized = str(candidate.resolve())
        return [symbol for symbol in symbols if self._symbol_has_file(symbol.usr, normalized, file_path)]

    def _symbol_has_file(self, usr: str, normalized: str, original: str) -> bool:
        rows = self._connection.execute("SELECT file_path FROM graph_occurrences WHERE usr = ?", (usr,)).fetchall()
        return any(str(Path(row[0]).expanduser().resolve()) == normalized or row[0] == original for row in rows)

    @staticmethod
    def _symbol_from_row(row: sqlite3.Row) -> GraphSymbol:
        return GraphSymbol(str(row["usr"]), str(row["name"]), int(row["language"]), int(row["kind"]))

    def _require_symbol(self, query: str, file_path: str | None = None) -> GraphSymbol:
        symbols = self.resolve_symbols(query, file_path=file_path)
        if not symbols:
            raise KeyError(f"No indexed symbol matched {query!r}")
        if len(symbols) > 1:
            raise KeyError(f"Ambiguous symbol {query!r}; use a USR or --file to disambiguate")
        symbol = symbols[0]
        row = self._connection.execute(
            """SELECT file_path, line, column_number FROM graph_definitions
               WHERE usr = ?
               ORDER BY file_path, line, column_number LIMIT 1""",
            (symbol.usr,),
        ).fetchone()
        if row is None:
            return symbol
        return GraphSymbol(symbol.usr, symbol.name, symbol.language, symbol.kind, str(row[0]), int(row[1]), int(row[2]))

    def callers(self, query: str, *, file_path: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """Return direct callers and retained call-site evidence."""
        symbol = self._require_symbol(query, file_path=file_path)
        rows = self._connection.execute(
            """SELECT e.*, s.name AS caller_name, s.language AS caller_language, s.kind AS caller_kind,
                      u.module_name AS unit_module, u.is_debug_compilation AS unit_is_debug
               FROM graph_edges e JOIN graph_symbols s ON s.usr = e.caller_usr
               LEFT JOIN graph_units u ON u.unit_name = e.unit_name AND u.target = e.target
               WHERE e.callee_usr = ? ORDER BY e.file_path, e.line, e.column_number LIMIT ?""",
            (symbol.usr, limit),
        ).fetchall()
        return [self._edge_dict(row, symbol, "caller_name", "caller_language", "caller_kind") for row in rows]

    def callees(self, query: str, *, file_path: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """Return direct callees and retained call-site evidence."""
        symbol = self._require_symbol(query, file_path=file_path)
        rows = self._connection.execute(
            """SELECT e.*, s.name AS callee_name, s.language AS callee_language, s.kind AS callee_kind,
                      u.module_name AS unit_module, u.is_debug_compilation AS unit_is_debug
               FROM graph_edges e JOIN graph_symbols s ON s.usr = e.callee_usr
               LEFT JOIN graph_units u ON u.unit_name = e.unit_name AND u.target = e.target
               WHERE e.caller_usr = ? ORDER BY e.file_path, e.line, e.column_number LIMIT ?""",
            (symbol.usr, limit),
        ).fetchall()
        return [self._edge_dict(row, symbol, "callee_name", "callee_language", "callee_kind") for row in rows]

    def _edge_dict(self, row: sqlite3.Row, symbol: GraphSymbol, name_key: str, language_key: str, kind_key: str) -> dict[str, Any]:
        related_key = "caller_usr" if name_key == "caller_name" else "callee_usr"
        target_triple = str(row["target"])
        return {
            "symbol": {
                "usr": str(row[related_key]),
                "name": str(row[name_key]),
                "language": int(row[language_key]),
                "kind": int(row[kind_key]),
            },
            "target": {"usr": symbol.usr, "name": symbol.name},
            "evidence": {
                "file": str(row["file_path"]),
                "line": int(row["line"]),
                "column": int(row["column_number"]),
                "unit": str(row["unit_name"]),
                "module": str(row["unit_module"] or ""),
                "target_triple": target_triple,
                "architecture": target_triple.split("-", 1)[0] if target_triple else "",
                "is_debug_compilation": bool(row["unit_is_debug"]),
                "dynamic": bool(row["dynamic"]),
                "relation_roles": int(row["relation_roles"]),
            },
        }

    def impact(
        self,
        query: str,
        *,
        direction: str = "callers",
        depth: int = 3,
        file_path: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Walk callers or callees to a bounded depth."""
        if direction not in {"callers", "callees"}:
            raise ValueError("direction must be 'callers' or 'callees'")
        if depth < 0:
            raise ValueError("depth must be non-negative")
        root = self._require_symbol(query, file_path=file_path)
        result: list[dict[str, Any]] = []
        visited = {root.usr}
        queue: deque[tuple[str, int]] = deque([(root.usr, 0)])
        while queue and len(result) < limit:
            current, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            column = "e.callee_usr" if direction == "callers" else "e.caller_usr"
            related_column = "e.caller_usr" if direction == "callers" else "e.callee_usr"
            related_key = "caller_usr" if direction == "callers" else "callee_usr"
            rows = self._connection.execute(
                f"""SELECT e.*, s.name AS related_name, s.language AS related_language, s.kind AS related_kind,
                           u.module_name AS unit_module, u.is_debug_compilation AS unit_is_debug
                    FROM graph_edges e JOIN graph_symbols s ON s.usr = {related_column}
                    LEFT JOIN graph_units u ON u.unit_name = e.unit_name AND u.target = e.target
                    WHERE {column} = ? ORDER BY e.file_path, e.line, e.column_number""",
                (current,),
            ).fetchall()
            for row in rows:
                related_usr = str(row[related_key])
                target_triple = str(row["target"])
                item = {
                    "usr": related_usr,
                    "name": str(row["related_name"]),
                    "language": int(row["related_language"]),
                    "kind": int(row["related_kind"]),
                    "depth": current_depth + 1,
                    "via": {
                        "file": str(row["file_path"]),
                        "line": int(row["line"]),
                        "column": int(row["column_number"]),
                        "unit": str(row["unit_name"]),
                        "module": str(row["unit_module"] or ""),
                        "target_triple": target_triple,
                        "architecture": target_triple.split("-", 1)[0] if target_triple else "",
                        "is_debug_compilation": bool(row["unit_is_debug"]),
                        "dynamic": bool(row["dynamic"]),
                    },
                }
                result.append(item)
                if related_usr not in visited:
                    visited.add(related_usr)
                    queue.append((related_usr, current_depth + 1))
                if len(result) >= limit:
                    break
        return result

    def trace(
        self,
        start: str,
        end: str,
        *,
        max_depth: int = 8,
        start_file: str | None = None,
        end_file: str | None = None,
        max_paths: int = 100,
    ) -> list[list[dict[str, Any]]]:
        """Return bounded direct-call paths between two symbols."""
        source = self._require_symbol(start, file_path=start_file)
        destination = self._require_symbol(end, file_path=end_file)
        paths: list[list[dict[str, Any]]] = []

        def visit(current: str, path: list[dict[str, Any]], seen: set[str]) -> None:
            if len(paths) >= max_paths or len(path) - 1 > max_depth:
                return
            if current == destination.usr:
                paths.append(path)
                return
            rows = self._connection.execute(
                """SELECT e.*, s.name AS callee_name, s.language AS callee_language, s.kind AS callee_kind,
                          u.module_name AS unit_module, u.is_debug_compilation AS unit_is_debug
                   FROM graph_edges e JOIN graph_symbols s ON s.usr = e.callee_usr
                   LEFT JOIN graph_units u ON u.unit_name = e.unit_name AND u.target = e.target
                   WHERE e.caller_usr = ? ORDER BY e.file_path, e.line, e.column_number""",
                (current,),
            ).fetchall()
            for row in rows:
                next_usr = str(row["callee_usr"])
                if next_usr in seen:
                    continue
                target_triple = str(row["target"])
                step = {
                    "usr": next_usr,
                    "name": str(row["callee_name"]),
                    "evidence": {
                        "file": str(row["file_path"]),
                        "line": int(row["line"]),
                        "column": int(row["column_number"]),
                        "unit": str(row["unit_name"]),
                        "module": str(row["unit_module"] or ""),
                        "target_triple": target_triple,
                        "architecture": target_triple.split("-", 1)[0] if target_triple else "",
                        "is_debug_compilation": bool(row["unit_is_debug"]),
                        "dynamic": bool(row["dynamic"]),
                    },
                }
                visit(next_usr, path + [step], seen | {next_usr})

        visit(source.usr, [{"usr": source.usr, "name": source.name}], {source.usr})
        return paths

    def to_json(self, value: Any) -> str:
        """Serialize query data for agent-friendly CLI output."""
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


__all__ = [
    "GraphEvidence",
    "GraphStatus",
    "GraphSymbol",
    "IndexStoreError",
    "IndexStoreGraph",
]
