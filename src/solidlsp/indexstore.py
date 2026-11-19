"""Read Apple's compiler-produced IndexStore on macOS.

The reader deliberately keeps the raw compiler identity (USR) and every
occurrence.  A caller can therefore build a persistent graph without losing
call-site locations or conflating different build configurations.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import plistlib
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solidlsp.util.subprocess_util import subprocess_kwargs

log = logging.getLogger(__name__)


# These values are part of Apple's public indexstore.h C API and have been
# stable since the IndexStore format used by Xcode 11.
ROLE_DECLARATION = 1 << 0
ROLE_DEFINITION = 1 << 1
ROLE_REFERENCE = 1 << 2
ROLE_READ = 1 << 3
ROLE_WRITE = 1 << 4
ROLE_CALL = 1 << 5
ROLE_DYNAMIC = 1 << 6
ROLE_IMPLICIT = 1 << 8
ROLE_REL_CALLEDBY = 1 << 13
ROLE_REL_CONTAINEDBY = 1 << 16


class IndexStoreError(RuntimeError):
    """Raised when an IndexStore cannot be opened or read."""


@dataclass(frozen=True)
class IndexStoreOccurrence:
    """One compiler occurrence in a source file.

    ``line`` and ``column`` use Apple's one-based line and one-based UTF-8
    column convention.  ``caller_usrs`` contains every
    containing symbol reported by the compiler, rather than just the first
    relation.
    """

    usr: str
    name: str
    language: int
    kind: int
    roles: int
    file_path: str
    line: int
    column: int
    caller_usrs: tuple[str, ...] = ()
    unit_name: str = ""
    target: str = ""
    caller_relations: tuple[tuple[str, int], ...] = ()

    @property
    def is_definition(self) -> bool:
        """Whether this occurrence defines the symbol."""
        return bool(self.roles & ROLE_DEFINITION)

    @property
    def is_call(self) -> bool:
        """Whether this occurrence is a call."""
        return bool(self.roles & ROLE_CALL)

    @property
    def is_dynamic(self) -> bool:
        """Whether the compiler marked the occurrence as dynamically dispatched."""
        return bool(self.roles & ROLE_DYNAMIC)


@dataclass(frozen=True)
class IndexStoreReadResult:
    """Occurrences and basic metadata collected from an IndexStore."""

    store_path: str
    root_path: str
    occurrences: tuple[IndexStoreOccurrence, ...]
    unit_count: int
    record_count: int
    fingerprint: str = ""
    units: tuple["IndexStoreUnit", ...] = ()

    @property
    def call_count(self) -> int:
        """Return the number of call occurrences."""
        return sum(1 for occurrence in self.occurrences if occurrence.is_call)


@dataclass(frozen=True)
class IndexStoreCandidate:
    """A discovered Xcode DataStore and its associated workspace, if known."""

    store_path: str
    workspace_path: str | None
    modified_time: float


@dataclass(frozen=True)
class IndexStoreUnit:
    """Metadata for one compiler-produced compilation unit."""

    name: str
    main_file: str
    target: str = ""
    module_name: str = ""
    output_file: str = ""
    working_directory: str = ""
    sysroot_path: str = ""
    provider_identifier: str = ""
    provider_version: str = ""
    modification_time: float = 0.0
    is_debug_compilation: bool = False
    is_module_unit: bool = False
    is_system_unit: bool = False


class _StringRef(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("length", ctypes.c_size_t)]


_VoidPointer = ctypes.c_void_p
_StoreCallback = ctypes.CFUNCTYPE(ctypes.c_bool, _VoidPointer, _StringRef)
_DependencyCallback = ctypes.CFUNCTYPE(ctypes.c_bool, _VoidPointer, _VoidPointer)
_OccurrenceCallback = ctypes.CFUNCTYPE(ctypes.c_bool, _VoidPointer, _VoidPointer)
_RelationCallback = ctypes.CFUNCTYPE(ctypes.c_bool, _VoidPointer, _VoidPointer)


def _decode_string(value: _StringRef) -> str:
    """Decode an IndexStore string reference without assuming NUL termination."""
    if not value.data or value.length == 0:
        return ""
    raw = ctypes.string_at(value.data, value.length)
    return raw.decode("utf-8", errors="replace")


def _expand_path(value: str, base_path: str | None = None) -> str:
    """Expand shell-style variables and make a relative path absolute."""
    expanded = os.path.expandvars(os.path.expanduser(value))
    if base_path and not os.path.isabs(expanded):
        expanded = os.path.join(base_path, expanded)
    return os.path.realpath(expanded)


def _path_within(path: str, root: str) -> bool:
    """Return whether ``path`` is contained by ``root``."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def index_store_fingerprint(store_path: str | os.PathLike[str]) -> str:
    """Return a content-oriented fingerprint for an IndexStore directory.

    The DataStore directory's own mtime is not updated consistently by Xcode.
    Hashing the relative unit/record file metadata makes freshness checks work
    for both the normal ``v*/units`` layout and older stores.
    """
    import hashlib

    root = Path(store_path).expanduser().resolve()
    digest = hashlib.sha256()
    if not root.exists():
        return ""
    try:
        paths = sorted(path for path in root.rglob("*") if path.is_file() and path.relative_to(root).parts[0].startswith("v"))
        if not paths:
            paths = sorted(path for path in root.rglob("*") if path.is_file())
    except OSError:
        return ""
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _latest_unit_modification_time(store_path: Path) -> float:
    """Return the newest unit-file mtime, falling back to the store mtime."""
    latest = 0.0
    try:
        for path in store_path.rglob("*"):
            if path.is_file() and path.relative_to(store_path).parts[0].startswith("v"):
                latest = max(latest, path.stat().st_mtime)
        return latest or store_path.stat().st_mtime
    except OSError:
        return 0.0


def discover_index_store_candidates(
    project_root: str | os.PathLike[str],
    *,
    derived_data_root: str | os.PathLike[str] | None = None,
) -> list[IndexStoreCandidate]:
    """Find Xcode DataStores which plausibly belong to a project.

    Discovery is intentionally opt-in.  It uses ``info.plist``'s
    ``WorkspacePath`` when available and otherwise falls back to a matching
    DerivedData directory name.  Candidates are newest first.
    """
    root = Path(project_root).expanduser().resolve()
    dd_root = Path(derived_data_root or (Path.home() / "Library/Developer/Xcode/DerivedData")).expanduser()
    if not dd_root.is_dir():
        return []

    candidates: list[IndexStoreCandidate] = []
    project_name = root.name.lower()
    for derived_dir in dd_root.iterdir():
        if not derived_dir.is_dir():
            continue
        store = derived_dir / "Index.noindex" / "DataStore"
        if not store.exists():
            continue

        workspace_path: str | None = None
        info_path = derived_dir / "info.plist"
        if info_path.is_file():
            try:
                with info_path.open("rb") as stream:
                    info = plistlib.load(stream)
                raw_workspace = info.get("WorkspacePath")
                if isinstance(raw_workspace, str) and raw_workspace:
                    workspace_path = _expand_path(raw_workspace)
            except (OSError, plistlib.InvalidFileException, ValueError) as exc:
                log.debug("Could not read DerivedData metadata %s: %s", info_path, exc)

        workspace_matches = False
        if workspace_path:
            workspace_matches = _path_within(workspace_path, str(root)) or _path_within(str(root), workspace_path)
        derived_name = derived_dir.name.lower()
        name_matches = derived_name == project_name or derived_name.startswith(project_name + "-")
        if workspace_path and not workspace_matches and not name_matches:
            continue
        if not workspace_path and not name_matches:
            continue

        try:
            modified_time = _latest_unit_modification_time(store)
        except OSError:
            continue
        candidates.append(IndexStoreCandidate(str(store.resolve()), workspace_path, modified_time))

    return sorted(candidates, key=lambda candidate: candidate.modified_time, reverse=True)


def discover_index_store_path(project_root: str | os.PathLike[str]) -> str | None:
    """Return the newest plausible Xcode DataStore for ``project_root``."""
    candidates = discover_index_store_candidates(project_root)
    return candidates[0].store_path if candidates else None


class IndexStoreReader:
    """Read source occurrences from Apple's ``libIndexStore`` C API."""

    def __init__(
        self,
        store_path: str | os.PathLike[str],
        root_path: str | os.PathLike[str],
        *,
        prefix_mappings: Mapping[str, str] | Sequence[str] | None = None,
        library_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.store_path = _expand_path(os.fspath(store_path))
        self.root_path = _expand_path(os.fspath(root_path))
        self.prefix_mappings = self._normalise_prefix_mappings(prefix_mappings)
        self.library_path = self._resolve_library_path(library_path)
        self._library = self._load_library()
        self._configure_functions()

    @staticmethod
    def _normalise_prefix_mappings(
        mappings: Mapping[str, str] | Sequence[str] | None,
    ) -> tuple[tuple[str, str], ...]:
        """Validate and sort path prefix mappings from configuration."""
        if mappings is None:
            return ()
        if isinstance(mappings, Mapping):
            pairs = list(mappings.items())
        else:
            pairs = []
            for raw in mappings:
                if not isinstance(raw, str) or "=" not in raw:
                    raise ValueError("IndexStore prefix mappings must use 'remote=local' syntax")
                pairs.append(tuple(raw.split("=", 1)))
        normalised: list[tuple[str, str]] = []
        for source, destination in pairs:
            if not isinstance(source, str) or not isinstance(destination, str) or not source or not destination:
                raise ValueError("IndexStore prefix mappings must contain non-empty string paths")
            normalised.append((_expand_path(source), _expand_path(destination)))
        return tuple(sorted(normalised, key=lambda pair: len(pair[0]), reverse=True))

    @staticmethod
    def _resolve_library_path(library_path: str | os.PathLike[str] | None) -> str:
        """Resolve the active Xcode toolchain's IndexStore library."""
        if library_path:
            path = _expand_path(os.fspath(library_path))
            if not os.path.isfile(path):
                raise IndexStoreError(f"Configured IndexStore library does not exist: {path}")
            return path
        if sys.platform != "darwin":
            raise IndexStoreError("Xcode IndexStore reading is only available on macOS")

        try:
            result = subprocess.run(
                ["xcrun", "--find", "swift"],
                check=False,
                capture_output=True,
                text=True,
                **subprocess_kwargs(),
            )
            if result.returncode == 0 and result.stdout.strip():
                swift_bin = Path(result.stdout.strip()).resolve()
                candidate = swift_bin.parent.parent / "lib" / "libIndexStore.dylib"
                if candidate.is_file():
                    return str(candidate)
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("Could not resolve IndexStore library through xcrun: %s", exc)

        discovered = ctypes.util.find_library("IndexStore")
        if discovered:
            return discovered
        raise IndexStoreError("Could not locate libIndexStore.dylib; install Xcode and select it with xcode-select")

    def _load_library(self) -> ctypes.CDLL:
        """Load the native library and translate loader failures."""
        try:
            return ctypes.CDLL(self.library_path)
        except OSError as exc:
            raise IndexStoreError(f"Could not load {self.library_path}: {exc}") from exc

    def _configure_functions(self) -> None:
        """Declare the small stable subset of the IndexStore ABI we use."""
        library = self._library

        def configure(name: str, argtypes: list[Any], restype: Any) -> Any:
            try:
                function = getattr(library, name)
            except AttributeError as exc:
                raise IndexStoreError(f"{self.library_path} does not export {name}") from exc
            function.argtypes = argtypes
            function.restype = restype
            return function

        self._store_create = configure("indexstore_store_create", [ctypes.c_char_p, ctypes.POINTER(_VoidPointer)], _VoidPointer)
        self._store_dispose = configure("indexstore_store_dispose", [_VoidPointer], None)
        self._error_description = configure("indexstore_error_get_description", [_VoidPointer], ctypes.c_char_p)
        self._error_dispose = getattr(library, "indexstore_error_dispose", None)
        if self._error_dispose is not None:
            self._error_dispose.argtypes = [_VoidPointer]
            self._error_dispose.restype = None
        self._store_units = configure(
            "indexstore_store_units_apply_f", [_VoidPointer, ctypes.c_uint, _VoidPointer, _StoreCallback], ctypes.c_bool
        )
        self._unit_create = configure(
            "indexstore_unit_reader_create", [_VoidPointer, ctypes.c_char_p, ctypes.POINTER(_VoidPointer)], _VoidPointer
        )
        self._unit_dispose = configure("indexstore_unit_reader_dispose", [_VoidPointer], None)
        self._unit_main_file = configure("indexstore_unit_reader_get_main_file", [_VoidPointer], _StringRef)
        self._unit_target = configure("indexstore_unit_reader_get_target", [_VoidPointer], _StringRef)
        self._unit_module_name = getattr(library, "indexstore_unit_reader_get_module_name", None)
        self._unit_output_file = getattr(library, "indexstore_unit_reader_get_output_file", None)
        self._unit_working_directory = getattr(library, "indexstore_unit_reader_get_working_dir", None)
        self._unit_sysroot_path = getattr(library, "indexstore_unit_reader_get_sysroot_path", None)
        self._unit_provider_identifier = getattr(library, "indexstore_unit_reader_get_provider_identifier", None)
        self._unit_provider_version = getattr(library, "indexstore_unit_reader_get_provider_version", None)
        self._unit_modification_time = getattr(library, "indexstore_unit_reader_get_modification_time", None)
        self._unit_is_debug = getattr(library, "indexstore_unit_reader_is_debug_compilation", None)
        self._unit_is_module = getattr(library, "indexstore_unit_reader_is_module_unit", None)
        self._unit_is_system = getattr(library, "indexstore_unit_reader_is_system_unit", None)
        for function in (
            self._unit_module_name,
            self._unit_output_file,
            self._unit_working_directory,
            self._unit_sysroot_path,
            self._unit_provider_identifier,
            self._unit_provider_version,
        ):
            if function is not None:
                function.argtypes = [_VoidPointer]
                function.restype = _StringRef
        if self._unit_modification_time is not None:
            self._unit_modification_time.argtypes = [
                _VoidPointer,
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.POINTER(ctypes.c_longlong),
            ]
            self._unit_modification_time.restype = None
        for function in (self._unit_is_debug, self._unit_is_module, self._unit_is_system):
            if function is not None:
                function.argtypes = [_VoidPointer]
                function.restype = ctypes.c_bool
        self._unit_dependencies = configure(
            "indexstore_unit_reader_dependencies_apply_f", [_VoidPointer, _VoidPointer, _DependencyCallback], ctypes.c_bool
        )
        self._dependency_kind = configure("indexstore_unit_dependency_get_kind", [_VoidPointer], ctypes.c_int)
        self._dependency_system = configure("indexstore_unit_dependency_is_system", [_VoidPointer], ctypes.c_bool)
        self._dependency_name = configure("indexstore_unit_dependency_get_name", [_VoidPointer], _StringRef)
        self._dependency_file = configure("indexstore_unit_dependency_get_filepath", [_VoidPointer], _StringRef)
        self._record_create = configure(
            "indexstore_record_reader_create", [_VoidPointer, ctypes.c_char_p, ctypes.POINTER(_VoidPointer)], _VoidPointer
        )
        self._record_dispose = configure("indexstore_record_reader_dispose", [_VoidPointer], None)
        self._record_occurrences = configure(
            "indexstore_record_reader_occurrences_apply_f", [_VoidPointer, _VoidPointer, _OccurrenceCallback], ctypes.c_bool
        )
        self._occurrence_symbol = configure("indexstore_occurrence_get_symbol", [_VoidPointer], _VoidPointer)
        self._occurrence_roles = configure("indexstore_occurrence_get_roles", [_VoidPointer], ctypes.c_uint64)
        self._occurrence_line_col = configure(
            "indexstore_occurrence_get_line_col", [_VoidPointer, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)], None
        )
        self._occurrence_relations = configure(
            "indexstore_occurrence_relations_apply_f", [_VoidPointer, _VoidPointer, _RelationCallback], ctypes.c_bool
        )
        self._relation_roles = configure("indexstore_symbol_relation_get_roles", [_VoidPointer], ctypes.c_uint64)
        self._relation_symbol = configure("indexstore_symbol_relation_get_symbol", [_VoidPointer], _VoidPointer)
        self._symbol_usr = configure("indexstore_symbol_get_usr", [_VoidPointer], _StringRef)
        self._symbol_name = configure("indexstore_symbol_get_name", [_VoidPointer], _StringRef)
        self._symbol_language = configure("indexstore_symbol_get_language", [_VoidPointer], ctypes.c_int)
        self._symbol_kind = configure("indexstore_symbol_get_kind", [_VoidPointer], ctypes.c_int)

    def _remap_path(self, path: str) -> str:
        """Apply configured prefix mappings to an IndexStore path."""
        if not path:
            return path
        candidate = _expand_path(path)
        for source, destination in self.prefix_mappings:
            if candidate == source or candidate.startswith(source + os.sep):
                return destination + candidate[len(source) :]
        return candidate

    def _open_store(self) -> _VoidPointer:
        """Open the store and produce a useful native error description."""
        error = _VoidPointer()
        store = self._store_create(self.store_path.encode("utf-8"), ctypes.byref(error))
        if store:
            return store
        description = "unknown error"
        if error:
            raw_description = self._error_description(error)
            if raw_description:
                description = raw_description.decode("utf-8", errors="replace")
            if self._error_dispose is not None:
                self._error_dispose(error)
        raise IndexStoreError(f"Could not open IndexStore {self.store_path}: {description}")

    def _collect_units(self, store: _VoidPointer) -> list[str]:
        """Collect all compilation unit names."""
        units: list[str] = []

        @_StoreCallback
        def callback(_context: _VoidPointer, name: _StringRef) -> bool:
            decoded = _decode_string(name)
            if decoded:
                units.append(decoded)
            return True

        if not self._store_units(store, 0, None, callback):
            log.warning("IndexStore stopped while enumerating units: %s", self.store_path)
        return units

    def _read_unit_records(self, store: _VoidPointer, unit_name: str) -> tuple[IndexStoreUnit, list[tuple[str, str]]]:
        """Read the main file, target and source records of one unit."""
        error = _VoidPointer()
        reader = self._unit_create(store, unit_name.encode("utf-8"), ctypes.byref(error))
        if not reader:
            if error and self._error_dispose is not None:
                self._error_dispose(error)
            return IndexStoreUnit(unit_name, ""), []
        try:
            main_file = self._remap_path(_decode_string(self._unit_main_file(reader)))
            target = _decode_string(self._unit_target(reader))
            unit = IndexStoreUnit(
                name=unit_name,
                main_file=main_file,
                target=target,
                module_name=self._optional_unit_string(self._unit_module_name, reader),
                output_file=self._optional_unit_string(self._unit_output_file, reader),
                working_directory=self._optional_unit_string(self._unit_working_directory, reader),
                sysroot_path=self._optional_unit_string(self._unit_sysroot_path, reader),
                provider_identifier=self._optional_unit_string(self._unit_provider_identifier, reader),
                provider_version=self._optional_unit_string(self._unit_provider_version, reader),
                modification_time=self._read_unit_modification_time(reader),
                is_debug_compilation=bool(self._unit_is_debug(reader)) if self._unit_is_debug is not None else False,
                is_module_unit=bool(self._unit_is_module(reader)) if self._unit_is_module is not None else False,
                is_system_unit=bool(self._unit_is_system(reader)) if self._unit_is_system is not None else False,
            )
            records: list[tuple[str, str]] = []

            @_DependencyCallback
            def callback(_context: _VoidPointer, dependency: _VoidPointer) -> bool:
                if self._dependency_kind(dependency) != 2 or self._dependency_system(dependency):
                    return True
                name = _decode_string(self._dependency_name(dependency))
                path = self._remap_path(_decode_string(self._dependency_file(dependency)))
                if name and path:
                    records.append((name, path))
                return True

            self._unit_dependencies(reader, None, callback)
            if not records and main_file and _path_within(main_file, self.root_path):
                # Some hand-produced stores omit record dependencies.  The unit
                # name itself is not necessarily a record name, so do not guess
                # one; callers can still use the unit metadata.
                log.debug("Unit %s has no source record dependency", unit_name)
            return unit, records
        finally:
            self._unit_dispose(reader)

    @staticmethod
    def _optional_unit_string(function: Any, reader: _VoidPointer) -> str:
        """Read an optional unit string exposed by newer IndexStore versions."""
        if function is None:
            return ""
        return _decode_string(function(reader))

    def _read_unit_modification_time(self, reader: _VoidPointer) -> float:
        """Read a unit's Unix modification timestamp when the ABI supports it."""
        if self._unit_modification_time is None:
            return 0.0
        seconds = ctypes.c_longlong(0)
        nanoseconds = ctypes.c_longlong(0)
        self._unit_modification_time(reader, ctypes.byref(seconds), ctypes.byref(nanoseconds))
        return float(seconds.value) + float(nanoseconds.value) / 1_000_000_000

    def _read_record(
        self,
        store: _VoidPointer,
        record_name: str,
        record_file: str,
        unit_name: str,
        target: str,
    ) -> list[IndexStoreOccurrence]:
        """Read every occurrence from one source record."""
        error = _VoidPointer()
        reader = self._record_create(store, record_name.encode("utf-8"), ctypes.byref(error))
        if not reader:
            if error and self._error_dispose is not None:
                self._error_dispose(error)
            return []
        occurrences: list[IndexStoreOccurrence] = []
        try:

            @_OccurrenceCallback
            def occurrence_callback(_context: _VoidPointer, occurrence: _VoidPointer) -> bool:
                roles = int(self._occurrence_roles(occurrence))
                symbol = self._occurrence_symbol(occurrence)
                if not symbol:
                    return True
                usr = _decode_string(self._symbol_usr(symbol))
                if not usr:
                    return True
                line = ctypes.c_uint(0)
                column = ctypes.c_uint(0)
                self._occurrence_line_col(occurrence, ctypes.byref(line), ctypes.byref(column))
                caller_usrs: list[str] = []
                caller_relations: list[tuple[str, int]] = []

                @_RelationCallback
                def relation_callback(_relation_context: _VoidPointer, relation: _VoidPointer) -> bool:
                    relation_roles = int(self._relation_roles(relation))
                    if relation_roles & (ROLE_REL_CALLEDBY | ROLE_REL_CONTAINEDBY):
                        related_symbol = self._relation_symbol(relation)
                        related_usr = _decode_string(self._symbol_usr(related_symbol)) if related_symbol else ""
                        if related_usr and related_usr not in caller_usrs:
                            caller_usrs.append(related_usr)
                            caller_relations.append((related_usr, relation_roles))
                    return True

                self._occurrence_relations(occurrence, None, relation_callback)
                occurrences.append(
                    IndexStoreOccurrence(
                        usr=usr,
                        name=_decode_string(self._symbol_name(symbol)),
                        language=int(self._symbol_language(symbol)),
                        kind=int(self._symbol_kind(symbol)),
                        roles=roles,
                        file_path=record_file,
                        line=int(line.value),
                        column=int(column.value),
                        caller_usrs=tuple(caller_usrs),
                        unit_name=unit_name,
                        target=target,
                        caller_relations=tuple(caller_relations),
                    )
                )
                return True

            self._record_occurrences(reader, None, occurrence_callback)
            return occurrences
        finally:
            self._record_dispose(reader)

    def read(self) -> IndexStoreReadResult:
        """Read project-local occurrences from the configured store."""
        if not os.path.exists(self.store_path):
            raise IndexStoreError(f"IndexStore path does not exist: {self.store_path}")
        if not os.path.isdir(self.root_path):
            raise IndexStoreError(f"Project root does not exist or is not a directory: {self.root_path}")

        store = self._open_store()
        try:
            units = self._collect_units(store)
            occurrences: list[IndexStoreOccurrence] = []
            seen_records: set[tuple[str, str, str, str]] = set()
            unit_metadata: list[IndexStoreUnit] = []
            record_count = 0
            for unit_name in units:
                unit, records = self._read_unit_records(store, unit_name)
                unit_metadata.append(unit)
                for record_name, record_file in records:
                    if not _path_within(record_file, self.root_path):
                        continue
                    key = (record_name, record_file, unit_name, unit.target)
                    if key in seen_records:
                        continue
                    seen_records.add(key)
                    record_count += 1
                    occurrences.extend(self._read_record(store, record_name, record_file, unit_name, unit.target))
            return IndexStoreReadResult(
                store_path=self.store_path,
                root_path=self.root_path,
                occurrences=tuple(occurrences),
                unit_count=len(units),
                record_count=record_count,
                fingerprint=index_store_fingerprint(self.store_path),
                units=tuple(unit_metadata),
            )
        finally:
            self._store_dispose(store)


def read_indexstore(
    store_path: str | os.PathLike[str],
    root_path: str | os.PathLike[str],
    *,
    prefix_mappings: Mapping[str, str] | Sequence[str] | None = None,
) -> IndexStoreReadResult:
    """Read a project-local Xcode IndexStore in one call."""
    return IndexStoreReader(store_path, root_path, prefix_mappings=prefix_mappings).read()


__all__ = [
    "ROLE_CALL",
    "ROLE_DEFINITION",
    "ROLE_DYNAMIC",
    "IndexStoreCandidate",
    "IndexStoreError",
    "IndexStoreOccurrence",
    "IndexStoreReadResult",
    "IndexStoreReader",
    "IndexStoreUnit",
    "discover_index_store_candidates",
    "discover_index_store_path",
    "index_store_fingerprint",
    "read_indexstore",
]
