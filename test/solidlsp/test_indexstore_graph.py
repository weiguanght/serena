import sqlite3
from pathlib import Path

from solidlsp.indexstore import (
    ROLE_CALL,
    ROLE_DEFINITION,
    ROLE_REL_CALLEDBY,
    IndexStoreOccurrence,
    IndexStoreReadResult,
    IndexStoreUnit,
)
from solidlsp.indexstore_graph import IndexStoreGraph


def _snapshot(root: Path) -> IndexStoreReadResult:
    foo = "s:foo"
    caller = "s:caller"
    occurrences = (
        IndexStoreOccurrence(foo, "foo", 1, 12, ROLE_DEFINITION, str(root / "Foo.swift"), 1, 1, unit_name="Foo.o", target="debug"),
        IndexStoreOccurrence(
            caller,
            "caller",
            1,
            12,
            ROLE_DEFINITION,
            str(root / "Caller.swift"),
            1,
            1,
            unit_name="Caller.o",
            target="debug",
        ),
        IndexStoreOccurrence(
            foo,
            "foo",
            1,
            12,
            ROLE_CALL,
            str(root / "Caller.swift"),
            3,
            5,
            caller_usrs=(caller,),
            unit_name="Caller.o",
            target="debug",
            caller_relations=((caller, ROLE_REL_CALLEDBY),),
        ),
    )
    return IndexStoreReadResult(
        store_path=str(root / "DataStore"),
        root_path=str(root),
        occurrences=occurrences,
        unit_count=2,
        record_count=2,
        fingerprint="test-fingerprint",
        units=(
            IndexStoreUnit("Foo.o", str(root / "Foo.swift"), target="debug"),
            IndexStoreUnit("Caller.o", str(root / "Caller.swift"), target="debug"),
        ),
    )


def test_graph_preserves_call_evidence_and_traversal(tmp_path: Path) -> None:
    with IndexStoreGraph(tmp_path / "graph.sqlite3") as graph:
        graph.replace_snapshot(_snapshot(tmp_path))

        callers = graph.callers("foo")
        assert len(callers) == 1
        assert callers[0]["symbol"]["name"] == "caller"
        assert callers[0]["evidence"]["line"] == 3
        assert callers[0]["evidence"]["relation_roles"] == ROLE_REL_CALLEDBY
        assert callers[0]["evidence"]["target_triple"] == "debug"
        assert callers[0]["evidence"]["architecture"] == "debug"

        impact = graph.impact("foo", direction="callers", depth=1)
        assert [item["name"] for item in impact] == ["caller"]

        traces = graph.trace("caller", "foo")
        assert len(traces) == 1
        assert [step["name"] for step in traces[0]] == ["caller", "foo"]

        status = graph.status()
        assert status.symbols == 2
        assert status.occurrences == 3
        assert status.edges == 1
        assert status.units == 2


def test_graph_requires_usr_for_ambiguous_names(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    duplicate = IndexStoreOccurrence(
        "s:other-foo",
        "foo",
        1,
        12,
        ROLE_DEFINITION,
        str(tmp_path / "Other.swift"),
        1,
        1,
    )
    snapshot = IndexStoreReadResult(
        snapshot.store_path,
        snapshot.root_path,
        snapshot.occurrences + (duplicate,),
        snapshot.unit_count,
        snapshot.record_count,
        snapshot.fingerprint,
        snapshot.units,
    )
    with IndexStoreGraph(tmp_path / "graph.sqlite3") as graph:
        graph.replace_snapshot(snapshot)
        try:
            graph.callers("foo")
        except KeyError as exc:
            assert "Ambiguous" in str(exc)
        else:
            raise AssertionError("expected ambiguous symbol failure")


def test_graph_keeps_same_unit_name_for_multiple_targets(tmp_path: Path) -> None:
    root = tmp_path
    caller = "s:caller"
    callee = "s:callee"
    occurrences = (
        IndexStoreOccurrence(callee, "callee", 1, 12, ROLE_DEFINITION, str(root / "C.swift"), 1, 1, unit_name="same.o", target="arm64"),
        IndexStoreOccurrence(caller, "caller", 1, 12, ROLE_DEFINITION, str(root / "A.swift"), 1, 1, unit_name="same.o", target="x86_64"),
        IndexStoreOccurrence(
            callee,
            "callee",
            1,
            12,
            ROLE_CALL,
            str(root / "A.swift"),
            3,
            5,
            caller_usrs=(caller,),
            unit_name="same.o",
            target="x86_64",
            caller_relations=((caller, ROLE_REL_CALLEDBY),),
        ),
    )
    snapshot = IndexStoreReadResult(
        store_path=str(root / "DataStore"),
        root_path=str(root),
        occurrences=occurrences,
        unit_count=2,
        record_count=2,
        units=(
            IndexStoreUnit("same.o", str(root / "C.swift"), target="arm64", module_name="Arm"),
            IndexStoreUnit("same.o", str(root / "A.swift"), target="x86_64", module_name="X86"),
        ),
    )
    with IndexStoreGraph(tmp_path / "graph.sqlite3") as graph:
        graph.replace_snapshot(snapshot)
        result = graph.callers("callee")
        assert result[0]["evidence"]["module"] == "X86"
        assert graph.status().units == 2


def test_graph_migrates_target_aware_unit_identity_without_data_loss(tmp_path: Path) -> None:
    database_path = tmp_path / "graph.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE graph_units (
                unit_name TEXT PRIMARY KEY,
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
                is_system_unit INTEGER NOT NULL
            );
            INSERT INTO graph_units VALUES (
                'same.o', '/tmp/Legacy.swift', 'arm64-apple-macosx', 'Legacy',
                '', '', '', '', '', 0, 0, 0, 0
            );
            """
        )

    with IndexStoreGraph(database_path) as graph:
        assert graph.status().units == 1
        graph.replace_snapshot(
            IndexStoreReadResult(
                store_path=str(tmp_path / "DataStore"),
                root_path=str(tmp_path),
                occurrences=(),
                unit_count=2,
                record_count=0,
                units=(
                    IndexStoreUnit("same.o", str(tmp_path / "Arm.swift"), target="arm64-apple-macosx"),
                    IndexStoreUnit("same.o", str(tmp_path / "Intel.swift"), target="x86_64-apple-macosx"),
                ),
            )
        )
        assert graph.status().units == 2
