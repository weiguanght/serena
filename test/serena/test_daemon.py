"""Tests for the native daemon lifecycle and protocol helpers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from serena.cli import DaemonCommands, ToolCommands
from serena.daemon import DaemonError, daemon_paths, daemon_status, response_result, stop_daemon


def test_daemon_paths_are_deterministic_and_keep_socket_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERENA_DAEMON_HOME", str(tmp_path / "daemons"))
    project = tmp_path / "project"
    project.mkdir()

    first = daemon_paths(project)
    second = daemon_paths(project)

    assert first == second
    assert first.project_root == project.resolve()
    assert first.socket_path.name.startswith("serena-")
    assert first.socket_path.suffix == ".sock"
    assert len(str(first.socket_path)) < 104
    assert first.lock_path == first.state_dir / "start.lock"


def test_daemon_status_without_state_is_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERENA_DAEMON_HOME", str(tmp_path / "daemons"))
    project = tmp_path / "project"
    project.mkdir()

    status = daemon_status(daemon_paths(project))

    assert status["running"] is False
    assert status["state"] == "stopped"


def test_daemon_status_reports_busy_when_supervised_process_is_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERENA_DAEMON_HOME", str(tmp_path / "daemons"))
    socket_dir = tmp_path / "sockets"
    socket_dir.mkdir()
    monkeypatch.setenv("SERENA_DAEMON_SOCKET_DIR", str(socket_dir))
    project = tmp_path / "project"
    project.mkdir()
    paths = daemon_paths(project)
    paths.state_dir.mkdir(parents=True)
    paths.socket_path.touch()
    paths.pid_path.write_text("123\n", encoding="utf-8")
    paths.metadata_path.write_text(
        json.dumps(
            {
                "pid": 123,
                "worker_pid": 456,
                "project": str(project),
                "socket": str(paths.socket_path),
                "started_at_unix_ms": 789,
            }
        ),
        encoding="utf-8",
    )

    def fake_request(*_args: object, **_kwargs: object) -> dict:
        raise DaemonError("timed out")

    monkeypatch.setattr("serena.daemon.request", fake_request)
    monkeypatch.setattr("serena.daemon._pid_exists", lambda pid: pid == 123)
    monkeypatch.setattr("serena.daemon._pid_matches_daemon", lambda pid, daemon_paths: pid == 123 and daemon_paths == paths)

    status = daemon_status(paths)

    assert status == {
        "running": True,
        "state": "busy",
        "responsive": False,
        "pid": 123,
        "daemon_pid": 123,
        "worker_pid": 456,
        "project": str(project),
        "socket": str(paths.socket_path),
        "started_at_unix_ms": 789,
        "error": "timed out",
    }


def test_response_result_returns_value_or_raises() -> None:
    assert response_result({"ok": True, "result": {"value": 1}}) == {"value": 1}

    with pytest.raises(DaemonError, match="bad tool"):
        response_result({"ok": False, "error": {"message": "bad tool"}})


def test_daemon_status_reports_starting_when_supervisor_has_no_worker_yet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERENA_DAEMON_HOME", str(tmp_path / "daemons"))
    socket_dir = tmp_path / "sockets"
    socket_dir.mkdir()
    monkeypatch.setenv("SERENA_DAEMON_SOCKET_DIR", str(socket_dir))
    project = tmp_path / "project"
    project.mkdir()
    paths = daemon_paths(project)
    paths.state_dir.mkdir(parents=True)
    paths.socket_path.touch()
    paths.pid_path.write_text("123\n", encoding="utf-8")
    paths.metadata_path.write_text(json.dumps({"pid": 123, "project": str(project)}), encoding="utf-8")

    monkeypatch.setattr("serena.daemon.request", lambda *_args, **_kwargs: (_ for _ in ()).throw(DaemonError("starting")))
    monkeypatch.setattr("serena.daemon._pid_exists", lambda pid: pid == 123)
    monkeypatch.setattr("serena.daemon._pid_matches_daemon", lambda pid, daemon_paths: pid == 123 and daemon_paths == paths)

    status = daemon_status(paths)

    assert status["running"] is True
    assert status["state"] == "starting"
    assert status["responsive"] is False
    assert status["worker_pid"] is None


def test_wait_for_daemon_does_not_accept_unresponsive_starting_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from serena.daemon import wait_for_daemon

    monkeypatch.setenv("SERENA_DAEMON_HOME", str(tmp_path / "daemons"))
    project = tmp_path / "project"
    project.mkdir()
    paths = daemon_paths(project)
    monkeypatch.setattr(
        "serena.daemon.daemon_status", lambda *_args, **_kwargs: {"running": True, "responsive": False, "state": "starting"}
    )

    with pytest.raises(DaemonError, match="did not become ready"):
        wait_for_daemon(paths, timeout=0.1)


def test_stop_daemon_escalates_to_verified_processes_when_request_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERENA_DAEMON_HOME", str(tmp_path / "daemons"))
    socket_dir = tmp_path / "sockets"
    socket_dir.mkdir()
    monkeypatch.setenv("SERENA_DAEMON_SOCKET_DIR", str(socket_dir))
    project = tmp_path / "project"
    project.mkdir()
    paths = daemon_paths(project)
    paths.state_dir.mkdir(parents=True)
    paths.socket_path.touch()
    paths.pid_path.write_text("123\n", encoding="utf-8")
    paths.metadata_path.write_text(
        json.dumps({"pid": 123, "worker_pid": 456, "project": str(project), "socket": str(paths.socket_path)}),
        encoding="utf-8",
    )
    signals: list[tuple[int, int]] = []
    alive = {123, 456}

    monkeypatch.setattr("serena.daemon.request", lambda *_args, **_kwargs: (_ for _ in ()).throw(DaemonError("busy")))
    monkeypatch.setattr("serena.daemon._pid_exists", lambda pid: pid in alive)
    monkeypatch.setattr("serena.daemon._pid_matches_daemon", lambda pid, daemon_paths: pid == 123 and daemon_paths == paths)
    monkeypatch.setattr("serena.daemon._pid_matches_worker", lambda pid, daemon_paths: pid == 456 and daemon_paths == paths)
    monkeypatch.setattr("serena.daemon._wait_for_pid_exit", lambda *_args: None)

    def fake_kill(pid: int, sig: int) -> None:
        signals.append((pid, sig))
        alive.discard(pid)

    fake_os = type(
        "OS",
        (),
        {
            "getpgid": staticmethod(lambda _pid: 999),
            "kill": staticmethod(fake_kill),
            "killpg": staticmethod(lambda _pgid, _sig: None),
        },
    )()
    monkeypatch.setattr("serena.daemon.os", fake_os)
    monkeypatch.setattr("serena.daemon.cleanup_stale_state", lambda _paths: None)

    result = stop_daemon(paths, timeout=0)

    assert result["stopped"] is True
    assert result["escalated"] is True
    assert signals


def test_protocol_payload_is_json_object() -> None:
    payload = {"id": 1, "method": "tool.call", "params": {"tool": "read_file", "arguments": {}}}
    assert json.loads(json.dumps(payload)) == payload


def test_worker_status_serializes_registered_backend_keys() -> None:
    from serena.daemon_worker import _dispatch

    class RegisteredId:
        def __init__(self, key: str) -> None:
            self.key = key

        def get_key(self) -> str:
            return self.key

    agent = SimpleNamespace(
        get_active_project=lambda: SimpleNamespace(project_root=Path("/project")),
        get_active_tool_names=lambda: ["find_symbol"],
        get_language_backend=lambda: RegisteredId("custom-backend"),
        get_active_language_server_ids=lambda: [RegisteredId("custom-language")],
        get_current_tasks=lambda: [SimpleNamespace(name="ready")],
    )

    status = _dispatch(agent, "status", {})

    assert status["language_backend"] == "custom-backend"
    assert status["language_servers"] == ["custom-language"]


def test_tools_call_prints_raw_tool_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("serena.daemon.daemon_paths", lambda project: project)

    def fake_request(paths: object, method: str, params: dict, *, timeout: float) -> dict:
        captured.update(paths=paths, method=method, params=params, timeout=timeout)
        return {"ok": True, "result": '[{"name_path":"User"}]'}

    monkeypatch.setattr("serena.daemon.request", fake_request)

    result = CliRunner().invoke(
        ToolCommands.call,
        [
            "find_symbol",
            str(tmp_path),
            "--arguments",
            '{"name_path_pattern":"User"}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == '[{"name_path":"User"}]'
    assert captured["method"] == "tool.call"
    assert captured["params"] == {"tool": "find_symbol", "arguments": {"name_path_pattern": "User"}}


def test_tools_call_rejects_non_object_arguments(tmp_path: Path) -> None:
    result = CliRunner().invoke(ToolCommands.call, ["find_symbol", str(tmp_path), "--arguments", "[]"])

    assert result.exit_code == 2
    assert "must be a JSON object" in result.output


def test_daemon_start_reports_reused_daemon_when_start_races(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    status = {"running": True, "state": "running", "responsive": True, "daemon_pid": 42}

    monkeypatch.setattr("serena.daemon.daemon_paths", lambda _project: object())
    monkeypatch.setattr("serena.daemon.daemon_status", lambda *_args, **_kwargs: status)
    monkeypatch.setattr("serena.daemon.start_native_daemon", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(DaemonCommands.start, [str(project), "--no-build"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["started"] is False
    assert payload["already_running"] is True
    assert payload["daemon_pid"] == 42
