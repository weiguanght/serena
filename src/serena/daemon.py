"""Lifecycle and IPC helpers for the native Serena daemon."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on Windows
    fcntl = None  # type: ignore[assignment]
    import msvcrt

PROTOCOL_VERSION = 1
DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_STARTUP_TIMEOUT = 120.0
DEFAULT_DAEMON_OPTIONAL_TOOLS = (
    "get_diagnostics_for_symbol",
    "live_call_hierarchy",
    "restart_language_server",
    "xcode_index_callees",
    "xcode_index_callers",
    "xcode_index_impact",
    "xcode_index_refresh",
    "xcode_index_status",
    "xcode_index_trace",
)


@dataclass(frozen=True)
class DaemonPaths:
    """Paths for one canonical project daemon."""

    project_root: Path
    state_dir: Path
    socket_path: Path
    pid_path: Path
    metadata_path: Path
    log_path: Path
    lock_path: Path


class DaemonError(RuntimeError):
    """Raised when a daemon operation cannot be completed."""


def canonical_project(project: str | Path) -> Path:
    """Return an existing project directory as an absolute path."""
    path = Path(project).expanduser().resolve()
    if not path.exists():
        raise DaemonError(f"Project does not exist: {path}")
    if not path.is_dir():
        raise DaemonError(f"Project is not a directory: {path}")
    return path


def daemon_root() -> Path:
    """Return the root directory containing per-project daemon state."""
    configured = os.environ.get("SERENA_DAEMON_HOME")
    if configured:
        return Path(configured).expanduser().resolve()

    serena_home = os.environ.get("SERENA_HOME")
    if serena_home:
        return Path(serena_home).expanduser().resolve() / "daemons"
    return Path.home() / ".cache" / "serena" / "daemons"


def daemon_paths(project: str | Path) -> DaemonPaths:
    """Return deterministic, short state paths for a project."""
    project_root = canonical_project(project)
    digest = hashlib.sha256(str(project_root).encode("utf-8")).hexdigest()[:32]
    state_dir = daemon_root() / digest
    socket_dir = Path(os.environ.get("SERENA_DAEMON_SOCKET_DIR", tempfile.gettempdir())).expanduser().resolve()
    return DaemonPaths(
        project_root=project_root,
        state_dir=state_dir,
        socket_path=socket_dir / f"serena-{digest}.sock",
        pid_path=state_dir / "daemon.pid",
        metadata_path=state_dir / "metadata.json",
        log_path=state_dir / "daemon.log",
        lock_path=state_dir / "start.lock",
    )


def native_source_root() -> Path:
    """Return the Serena source root when running from a checkout."""
    return Path(__file__).resolve().parents[2]


def native_manifest_path() -> Path:
    """Return the Cargo manifest for the native daemon."""
    return native_source_root() / "native" / "serena-native" / "Cargo.toml"


def native_binary_candidates() -> list[Path]:
    """Return native daemon binary candidates in preference order."""
    candidates: list[Path] = []
    configured = os.environ.get("SERENA_NATIVE_BINARY")
    if configured:
        candidates.append(Path(configured).expanduser())

    source_native = native_source_root() / "native" / "serena-native"
    candidates.extend(
        [
            source_native / "target" / "release" / "serena-native",
            source_native / "target" / "debug" / "serena-native",
        ]
    )

    executable_dir = Path(sys.executable).resolve().parent
    candidates.append(executable_dir / "serena-native")
    path_binary = shutil.which("serena-native")
    if path_binary:
        candidates.append(Path(path_binary))

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def find_native_binary() -> Path | None:
    """Find an executable native daemon binary."""
    for candidate in native_binary_candidates():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def build_native_binary(*, release: bool = True) -> Path:
    """Build and return the Rust daemon binary."""
    manifest = native_manifest_path()
    if not manifest.is_file():
        raise DaemonError(f"Native daemon source is missing: {manifest}")

    command = ["cargo", "build", "--manifest-path", str(manifest)]
    if release:
        command.append("--release")
    try:
        subprocess.run(command, cwd=manifest.parent, check=True)
    except FileNotFoundError as exc:
        raise DaemonError("cargo is required to build Serena's native daemon") from exc
    except subprocess.CalledProcessError as exc:
        raise DaemonError(f"native daemon build failed with exit code {exc.returncode}") from exc

    binary = manifest.parent / "target" / ("release" if release else "debug") / "serena-native"
    if not binary.is_file():
        raise DaemonError(f"cargo completed but did not produce {binary}")
    return binary


def ensure_native_binary(*, auto_build: bool = True) -> Path:
    """Find the native binary, optionally building a source checkout."""
    binary = find_native_binary()
    if binary is not None:
        return binary
    if not auto_build or os.environ.get("SERENA_NATIVE_AUTO_BUILD") == "0":
        raise DaemonError("serena-native was not found. Run `serena daemon build` or set SERENA_NATIVE_BINARY to its path.")
    return build_native_binary()


def worker_config(
    *,
    project: Path,
    context: str,
    default_modes: tuple[str, ...],
    added_modes: tuple[str, ...],
    language_backend: str | None,
    log_level: str | None,
    tool_timeout: float | None,
    enabled_tools: tuple[str, ...],
) -> dict[str, Any]:
    """Build the JSON configuration passed to the persistent Python worker."""
    return {
        "project": str(project),
        "context": context,
        "default_modes": list(default_modes),
        "added_modes": list(added_modes),
        "language_backend": language_backend,
        "log_level": log_level,
        "tool_timeout": tool_timeout,
        "enabled_tools": list(enabled_tools),
    }


def start_native_daemon(
    paths: DaemonPaths,
    *,
    context: str = "desktop-app",
    default_modes: tuple[str, ...] = (),
    added_modes: tuple[str, ...] = (),
    language_backend: str | None = None,
    log_level: str | None = None,
    tool_timeout: float | None = None,
    enabled_tools: tuple[str, ...] = DEFAULT_DAEMON_OPTIONAL_TOOLS,
    foreground: bool = False,
    auto_build: bool = True,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
) -> subprocess.Popen[bytes] | None:
    """Start a native daemon and wait until its socket answers ping."""
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_start_lock(paths):
        existing = daemon_status(paths, timeout=0.5)
        if existing.get("running"):
            return None
        cleanup_stale_state(paths)

        binary = ensure_native_binary(auto_build=auto_build)
        config_json = json.dumps(
            worker_config(
                project=paths.project_root,
                context=context,
                default_modes=default_modes,
                added_modes=added_modes,
                language_backend=language_backend,
                log_level=log_level,
                tool_timeout=tool_timeout,
                enabled_tools=enabled_tools,
            ),
            separators=(",", ":"),
        )
        command = [
            str(binary),
            "serve",
            "--project",
            str(paths.project_root),
            "--socket",
            str(paths.socket_path),
            "--pid-file",
            str(paths.pid_path),
            "--metadata-file",
            str(paths.metadata_path),
            "--python",
            sys.executable,
            "--worker-module",
            "serena.daemon_worker",
            "--worker-config-json",
            config_json,
        ]

        log_file = paths.log_path.open("ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(native_source_root()),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL if not foreground else None,
                stderr=log_file,
                start_new_session=not foreground,
                close_fds=True,
            )
        finally:
            log_file.close()

        try:
            wait_for_daemon(paths, timeout=startup_timeout)
        except Exception:
            _terminate_spawned_daemon(process, process_group=not foreground)
            cleanup_stale_state(paths)
            raise
        return process


@contextmanager
def _exclusive_start_lock(paths: DaemonPaths) -> Iterator[None]:
    """Serialize start attempts for one project across CLI processes."""
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    with paths.lock_path.open("a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        else:
            lock_file.seek(0)
            lock_file.write(b"0")
            lock_file.flush()
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            else:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _terminate_spawned_daemon(process: subprocess.Popen[bytes], *, process_group: bool, timeout: float = 5.0) -> None:
    """Terminate a daemon process created by this Python process."""
    if process.poll() is not None:
        return

    def send(sig: signal.Signals) -> None:
        try:
            if process_group:
                os.killpg(process.pid, sig)
            else:
                os.kill(process.pid, sig)
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL)
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def request(
    paths: DaemonPaths, method: str, params: dict[str, Any] | None = None, *, timeout: float = DEFAULT_REQUEST_TIMEOUT
) -> dict[str, Any]:
    """Send one JSON request to a daemon Unix socket."""
    payload = {
        "id": f"cli-{os.getpid()}-{time.monotonic_ns()}",
        "method": method,
        "params": params or {},
    }
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(paths.socket_path))
            client.sendall((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
            response = _read_line(client)
    except (OSError, TimeoutError) as exc:
        raise DaemonError(f"could not contact daemon for {paths.project_root}: {exc}") from exc

    try:
        decoded = json.loads(response)
    except json.JSONDecodeError as exc:
        raise DaemonError(f"daemon returned invalid JSON: {response!r}") from exc
    if not isinstance(decoded, dict):
        raise DaemonError("daemon response must be a JSON object")
    return decoded


def _read_line(client: socket.socket) -> str:
    """Read one newline-delimited response from a Unix socket."""
    chunks: list[bytes] = []
    while True:
        chunk = client.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    if not chunks:
        raise DaemonError("daemon closed the connection without a response")
    return b"".join(chunks).split(b"\n", 1)[0].decode("utf-8")


def daemon_status(paths: DaemonPaths, *, timeout: float = 1.0) -> dict[str, Any]:
    """Return daemon status without raising for an absent daemon."""
    if not paths.socket_path.exists():
        pid = read_pid(paths)
        if pid is not None and _pid_exists(pid) and _pid_matches_daemon(pid, paths):
            return _unresponsive_status(paths, pid, state="unresponsive", error="daemon socket is missing")
        return {"running": False, "state": "stopped", "pid": pid}
    try:
        response = request(paths, "status", timeout=timeout)
    except DaemonError as exc:
        pid = read_pid(paths)
        if pid is not None and _pid_exists(pid) and _pid_matches_daemon(pid, paths):
            metadata = read_metadata(paths)
            state = "busy" if metadata.get("worker_pid") is not None else "starting"
            return _unresponsive_status(paths, pid, state=state, error=str(exc), metadata=metadata)
        return {"running": False, "state": "stale", "pid": pid, "error": str(exc)}
    if response.get("ok"):
        result = response.get("result")
        if isinstance(result, dict):
            return {"running": True, "state": "running", "responsive": True, **result}
        return {"running": True, "state": "running", "responsive": True, "result": result}
    return {"running": False, "state": "unhealthy", "error": response.get("error")}


def _unresponsive_status(
    paths: DaemonPaths,
    pid: int,
    *,
    state: str,
    error: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build status for a live supervisor that cannot answer requests."""
    metadata = metadata or read_metadata(paths)
    return {
        "running": True,
        "state": state,
        "responsive": False,
        "pid": pid,
        "daemon_pid": metadata.get("pid", pid),
        "worker_pid": metadata.get("worker_pid"),
        "project": metadata.get("project", str(paths.project_root)),
        "socket": metadata.get("socket", str(paths.socket_path)),
        "started_at_unix_ms": metadata.get("started_at_unix_ms"),
        "error": error,
    }


def wait_for_daemon(paths: DaemonPaths, *, timeout: float = 30.0) -> dict[str, Any]:
    """Wait until the daemon socket responds to a status request."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        status = daemon_status(paths, timeout=min(1.0, max(0.1, deadline - time.monotonic())))
        if status.get("running") and status.get("responsive"):
            return status
        last_error = DaemonError(str(status.get("error", status.get("state", "not ready"))))
        time.sleep(0.05)
    raise DaemonError(f"daemon did not become ready within {timeout:g}s: {last_error}")


def read_pid(paths: DaemonPaths) -> int | None:
    """Read a daemon PID file if it contains a valid PID."""
    try:
        pid = int(paths.pid_path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        return None
    return pid if pid > 0 else None


def read_metadata(paths: DaemonPaths) -> dict[str, Any]:
    """Read supervisor metadata, returning an empty mapping when unavailable."""
    try:
        value = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def cleanup_stale_state(paths: DaemonPaths) -> None:
    """Remove state left by a daemon that is no longer reachable."""
    status = daemon_status(paths, timeout=0.2)
    if status.get("running"):
        return
    for path in (paths.socket_path, paths.pid_path, paths.metadata_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def stop_daemon(paths: DaemonPaths, *, timeout: float = 10.0) -> dict[str, Any]:
    """Gracefully stop a daemon, escalating only if it does not respond."""
    if timeout < 0:
        raise DaemonError("daemon stop timeout must be non-negative")

    status_before = daemon_status(paths, timeout=0.5)
    if not status_before.get("running"):
        cleanup_stale_state(paths)
        return {"stopped": True, "already_stopped": True, "project": str(paths.project_root)}

    graceful_error: str | None = None
    try:
        response = request(paths, "shutdown", timeout=min(max(timeout, 0.1), 1.0, DEFAULT_REQUEST_TIMEOUT))
        if not response.get("ok"):
            graceful_error = str(response.get("error"))
    except DaemonError as exc:
        graceful_error = str(exc)

    pid = read_pid(paths) or _positive_pid(status_before.get("daemon_pid"))
    worker_pid = _positive_pid(status_before.get("worker_pid"))
    escalated = False
    if pid is not None:
        if graceful_error is None:
            _wait_for_pid_exit(pid, time.monotonic() + timeout)
        if _pid_exists(pid) and _pid_matches_daemon(pid, paths):
            escalated = True
            _signal_daemon_processes(paths, pid, worker_pid, signal.SIGTERM)
            _wait_for_pid_exit(pid, time.monotonic() + min(max(timeout, 1.0), 5.0))
        if _pid_exists(pid) and _pid_matches_daemon(pid, paths):
            _signal_daemon_processes(paths, pid, worker_pid, signal.SIGKILL)
            _wait_for_pid_exit(pid, time.monotonic() + 2.0)

    if pid is not None and _pid_exists(pid) and _pid_matches_daemon(pid, paths):
        raise DaemonError(f"daemon for {paths.project_root} did not stop")

    cleanup_stale_state(paths)
    result: dict[str, Any] = {"stopped": True, "already_stopped": False, "project": str(paths.project_root)}
    if escalated:
        result["escalated"] = True
    if graceful_error:
        result["graceful_error"] = graceful_error
    return result


def _positive_pid(value: Any) -> int | None:
    """Return a positive process identifier from an arbitrary value."""
    return value if isinstance(value, int) and value > 0 else None


def _wait_for_pid_exit(pid: int, deadline: float) -> None:
    """Wait until a process exits or the monotonic deadline is reached."""
    while time.monotonic() < deadline and _pid_exists(pid):
        time.sleep(0.05)


def _signal_daemon_processes(
    paths: DaemonPaths,
    daemon_pid: int,
    worker_pid: int | None,
    sig: signal.Signals,
) -> None:
    """Signal a verified daemon and its worker without touching the caller's process group."""
    if not _pid_matches_daemon(daemon_pid, paths):
        return

    try:
        daemon_group = os.getpgid(daemon_pid)
    except OSError:
        daemon_group = None
    if daemon_group is not None and daemon_group == daemon_pid:
        try:
            os.killpg(daemon_group, sig)
        except OSError:
            pass
        return

    if worker_pid is not None and _pid_exists(worker_pid) and _pid_matches_worker(worker_pid, paths):
        try:
            os.kill(worker_pid, sig)
        except OSError:
            pass
    try:
        os.kill(daemon_pid, sig)
    except OSError:
        pass


def _pid_exists(pid: int) -> bool:
    """Return whether a process exists, without requiring it to be ours."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_matches_daemon(pid: int, paths: DaemonPaths) -> bool:
    """Return whether a PID appears to be this native daemon before signaling it."""
    try:
        command = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "serena-native" in command and str(paths.socket_path) in command


def _pid_matches_worker(pid: int, paths: DaemonPaths) -> bool:
    """Return whether a PID appears to be this daemon's Python worker."""
    try:
        command = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "serena.daemon_worker" in command and str(paths.project_root) in command


def response_result(response: dict[str, Any]) -> Any:
    """Extract a successful result or raise a useful daemon error."""
    if response.get("ok"):
        return response.get("result")
    error = response.get("error")
    if isinstance(error, dict):
        message = error.get("message", str(error))
    else:
        message = str(error)
    raise DaemonError(message)
