"""Persistent Python semantic worker used by the native Serena daemon."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from collections.abc import Mapping
from typing import Any

from sensai.util import logging

from serena.agent import SerenaAgent
from serena.config.context_mode import SerenaAgentContext
from serena.config.serena_config import LanguageBackend, ModeSelectionDefinitionWithAddedModes, SerenaConfig
from serena.tools import ToolCallError, ToolRegistry


class _TerminationSignal(BaseException):
    """Raised in the main thread when the supervisor asks the worker to stop."""

    def __init__(self, signum: int) -> None:
        self.signum = signum


def main() -> int:
    """Run a newline-delimited request worker until shutdown."""
    parser = argparse.ArgumentParser(description="Serena persistent semantic worker")
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    agent: SerenaAgent | None = None
    shutdown_complete = False
    termination_signal: int | None = None
    _install_termination_handlers()
    try:
        try:
            config = json.loads(args.config_json)
            if not isinstance(config, Mapping):
                raise ValueError("worker configuration must be a JSON object")
            _configure_logging(config.get("log_level"))
            agent = _create_agent(config)
            _wait_for_agent_startup(agent)
        except Exception as exc:
            _write_json({"event": "fatal", "error": f"{exc.__class__.__name__}: {exc}"})
            traceback.print_exc(file=sys.stderr)
            return 1

        _write_json({"event": "ready", "pid": os.getpid(), "project": config.get("project")})
        for line in sys.stdin:
            if not line.strip():
                continue
            should_exit, shutdown_complete = _handle_request(agent, line, shutdown_complete)
            if should_exit:
                break
    except _TerminationSignal as exc:
        termination_signal = exc.signum
        _ignore_termination_signals()
    finally:
        if agent is not None and not shutdown_complete:
            try:
                agent.on_shutdown()
            except Exception:
                traceback.print_exc(file=sys.stderr)

    if termination_signal is not None:
        sys.stderr.flush()
        os._exit(128 + termination_signal)
    return 0


def _install_termination_handlers() -> None:
    """Turn supervisor termination signals into main-thread cleanup."""
    signal.signal(signal.SIGTERM, _raise_termination_signal)
    signal.signal(signal.SIGINT, _raise_termination_signal)


def _ignore_termination_signals() -> None:
    """Keep repeated termination signals from interrupting bounded cleanup."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _raise_termination_signal(signum: int, _frame: Any) -> None:
    """Interrupt the worker's main thread for graceful resource cleanup."""
    raise _TerminationSignal(signum)


def _configure_logging(log_level: Any) -> None:
    """Send worker logs to stderr and apply the requested level."""
    level = logging.INFO
    if isinstance(log_level, str):
        level = getattr(logging, log_level.upper(), logging.INFO)
    logging.configure(level=level, stream=sys.stderr)


def _create_agent(config: Mapping[str, Any]) -> SerenaAgent:
    """Create one headless agent whose language servers live for the worker lifetime."""
    serena_config = SerenaConfig.from_config_file().with_headless_mode_overrides()
    backend = config.get("language_backend")
    if backend:
        serena_config.language_backend = LanguageBackend.from_str(str(backend))
    tool_timeout = config.get("tool_timeout")
    if tool_timeout is not None:
        serena_config.tool_timeout = float(tool_timeout)
    _enable_cli_tools(serena_config, config.get("enabled_tools", []) or [])

    context_name = str(config.get("context") or "desktop-app")
    context = SerenaAgentContext.load(context_name)
    default_modes = tuple(str(mode) for mode in config.get("default_modes", []) or [])
    added_modes = tuple(str(mode) for mode in config.get("added_modes", []) or [])
    modes = None
    if default_modes or added_modes:
        modes = ModeSelectionDefinitionWithAddedModes(
            default_modes=default_modes or None,
            added_modes=added_modes or None,
        )
    return SerenaAgent(
        project=str(config["project"]),
        serena_config=serena_config,
        context=context,
        modes=modes,
    )


def _wait_for_agent_startup(agent: SerenaAgent) -> None:
    """Wait for the startup project's activation and language servers."""
    startup_tasks = agent.get_current_tasks()
    for task in startup_tasks:
        task.future.result()
    if agent.get_active_project() is None:
        raise RuntimeError("Serena did not activate the daemon project")


def _enable_cli_tools(serena_config: SerenaConfig, enabled_tools: Any) -> None:
    """Enable explicitly requested optional tools for the daemon session."""
    if not isinstance(enabled_tools, list):
        raise ValueError("enabled_tools must be a JSON array")
    registry = ToolRegistry()
    optional_names = set(registry.get_tool_names_optional())
    requested = []
    for name_value in enabled_tools:
        name = str(name_value)
        if name not in optional_names:
            raise ValueError(f"Tool '{name}' is not a registered optional Serena tool")
        if name not in requested:
            requested.append(name)
    if serena_config.fixed_tools:
        serena_config.fixed_tools = list(dict.fromkeys([*serena_config.fixed_tools, *requested]))
    else:
        serena_config.included_optional_tools = list(dict.fromkeys([*serena_config.included_optional_tools, *requested]))


def _handle_request(agent: SerenaAgent, line: str, shutdown_complete: bool) -> tuple[bool, bool]:
    """Dispatch one request and write one response; return whether to exit."""
    try:
        request = json.loads(line)
        if not isinstance(request, Mapping):
            raise ValueError("request must be a JSON object")
        request_id = request.get("id")
        method = request.get("method")
        if not isinstance(method, str):
            raise ValueError("request method must be a string")
        params = request.get("params") or {}
        if not isinstance(params, Mapping):
            raise ValueError("request params must be a JSON object")

        if method == "shutdown":
            if not shutdown_complete:
                agent.on_shutdown()
            _write_json({"id": request_id, "ok": True, "result": {"stopping": True}})
            return True, True
        result = _dispatch(agent, method, params)
        _write_json({"id": request_id, "ok": True, "result": result})
    except Exception as exc:
        _write_json(
            {
                "id": _safe_request_id(line),
                "ok": False,
                "error": {"code": exc.__class__.__name__, "message": str(exc)},
            }
        )
        traceback.print_exc(file=sys.stderr)
    return False, shutdown_complete


def _safe_request_id(line: str) -> Any:
    """Recover a request id after malformed dispatch input."""
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value.get("id") if isinstance(value, Mapping) else None


def _dispatch(agent: SerenaAgent, method: str, params: Mapping[str, Any]) -> Any:
    """Execute a protocol method against the persistent agent."""
    if method == "ping" or method == "status":
        active_project = agent.get_active_project()
        return {
            "worker_pid": os.getpid(),
            "project": str(active_project.project_root) if active_project is not None else None,
            "active_tools": agent.get_active_tool_names(),
            "language_backend": agent.get_language_backend().value,
            "language_servers": [language.value for language in agent.get_active_language_server_ids()],
            "tasks": [task.name for task in agent.get_current_tasks()],
        }
    if method == "tool.list":
        return _list_tools(agent)
    if method == "tool.describe":
        tool_name = _required_string(params, "tool")
        return _describe_tool(agent, tool_name)
    if method == "tool.call":
        tool_name = _required_string(params, "tool")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, Mapping):
            raise ValueError("tool arguments must be a JSON object")
        return _call_tool(agent, tool_name, dict(arguments))
    raise ValueError(f"unknown daemon method: {method}")


def _list_tools(agent: SerenaAgent) -> list[dict[str, Any]]:
    """Return all registered tools and their current availability."""
    result = []
    for tool_name in sorted(ToolRegistry().get_tool_names()):
        tool = agent.get_tool_by_name(tool_name)
        result.append(
            {
                "name": tool_name,
                "active": agent.tool_is_active(tool_name),
                "exposed": agent.tool_is_exposed(tool_name),
                "readonly": tool.is_readonly(),
                "symbolic": tool.is_symbolic(),
                "can_edit": tool.can_edit(),
            }
        )
    return result


def _describe_tool(agent: SerenaAgent, tool_name: str) -> dict[str, Any]:
    """Return a tool's description and validated argument schema."""
    tool = agent.get_tool_by_name(tool_name)
    metadata = tool.get_apply_fn_metadata(structured_output=False)
    return {
        "name": tool_name,
        "description": tool.get_apply_docstring(),
        "parameters": metadata.arg_model.model_json_schema(),
        "active": agent.tool_is_active(tool_name),
        "exposed": agent.tool_is_exposed(tool_name),
        "readonly": tool.is_readonly(),
        "symbolic": tool.is_symbolic(),
        "can_edit": tool.can_edit(),
        "aliases": tool.get_param_aliases(),
    }


def _call_tool(agent: SerenaAgent, tool_name: str, arguments: dict[str, Any]) -> str:
    """Validate and execute one tool serially through Serena's task executor."""
    tool = agent.get_tool_by_name(tool_name)
    for alias, parameter in tool.get_param_aliases().items():
        if alias in arguments and parameter not in arguments:
            arguments[parameter] = arguments.pop(alias)

    metadata = tool.get_apply_fn_metadata(structured_output=False)
    parsed_arguments = metadata.pre_parse_json(arguments)
    model = metadata.arg_model.model_validate(parsed_arguments)
    validated_arguments = model.model_dump_one_level()
    try:
        return tool.apply_ex(log_call=True, catch_exceptions=False, **validated_arguments)
    except ToolCallError:
        raise


def _required_string(params: Mapping[str, Any], name: str) -> str:
    """Read a required non-empty string protocol parameter."""
    value = params.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _write_json(value: Any) -> None:
    """Write one protocol line without contaminating stdout with logs."""
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
