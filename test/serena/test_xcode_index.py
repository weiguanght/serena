from pathlib import Path

from serena.config.serena_config import SerenaConfig
from serena.xcode_index import XcodeIndexService


def test_xcode_index_defaults_to_ignored_cache_and_resolves_settings(tmp_path: Path) -> None:
    service = XcodeIndexService.from_path(tmp_path, SerenaConfig())

    assert service.database_path == str(tmp_path / ".serena" / "cache" / "xcode-index.sqlite3")
    with service.graph():
        pass
    assert (tmp_path / ".serena" / ".gitignore").read_text() == "/cache\n"

    service.swift_settings["index_graph_path"] = "graphs/project.sqlite3"
    service.swift_settings["indexstore_library_path"] = "Toolchains/Xcode/libIndexStore.dylib"
    assert service.database_path == str(tmp_path / ".serena" / "graphs" / "project.sqlite3")
    assert service.library_path() == str(tmp_path / "Toolchains" / "Xcode" / "libIndexStore.dylib")
