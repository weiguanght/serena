# Xcode IndexStore and Serena

Serena uses two complementary semantic tiers for Swift and Objective-C projects:

* SourceKit-LSP and clangd answer live, position-based requests and see files opened by the agent.
* The optional Xcode IndexStore graph reads the compiler snapshot produced by an Xcode build and persists USRs, definitions, occurrences, compilation units, targets and call-site evidence in SQLite.

The compiler snapshot is useful for whole-repository callers, callees, impact analysis and bounded traces. It is not a substitute for live LSP state: unsaved edits and code not present in the selected build's IndexStore are not represented.

## Configure

In a trusted project's `.serena/project.yml` or `project.local.yml`:

```yaml
language_servers: [swift, cpp]
ls_specific_settings:
  swift:
    use_xcode_toolchain: true
    index_store_path: "/Users/me/Library/Developer/Xcode/DerivedData/App-.../Index.noindex/DataStore"
    index_database_path: "/Users/me/Library/Developer/Xcode/DerivedData/App-.../Index.noindex/DB"
    index_prefix_map: ["/private/tmp=/tmp"]
    auto_discover_xcode_index_store: true
    index_graph_path: "cache/xcode-index.sqlite3"
  cpp:
    use_xcode_toolchain: true
```

`sourcekit-lsp -h` must show `-index-store-path`, `-index-db-path` and `-index-prefix-map`. Serena passes these options directly; it does not claim that every Xcode build configuration is present in one store.

Build the intended Xcode scheme once before refreshing the graph. Discovery searches Xcode DerivedData and prefers stores whose workspace metadata or project name matches the project root. For reproducible automation, pass the DataStore explicitly.

## CLI

The graph works without MCP:

```bash
serena xcode-index refresh /path/to/project --store /path/to/Index.noindex/DataStore
serena xcode-index status /path/to/project
serena xcode-index callers 'MyType.myMethod()' /path/to/project
serena xcode-index callees 'usr:s:...' /path/to/project
serena xcode-index impact 's:...' /path/to/project --direction callers --depth 3
serena xcode-index trace 's:entry' 's:callee' /path/to/project
```

The optional tools `xcode_index_status`, `xcode_index_refresh`,
`xcode_index_callers`, `xcode_index_callees`, `xcode_index_impact`,
`xcode_index_trace` and `live_call_hierarchy` expose the same capabilities to
an agent. Enable optional tools through the normal Serena mode/project tool
configuration when using MCP.

## Accuracy notes

IndexStore records are compiler-produced and preserve USRs, target identity,
line/UTF-8-column, dynamic dispatch and relation roles. Serena keeps separate
evidence rows for separate units and targets instead of collapsing them into a
single caller/callee pair. A dynamic call remains marked dynamic and should be
reviewed with live SourceKit-LSP/clangd when dispatch is ambiguous.
