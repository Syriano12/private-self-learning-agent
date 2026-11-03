"""Bounded, snapshot-bound Python call-graph candidates from Phase 2 AST records.

This module never imports or executes repository source. A call expression is an
observed syntax fact; any resolved callee is only a static candidate, not proof of
runtime dispatch or reachability.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Any

from private_agent.tools.repository_snapshot import CodeEvidenceRef, RepositorySnapshot

SCHEMA_VERSION = 1
PARSER_ID = "python.call_graph"
PARSER_VERSION = "1"
MAX_NODES = 96
MAX_EDGES = 128
MAX_REFERENCES = 160


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def evidence_key(ref: CodeEvidenceRef) -> str:
    """Stable within a source revision, including when BugInvestigator merges snapshots."""
    value = ref.to_dict()
    value.pop("snapshot_id", None)
    return "ev:" + hashlib.sha256(_canonical(value)).hexdigest()[:32]


class EvidenceRegistry:
    def __init__(self, snapshot: RepositorySnapshot) -> None:
        self.snapshot = snapshot
        self._items: dict[str, dict[str, Any]] = {}

    def add_record(self, record: dict[str, Any]) -> str:
        raw = record.get("evidence_ref")
        if not isinstance(raw, dict):
            raise ValueError("phase2_evidence_ref_missing")
        ref = CodeEvidenceRef.from_dict(raw, self.snapshot)
        return self.add_ref(ref)

    def add_location(
        self, *, path: str, line_start: int, line_end: int, column_start: int | None,
        column_end: int | None, symbol_id: str | None, rule_id: str,
        evidence_type: str = "static_data_flow", truncated: bool = False,
        uncertainty: tuple[str, ...] = (),
    ) -> str:
        ref = CodeEvidenceRef.create(
            self.snapshot, path=path, line_start=line_start, line_end=line_end,
            column_start=column_start, column_end=column_end, symbol_id=symbol_id,
            parser_id=PARSER_ID, parser_version=PARSER_VERSION, rule_id=rule_id,
            rule_version="1", evidence_type=evidence_type, truncated=truncated,
            uncertainty=uncertainty,
        )
        return self.add_ref(ref)

    def add_ref(self, ref: CodeEvidenceRef) -> str:
        ref.validate_against(self.snapshot)
        key = evidence_key(ref)
        rendered = ref.to_dict()
        previous = self._items.get(key)
        if previous is not None and previous != rendered:
            raise ValueError("evidence_key_collision")
        if previous is None and len(self._items) >= MAX_REFERENCES:
            raise ValueError("analysis_evidence_limit")
        self._items[key] = rendered
        return key

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "references": [
                {"evidence_id": key, "evidence_ref": self._items[key]}
                for key in sorted(self._items)
            ],
        }


def _callable(kind: str) -> bool:
    return kind in {"function", "async_function", "method", "async_method", "lambda"}


def _module_path(module_name: str, path_index: dict[str, dict[str, Any]]) -> str | None:
    normalized = module_name.strip(".")
    if not normalized:
        return None
    exact = [path for path, item in path_index.items() if item.get("module_name") == normalized]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    suffix = [
        path for path, item in path_index.items()
        if str(item.get("module_name", "")).endswith("." + normalized)
    ]
    return suffix[0] if len(suffix) == 1 else None


def _absolute_import_module(path: str, module: str, level: int) -> str | None:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        package = parts[:-1]
    else:
        package = parts[:-1]
    if level > len(package) + 1:
        return None
    if level:
        package = package[: len(package) - level + 1]
    suffix = [item for item in (module or "").split(".") if item]
    return ".".join((*package, *suffix))


def build_call_graph(
    python_ast: dict[str, Any], snapshot: RepositorySnapshot,
    registry: EvidenceRegistry | None = None,
) -> tuple[dict[str, Any], EvidenceRegistry]:
    """Return bounded graph records and their shared, validated EvidenceRef registry."""
    catalog = registry or EvidenceRegistry(snapshot)
    files = [item for item in python_ast.get("files", []) if isinstance(item, dict)]
    path_index = {str(item.get("file_path", "")): item for item in files if item.get("file_path")}
    symbol_index: dict[str, dict[str, Any]] = {}
    symbols_by_path: dict[str, list[dict[str, Any]]] = {}
    imports_by_path: dict[str, list[dict[str, Any]]] = {}
    calls: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    truncated = bool(python_ast.get("truncated", False))
    limitations = [
        "Call targets are static candidates only; runtime dispatch, monkey-patching and reflection are not proven.",
        "Only Phase 2 Python files selected within repository limits are represented.",
    ]

    def add_node(node: dict[str, Any]) -> bool:
        nonlocal truncated
        if node["node_id"] in node_ids:
            return True
        if len(nodes) >= MAX_NODES:
            truncated = True
            return False
        node_ids.add(node["node_id"])
        nodes.append(node)
        return True

    def add_edge(edge: dict[str, Any]) -> None:
        nonlocal truncated
        if len(edges) >= MAX_EDGES:
            truncated = True
            return
        edges.append(edge)

    def add_record(record: dict[str, Any]) -> str | None:
        nonlocal truncated
        try:
            return catalog.add_record(record)
        except ValueError as exc:
            if str(exc) == "analysis_evidence_limit":
                truncated = True
                return None
            raise

    for file in files:
        path = str(file["file_path"])
        if file.get("status") not in {"parsed", "parsed_truncated", "empty_source"}:
            if file.get("status") not in {"empty_source"}:
                limitations.append(f"Call graph unavailable for {path}: {str(file.get('reason', file.get('status', 'unknown')))[:96]}.")
            continue
        file_symbols = [item for item in file.get("symbols", []) if isinstance(item, dict)]
        symbols_by_path[path] = file_symbols
        for symbol in file_symbols:
            symbol_index[str(symbol.get("symbol_id", ""))] = symbol
        imports_by_path[path] = [item for item in file.get("imports", []) if isinstance(item, dict)]
        for symbol in file_symbols:
            symbol_id = str(symbol.get("symbol_id", ""))
            if not symbol_id:
                truncated = True
                continue
            evidence_id = add_record(symbol)
            if evidence_id is None:
                continue
            evidence_ids = [evidence_id]
            add_node({
                "node_id": symbol_id, "kind": str(symbol.get("kind", "symbol")),
                "name": str(symbol.get("name", ""))[:256],
                "qualname": str(symbol.get("qualname", ""))[:512],
                "path": path, "line_start": symbol.get("line_start"),
                "line_end": symbol.get("line_end"), "evidence_ids": evidence_ids,
            })
        for call in file.get("calls", []):
            if isinstance(call, dict):
                calls.append(call)

    def nearest_caller(call: dict[str, Any]) -> str | None:
        current = call.get("parent_id")
        visited: set[str] = set()
        while isinstance(current, str) and current and current not in visited:
            visited.add(current)
            record = symbol_index.get(current)
            if record is None:
                return None
            if _callable(str(record.get("kind", ""))) or record.get("kind") == "module":
                return current
            current = record.get("parent_id")
        return None

    def class_owner(call: dict[str, Any]) -> dict[str, Any] | None:
        current = call.get("parent_id")
        visited: set[str] = set()
        while isinstance(current, str) and current and current not in visited:
            visited.add(current)
            record = symbol_index.get(current)
            if record is None:
                return None
            if record.get("kind") == "class":
                return record
            current = record.get("parent_id")
        return None

    def import_candidates(path: str, target: str) -> list[tuple[str, dict[str, Any], str]]:
        first, dot, suffix = target.partition(".")
        matches: list[tuple[str, dict[str, Any], str]] = []
        for record in imports_by_path.get(path, []):
            module = str(record.get("module", ""))
            level = int(record.get("level", 0))
            for alias in record.get("aliases", []):
                if not isinstance(alias, dict):
                    continue
                alias_name = str(alias.get("name") or "")
                alias_asname = str(alias.get("asname") or "")
                bound = alias_asname or (alias_name if record.get("kind") == "import_from" else alias_name.split(".", 1)[0])
                if record.get("kind") == "import_from" and bound == first:
                    absolute = _absolute_import_module(path, module, level)
                    imported_name = alias_name
                    local_path = _module_path(f"{absolute}.{imported_name}" if absolute else imported_name, path_index)
                    if local_path is None:
                        local_path = _module_path(absolute or "", path_index)
                    if local_path:
                        target_name = f"{absolute}.{imported_name}" if absolute else imported_name
                        if dot:
                            target_name = f"{target_name}.{suffix}"
                        matches.append((local_path, record, target_name))
                    else:
                        external = f"{absolute or module or '<relative>'}.{imported_name}"
                        if dot:
                            external = f"{external}.{suffix}"
                        matches.append(("", record, external[:512]))
                elif record.get("kind") == "import" and bound == first:
                    absolute = alias_name
                    target_parts = target.split(".")
                    alias_parts = absolute.split(".") if absolute else []
                    if alias_asname:
                        target_suffix = target_parts[1:]
                    elif target_parts[:len(alias_parts)] == alias_parts:
                        target_suffix = target_parts[len(alias_parts):]
                    else:
                        continue
                    local_path = _module_path(absolute, path_index)
                    imported_target = ".".join(target_suffix)
                    if local_path:
                        matches.append((local_path, record, imported_target))
                    else:
                        external_name = absolute + ("." + imported_target if imported_target else "")
                        matches.append(("", record, external_name[:512]))
        return matches

    for call in calls:
        path = str(call.get("path", ""))
        call_id = str(call.get("symbol_id", ""))
        target = str(call.get("target", "<dynamic>"))[:256]
        call_ref = add_record(call)
        if call_ref is None:
            continue
        callsite_id = "callsite:" + call_id
        call_node = {
            "node_id": callsite_id, "kind": "call_site", "name": target,
            "path": path, "line_start": call.get("line_start"),
            "line_end": call.get("line_end"), "evidence_ids": [call_ref],
            "dynamic_target": bool(call.get("dynamic_target", False)),
        }
        if not add_node(call_node):
            continue
        caller_id = nearest_caller(call)
        if caller_id:
            add_edge({
                "source_symbol_id": caller_id, "target_symbol_id": callsite_id,
                "kind": "contains_callsite", "certainty": "OBSERVED_SYNTAX",
                "runtime_target_confirmed": False, "evidence_ids": [call_ref],
            })
        if call.get("dynamic_target"):
            add_edge({
                "source_symbol_id": callsite_id, "target_symbol_id": None,
                "kind": "unresolved_call_target", "resolution": "UNRESOLVED_DYNAMIC",
                "certainty": "UNRESOLVED", "runtime_target_confirmed": False,
                "evidence_ids": [call_ref],
            })
            continue

        candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        unindexed_imports: list[tuple[dict[str, Any], str]] = []
        if target.startswith(("self.", "cls.")):
            owner = class_owner(call)
            method_name = target.split(".", 1)[1]
            if owner:
                local_methods = [
                    symbol for symbol in symbols_by_path.get(path, [])
                    if symbol.get("kind") in {"method", "async_method"}
                    and symbol.get("name") == method_name
                    and symbol.get("parent_id") == owner.get("symbol_id")
                ]
                candidates.extend((symbol, symbol) for symbol in local_methods)
        if not candidates and target.isidentifier():
            module = next((item for item in symbols_by_path.get(path, []) if item.get("kind") == "module"), None)
            local = [
                symbol for symbol in symbols_by_path.get(path, [])
                if _callable(str(symbol.get("kind", ""))) and symbol.get("name") == target
                and (symbol.get("parent_id") == call.get("parent_id") or
                     (module and symbol.get("parent_id") == module.get("symbol_id")))
            ]
            candidates.extend((symbol, symbol) for symbol in local)
        if not candidates:
            for module_path, import_record, imported_target in import_candidates(path, target):
                if module_path:
                    terminal = imported_target.rsplit(".", 1)[-1]
                    found = [
                        symbol for symbol in symbols_by_path.get(module_path, [])
                        if _callable(str(symbol.get("kind", ""))) and symbol.get("name") == terminal
                    ]
                    candidates.extend((symbol, import_record) for symbol in found)
                else:
                    unindexed_imports.append((import_record, imported_target))
        unique: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
            str(symbol.get("symbol_id")): (symbol, import_record)
            for symbol, import_record in candidates if symbol.get("symbol_id")
        }
        if unique:
            resolution = "POTENTIAL_UNIQUE" if len(unique) == 1 else "POTENTIAL_AMBIGUOUS"
            for target_id, (symbol, binding) in list(unique.items())[:8]:
                target_ref = add_record(symbol)
                binding_ref = add_record(binding) if binding is not symbol else target_ref
                if target_ref is None or binding_ref is None:
                    continue
                add_edge({
                    "source_symbol_id": caller_id or callsite_id,
                    "target_symbol_id": target_id,
                    "callsite_id": callsite_id,
                    "kind": "calls_candidate",
                    "resolution": resolution,
                    "certainty": "POTENTIAL",
                    "resolution_basis": "bounded_ast_name_and_import_match",
                    "runtime_target_confirmed": False,
                    "evidence_ids": list(dict.fromkeys([call_ref, target_ref, binding_ref])),
                })
        elif unindexed_imports:
            for import_record, external_name in unindexed_imports[:4]:
                import_ref = add_record(import_record)
                if import_ref is None:
                    continue
                external_id = "declared-import:" + hashlib.sha256(external_name.encode("utf-8")).hexdigest()[:24]
                add_node({
                    "node_id": external_id, "kind": "declared_import_candidate",
                    "name": external_name[:256], "path": path,
                    "line_start": import_record.get("line_start"),
                    "line_end": import_record.get("line_end"),
                    "evidence_ids": [import_ref],
                })
                add_edge({
                    "source_symbol_id": caller_id or callsite_id,
                    "target_symbol_id": external_id, "callsite_id": callsite_id,
                    "kind": "calls_declared_import_candidate", "resolution": "DECLARED_IMPORT_NOT_IN_ANALYZED_TREE",
                    "certainty": "POTENTIAL_EXTERNAL_OR_UNANALYZED", "target_origin": "declared_import_not_in_analyzed_tree",
                    "runtime_target_confirmed": False,
                    "evidence_ids": list(dict.fromkeys([call_ref, import_ref])),
                })
        else:
            add_edge({
                "source_symbol_id": caller_id or callsite_id,
                "target_symbol_id": None, "callsite_id": callsite_id,
                "kind": "unresolved_call_target", "resolution": "UNRESOLVED_STATIC",
                "certainty": "UNRESOLVED", "runtime_target_confirmed": False,
                "evidence_ids": [call_ref],
            })

    status_complete = not truncated and all(
        item.get("status") in {"parsed", "empty_source"} and item.get("truncated") is not True
        for item in files
    )
    if any(item.get("dynamic_target") for item in calls):
        limitations.append("Dynamic, reflective and runtime-generated targets remain unresolved.")
    result = {
        "schema_version": SCHEMA_VERSION, "parser_id": PARSER_ID,
        "parser_version": PARSER_VERSION, "nodes": nodes, "edges": edges,
        "truncated": truncated, "complete": status_complete,
        "runtime_dispatch_verified": False,
        "limitations": list(dict.fromkeys(limitations))[:32],
    }
    return result, catalog
