"""Bounded syntax-only Python data-flow candidates; repository code is never executed."""
from __future__ import annotations

import ast
import hashlib
import sys
from typing import Any

from private_agent.tools.python_call_graph import EvidenceRegistry
from private_agent.tools.repository_snapshot import RepositorySnapshot

SCHEMA_VERSION = 1
PARSER_ID = "python.interprocedural_data_flow"
PARSER_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
MAX_AST_NODES = 10000
MAX_AST_DEPTH = 128
MAX_FLOW_NODES = 16
MAX_FLOW_EDGES = 64

SOURCE_NAMES = {"request", "req", "input", "user_input", "data", "payload", "body", "query", "argv", "command"}
SANITIZER_NAMES = {"clean", "sanitize", "escape", "quote", "validate", "normalize", "encode", "parameterize"}
SINK_NAMES = {"execute", "executemany", "eval", "exec", "system", "popen", "run", "open", "render", "send_file", "query"}


def _char_column(line: str, byte_offset: int, *, endpoint: bool) -> int | None:
    encoded = line.encode("utf-8")
    if byte_offset < 0 or byte_offset > len(encoded):
        return None
    try:
        count = len(encoded[:byte_offset].decode("utf-8"))
    except UnicodeDecodeError:
        return None
    return count if endpoint and count else count + (0 if endpoint else 1)


def _location(node: ast.AST, lines: list[str]) -> tuple[int, int, int | None, int | None]:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", start)
    if not isinstance(start, int) or not isinstance(end, int) or not (1 <= start <= end <= len(lines)):
        raise ValueError("invalid_dataflow_location")
    start_byte = getattr(node, "col_offset", None)
    end_byte = getattr(node, "end_col_offset", None)
    if not isinstance(start_byte, int) or not isinstance(end_byte, int):
        return start, end, None, None
    first = _char_column(lines[start - 1], start_byte, endpoint=False)
    last = _char_column(lines[end - 1], end_byte, endpoint=True)
    return start, end, first, last


def _digest_id(*parts: object) -> str:
    rendered = "|".join(str(part) for part in parts)
    return "flow:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:24]


def _has_excessive_ast_depth(root: ast.AST) -> bool:
    pending: list[tuple[ast.AST, int]] = [(root, 1)]
    while pending:
        node, depth = pending.pop()
        if depth > MAX_AST_DEPTH:
            return True
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return False


def _target_terminal(value: str) -> str:
    return value.rsplit(".", 1)[-1].split("(", 1)[0].strip("<>")[-128:]


def build_data_flow(
    python_ast: dict[str, Any], sources_by_path: dict[str, str], snapshot: RepositorySnapshot,
    call_graph: dict[str, Any], registry: EvidenceRegistry,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    node_refs: dict[str, list[str]] = {}
    unresolved: list[dict[str, Any]] = []
    limitations = [
        "Data-flow is a bounded may-flow approximation; aliasing, control-flow paths and runtime values are not executed.",
        "Source, sanitizer and sink labels are lexical candidates only and do not establish a vulnerability or safe transformation.",
    ]
    truncated = False
    incomplete_input = python_ast.get("complete") is not True
    function_contexts: dict[str, dict[str, Any]] = {}
    call_contexts: dict[str, dict[str, Any]] = {}

    def add_node(
        *, node_id: str, kind: str, name: str, path: str, location: tuple[int, int, int | None, int | None],
        evidence_id: str, role_candidate: str | None = None, uncertainty: tuple[str, ...] = (),
    ) -> str:
        nonlocal truncated
        if node_id in node_by_id:
            return node_id
        if len(nodes) >= MAX_FLOW_NODES:
            truncated = True
            return ""
        line_start, line_end, column_start, column_end = location
        record: dict[str, Any] = {
            "node_id": node_id, "kind": kind, "name": name[:256], "path": path,
            "line_start": line_start, "line_end": line_end,
            "column_start": column_start, "column_end": column_end,
            "evidence_ids": [evidence_id], "uncertainty": list(uncertainty),
        }
        if role_candidate:
            record["role_candidate"] = role_candidate
            record["role_basis"] = "bounded_lexical_name_heuristic_only"
        nodes.append(record)
        node_by_id[node_id] = record
        node_refs[node_id] = [evidence_id]
        return node_id

    def add_edge(source: str, target: str, kind: str, certainty: str = "OBSERVED_SYNTAX", extra_refs: list[str] | None = None) -> None:
        nonlocal truncated
        if not source or not target:
            return
        if len(edges) >= MAX_FLOW_EDGES:
            truncated = True
            return
        ref_ids = list(dict.fromkeys([*node_refs.get(source, []), *node_refs.get(target, []), *(extra_refs or [])]))
        if not ref_ids:
            unresolved.append({"kind": "edge_without_evidence", "source_node_id": source, "target_node_id": target})
            return
        edges.append({
            "source_node_id": source, "target_node_id": target, "kind": kind,
            "certainty": certainty, "runtime_value_verified": False, "evidence_ids": ref_ids,
        })

    for file in python_ast.get("files", []):
        if not isinstance(file, dict):
            continue
        path = str(file.get("file_path", ""))
        status = file.get("status")
        if status not in {"parsed", "empty_source"}:
            if status not in {"empty_source"}:
                unresolved.append({"path": path, "kind": "file_flow_unavailable", "reason": str(file.get("reason", status))[:96]})
                truncated = truncated or bool(file.get("truncated", False))
                incomplete_input = True
            continue
        source = sources_by_path.get(path)
        file_record = snapshot.file(path)
        if not isinstance(source, str) or file_record is None:
            unresolved.append({"path": path, "kind": "source_not_bound_to_snapshot"})
            truncated = True
            continue
        raw = source.encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != file_record["sha256"] or len(raw) != file_record["size_bytes"]:
            raise ValueError("source_changed_during_analysis")
        if len(raw) > 65536:
            unresolved.append({"path": path, "kind": "source_byte_limit"})
            truncated = True
            continue
        try:
            tree = ast.parse(source, filename=path, mode="exec", type_comments=True)
        except (SyntaxError, ValueError, RecursionError):
            unresolved.append({"path": path, "kind": "second_parse_failed"})
            incomplete_input = True
            continue
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            unresolved.append({"path": path, "kind": "ast_node_limit"})
            truncated = True
            continue
        if _has_excessive_ast_depth(tree):
            unresolved.append({"path": path, "kind": "ast_depth_limit"})
            truncated = True
            incomplete_input = True
            continue
        lines = source.splitlines() or [""]
        symbol_records = [item for item in file.get("symbols", []) if isinstance(item, dict)]
        parameter_records = [item for item in file.get("parameters", []) if isinstance(item, dict)]
        call_records = [item for item in file.get("calls", []) if isinstance(item, dict)]
        symbols_at = {(item.get("name"), item.get("line_start")): item for item in symbol_records if item.get("kind") in {"function", "async_function", "method", "async_method"}}
        calls_at = {(item.get("line_start"), item.get("column_start")): item for item in call_records}
        params_by_parent: dict[str, list[dict[str, Any]]] = {}
        for parameter in parameter_records:
            params_by_parent.setdefault(str(parameter.get("parent_id", "")), []).append(parameter)

        function_nodes = [item for item in ast.walk(tree) if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for function_node in function_nodes:
            symbol = symbols_at.get((function_node.name, getattr(function_node, "lineno", None)))
            if not symbol:
                unresolved.append({"path": path, "line_start": getattr(function_node, "lineno", 1), "kind": "function_symbol_unmatched"})
                truncated = True
                continue
            symbol_id = str(symbol["symbol_id"])
            parameters = params_by_parent.get(symbol_id, [])
            env: dict[str, list[str]] = {}
            parameter_node_ids: list[str] = []
            for parameter in parameters:
                name = str(parameter.get("name", ""))
                node_id = str(parameter["symbol_id"])
                if node_id not in node_by_id and len(nodes) >= MAX_FLOW_NODES:
                    truncated = True
                    continue
                try:
                    evidence_id = registry.add_record(parameter)
                except ValueError as exc:
                    if str(exc) != "analysis_evidence_limit":
                        raise
                    truncated = True
                    continue
                role = "SOURCE_CANDIDATE" if name.lower() in SOURCE_NAMES else None
                location = (parameter["line_start"], parameter["line_end"], parameter.get("column_start"), parameter.get("column_end"))
                add_node(node_id=node_id, kind="function_parameter", name=name, path=path, location=location,
                         evidence_id=evidence_id, role_candidate=role,
                         uncertainty=("source_classification_is_heuristic",) if role else ())
                env[name] = [node_id]
                parameter_node_ids.append(node_id)
            returns: list[str] = []
            local_calls: list[dict[str, Any]] = []

            def new_location_node(node: ast.AST, kind: str, name: str, *, role: str | None = None, uncertainty: tuple[str, ...] = ()) -> str:
                nonlocal truncated
                location = _location(node, lines)
                line_start, line_end, column_start, column_end = location
                call_record = calls_at.get((line_start, column_start)) if isinstance(node, ast.Call) else None
                symbol_key = str(call_record.get("symbol_id")) if call_record else symbol_id
                node_id = _digest_id(path, symbol_id, kind, line_start, column_start, name)
                if node_id in node_by_id:
                    return node_id
                if len(nodes) >= MAX_FLOW_NODES:
                    truncated = True
                    return ""
                rule_id = f"python.data_flow.{kind}"[:128]
                try:
                    evidence_id = registry.add_location(
                        path=path, line_start=line_start, line_end=line_end,
                        column_start=column_start, column_end=column_end,
                        symbol_id=symbol_key, rule_id=rule_id,
                        uncertainty=uncertainty,
                    )
                except ValueError as exc:
                    if str(exc) != "analysis_evidence_limit":
                        raise
                    truncated = True
                    return ""
                result = add_node(node_id=node_id, kind=kind, name=name, path=path,
                                  location=location, evidence_id=evidence_id,
                                  role_candidate=role, uncertainty=uncertainty)
                if not result:
                    truncated = True
                return result

            def expression_origins(expression: ast.AST) -> list[str]:
                if isinstance(expression, ast.Name):
                    if expression.id in env:
                        return list(env[expression.id])
                    value_id = new_location_node(expression, "unresolved_value", expression.id, uncertainty=("local_definition_unresolved",))
                    unresolved.append({"path": path, "line_start": getattr(expression, "lineno", 1), "kind": "local_definition_unresolved", "name": expression.id[:128]})
                    return [value_id] if value_id else []
                if isinstance(expression, ast.Call):
                    call_location = _location(expression, lines)
                    call_record = calls_at.get((call_location[0], call_location[2]))
                    callsite_id = "callsite:" + str(call_record["symbol_id"]) if call_record else ""
                    target = str(call_record.get("target", "<dynamic>")) if call_record else "<unmatched>"
                    terminal = _target_terminal(target)
                    result_role = "SANITIZER_CANDIDATE" if terminal.lower() in SANITIZER_NAMES else None
                    result_id = new_location_node(expression, "call_result", terminal or "<call>", role=result_role,
                                                  uncertainty=("callee_return_semantics_unverified",))
                    argument_records: list[dict[str, Any]] = []
                    for index, argument in enumerate(expression.args):
                        arg_name = f"#{index}"
                        if isinstance(argument, ast.Starred):
                            inner = argument.value
                            argument_origins = expression_origins(inner)
                            arg_name = "*" + arg_name
                            unresolved.append({"path": path, "line_start": getattr(argument, "lineno", 1), "kind": "star_argument_mapping_unresolved"})
                        else:
                            argument_origins = expression_origins(argument)
                        argument_is_sink = terminal.lower() in SINK_NAMES
                        arg_node = new_location_node(argument, "call_argument", arg_name,
                                                     role="SINK_CANDIDATE" if argument_is_sink else None,
                                                     uncertainty=("argument_binding_is_candidate", "sink_classification_is_heuristic") if argument_is_sink else ("argument_binding_is_candidate",))
                        if argument_is_sink and arg_node:
                            node_by_id[arg_node]["sink_target_candidate"] = terminal[:128]
                        for origin in argument_origins:
                            add_edge(origin, arg_node, "value_to_call_argument")
                        argument_records.append({"name": arg_name, "node_id": arg_node})
                    for keyword in expression.keywords:
                        keyword_name = "**" if keyword.arg is None else keyword.arg
                        if keyword.arg is None:
                            unresolved.append({"path": path, "line_start": getattr(keyword, "lineno", 1), "kind": "keyword_unpack_mapping_unresolved"})
                        keyword_origins = expression_origins(keyword.value)
                        argument_is_sink = terminal.lower() in SINK_NAMES
                        arg_node = new_location_node(keyword.value, "call_argument", keyword_name,
                                                     role="SINK_CANDIDATE" if argument_is_sink else None,
                                                     uncertainty=("argument_binding_is_candidate", "sink_classification_is_heuristic") if argument_is_sink else ("argument_binding_is_candidate",))
                        if argument_is_sink and arg_node:
                            node_by_id[arg_node]["sink_target_candidate"] = terminal[:128]
                        for origin in keyword_origins:
                            add_edge(origin, arg_node, "value_to_call_argument")
                        argument_records.append({"name": keyword_name, "node_id": arg_node})
                    if callsite_id and result_id:
                        local_calls.append({"callsite_id": callsite_id, "target": target,
                                            "result_node_id": result_id, "arguments": argument_records,
                                            "path": path, "line_start": call_location[0]})
                    return [result_id] if result_id else []
                if isinstance(expression, ast.Attribute):
                    origins = expression_origins(expression.value)
                    if not origins:
                        return []
                    attribute_id = new_location_node(expression, "attribute_value", expression.attr)
                    for origin in origins:
                        add_edge(origin, attribute_id, "attribute_value_flow")
                    return [attribute_id] if attribute_id else origins
                if isinstance(expression, ast.Constant):
                    return []
                origins: list[str] = []
                for child in ast.iter_child_nodes(expression):
                    if isinstance(child, ast.expr):
                        origins.extend(expression_origins(child))
                origins = list(dict.fromkeys(item for item in origins if item))
                if not origins:
                    return []
                expression_id = new_location_node(expression, "expression_value", type(expression).__name__)
                for origin in origins:
                    add_edge(origin, expression_id, "value_to_expression")
                return [expression_id] if expression_id else origins

            def assign_target(target: ast.AST, values: list[str], *, uncertainty: tuple[str, ...] = ()) -> None:
                if isinstance(target, ast.Name):
                    assignment_id = new_location_node(target, "assignment", target.id, uncertainty=uncertainty)
                    for value_id in values:
                        add_edge(value_id, assignment_id, "assignment_alias")
                    env[target.id] = [assignment_id] if assignment_id else list(values)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    unresolved.append({"path": path, "line_start": getattr(target, "lineno", 1), "kind": "tuple_unpack_flow_approximation"})
                    for item in target.elts:
                        assign_target(item, values, uncertainty=(*uncertainty, "tuple_unpack_approximation"))
                elif isinstance(target, ast.Attribute):
                    for value_id in values:
                        add_edge(value_id, new_location_node(target, "attribute_assignment", ast.unparse(target)[:128], uncertainty=uncertainty), "attribute_assignment")

            def merge_env(base: dict[str, list[str]], branches: list[dict[str, list[str]]]) -> dict[str, list[str]]:
                merged = {key: list(value) for key, value in base.items()}
                for branch in branches:
                    for key, values in branch.items():
                        merged[key] = list(dict.fromkeys([*merged.get(key, []), *values]))[:8]
                return merged

            def walk_statements(statements: list[ast.stmt], current_env: dict[str, list[str]]) -> dict[str, list[str]]:
                nonlocal env
                original_env = env
                env = current_env
                for statement in statements:
                    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    if isinstance(statement, ast.Assign):
                        values = expression_origins(statement.value)
                        for target in statement.targets:
                            assign_target(target, values)
                    elif isinstance(statement, ast.AnnAssign):
                        values = expression_origins(statement.value) if statement.value is not None else []
                        assign_target(statement.target, values)
                    elif isinstance(statement, ast.NamedExpr):
                        values = expression_origins(statement.value)
                        assign_target(statement.target, values)
                    elif isinstance(statement, ast.AugAssign):
                        values = list(dict.fromkeys([*expression_origins(statement.target), *expression_origins(statement.value)]))
                        assign_target(statement.target, values, uncertainty=("augmented_assignment_approximation",))
                    elif isinstance(statement, ast.Return):
                        values = expression_origins(statement.value) if statement.value is not None else []
                        return_node = new_location_node(statement, "return_value", "return")
                        for value_id in values:
                            add_edge(value_id, return_node, "value_to_return")
                        if return_node:
                            returns.append(return_node)
                    elif isinstance(statement, ast.Expr):
                        expression_origins(statement.value)
                    elif isinstance(statement, ast.If):
                        expression_origins(statement.test)
                        base = {key: list(value) for key, value in env.items()}
                        left = walk_statements(statement.body, {key: list(value) for key, value in base.items()})
                        right = walk_statements(statement.orelse, {key: list(value) for key, value in base.items()})
                        env = merge_env(base, [left, right])
                        unresolved.append({"path": path, "line_start": statement.lineno, "kind": "control_flow_path_not_modeled"})
                    elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                        expression_origins(statement.iter if isinstance(statement, (ast.For, ast.AsyncFor)) else statement.test)
                        base = {key: list(value) for key, value in env.items()}
                        if isinstance(statement, (ast.For, ast.AsyncFor)):
                            assign_target(statement.target, [])
                        body = walk_statements(statement.body, {key: list(value) for key, value in base.items()})
                        other = walk_statements(statement.orelse, {key: list(value) for key, value in base.items()})
                        env = merge_env(base, [body, other])
                        unresolved.append({"path": path, "line_start": statement.lineno, "kind": "loop_flow_approximation"})
                    elif isinstance(statement, (ast.With, ast.AsyncWith)):
                        for item in statement.items:
                            expression_origins(item.context_expr)
                            if item.optional_vars is not None:
                                assign_target(item.optional_vars, [])
                        env = walk_statements(statement.body, env)
                    elif isinstance(statement, ast.Try):
                        base = {key: list(value) for key, value in env.items()}
                        branches = [walk_statements(statement.body, {key: list(value) for key, value in base.items()})]
                        branches.extend(walk_statements(handler.body, {key: list(value) for key, value in base.items()}) for handler in statement.handlers)
                        branches.append(walk_statements(statement.orelse, {key: list(value) for key, value in base.items()}))
                        branches.append(walk_statements(statement.finalbody, {key: list(value) for key, value in base.items()}))
                        env = merge_env(base, branches)
                        unresolved.append({"path": path, "line_start": statement.lineno, "kind": "exception_flow_approximation"})
                    else:
                        for child in ast.iter_child_nodes(statement):
                            if isinstance(child, ast.expr):
                                expression_origins(child)
                result_env = env
                env = original_env
                return result_env

            function_contexts[symbol_id] = {
                "symbol": symbol, "parameter_records": parameters,
                "parameter_node_ids": parameter_node_ids, "return_node_ids": returns,
                "path": path,
            }
            env = {key: list(value) for key, value in env.items()}
            final_env = walk_statements(function_node.body, env)
            function_contexts[symbol_id]["return_node_ids"] = returns
            function_contexts[symbol_id]["final_env"] = final_env
            call_contexts.update({item["callsite_id"]: item for item in local_calls})

    graph_edges = [item for item in call_graph.get("edges", []) if item.get("kind") == "calls_candidate"]
    for graph_edge in graph_edges:
        callsite_id = str(graph_edge.get("callsite_id", ""))
        call_context = call_contexts.get(callsite_id)
        target_context = function_contexts.get(str(graph_edge.get("target_symbol_id", "")))
        if not call_context or not target_context:
            continue
        target_symbol = target_context["symbol"]
        params = list(target_context["parameter_records"])
        call_target = str(call_context.get("target", ""))
        if call_target.startswith(("self.", "cls.")) and params and params[0].get("name") in {"self", "cls"}:
            params = params[1:]
        positional = [item for item in call_context["arguments"] if str(item.get("name", "")).startswith("#")]
        keywords = {str(item["name"]): item for item in call_context["arguments"] if not str(item.get("name", "")).startswith("#") and not str(item.get("name", "")).startswith("*") and item.get("name") != "**"}
        positional_index = 0
        for argument in call_context["arguments"]:
            name = str(argument.get("name", ""))
            if name.startswith("*") or name == "**":
                continue
            parameter: dict[str, Any] | None = None
            if name.startswith("#"):
                while positional_index < len(params) and params[positional_index].get("parameter_kind") == "keyword_only":
                    positional_index += 1
                if positional_index < len(params):
                    parameter = params[positional_index]
                    positional_index += 1
            else:
                parameter = next((item for item in params if item.get("name") == name), None)
            if parameter and str(argument.get("node_id", "")) and str(parameter.get("symbol_id", "")):
                add_edge(str(argument["node_id"]), str(parameter["symbol_id"]), "argument_to_parameter",
                         "POTENTIAL_INTERPROCEDURAL", list(graph_edge.get("evidence_ids", [])))
        for return_node in target_context["return_node_ids"]:
            add_edge(str(return_node), str(call_context["result_node_id"]), "callee_return_to_call_result",
                     "POTENTIAL_INTERPROCEDURAL", list(graph_edge.get("evidence_ids", [])))
        if not target_context["return_node_ids"]:
            unresolved.append({"path": target_context["path"], "line_start": target_symbol.get("line_start"),
                               "kind": "callee_return_not_observed"})

    complete = not incomplete_input and not truncated and call_graph.get("complete") is True
    if any(item.get("kind") in {"control_flow_path_not_modeled", "loop_flow_approximation", "exception_flow_approximation"} for item in unresolved):
        limitations.append("Conditional, loop and exception paths are merged conservatively and are not path-sensitive.")
    return {
        "schema_version": SCHEMA_VERSION, "parser_id": PARSER_ID,
        "parser_version": PARSER_VERSION, "nodes": nodes, "edges": edges,
        "source_candidates": [item["node_id"] for item in nodes if item.get("role_candidate") == "SOURCE_CANDIDATE"],
        "sanitizer_candidates": [item["node_id"] for item in nodes if item.get("role_candidate") == "SANITIZER_CANDIDATE"],
        "sink_candidates": [item["node_id"] for item in nodes if item.get("role_candidate") == "SINK_CANDIDATE"],
        "unresolved": unresolved[:64], "truncated": truncated,
        "complete": complete, "vulnerability_confirmed": False,
        "limitations": list(dict.fromkeys(limitations))[:16],
    }
