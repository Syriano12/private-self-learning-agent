"""Bounded Python syntax-tree extraction. Repository source is parsed, never imported or executed."""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import PurePosixPath
from typing import Any

from private_agent.tools.github_access import validate_repository_path

PARSER_ID = "python.ast"
PARSER_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
RULE_VERSION = "1"
MAX_SOURCE_BYTES = 65536
MAX_AST_NODES = 10000
DEFAULT_MAX_RECORDS = 32


class _RecordLimitReached(Exception):
    pass


def _module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or "<module>"


def _bounded_identifier(value: str, limit: int = 512) -> str:
    if len(value) <= limit:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return value[: limit - len(digest) - 1] + ":" + digest


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id[:256]
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}"[:256] if parent else None
    return None


def _call_target(node: ast.AST) -> tuple[str, bool]:
    if isinstance(node, ast.Call):
        parent, _ = _call_target(node.func)
        return f"{parent}(...)"[:256], True
    value = _dotted_name(node)
    if value is not None:
        return value, False
    return "<dynamic>", True


def _expression_label(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        value = _dotted_name(node.func)
        return f"{value or '<dynamic>'}(...)"[:256]
    value = _dotted_name(node)
    if value is not None:
        return value
    return type(node).__name__[:64]


def _annotation_text(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)[:256]
    except (AttributeError, RecursionError, ValueError):
        return f"<{type(node).__name__}>"


def _byte_offset_to_character_column(line: str, offset: int, *, endpoint: bool) -> int | None:
    if offset < 0:
        return None
    encoded = line.encode("utf-8")
    if offset > len(encoded):
        return None
    try:
        character_count = len(encoded[:offset].decode("utf-8"))
    except UnicodeDecodeError:
        return None
    # AST offsets are UTF-8 byte offsets. EvidenceRef columns use 1-based,
    # inclusive Unicode-codepoint positions.
    if endpoint:
        return character_count if character_count > 0 else None
    return character_count + 1


def _node_location(node: ast.AST, lines: list[str]) -> tuple[int, int, int | None, int | None]:
    raw_start = getattr(node, "lineno", 1)
    line_start = raw_start if isinstance(raw_start, int) and not isinstance(raw_start, bool) else 1
    raw_end = getattr(node, "end_lineno", line_start)
    line_end = raw_end if isinstance(raw_end, int) and not isinstance(raw_end, bool) else line_start
    if not (1 <= line_start <= len(lines) and 1 <= line_end <= len(lines) and line_end >= line_start):
        raise ValueError("invalid_ast_line_range")
    col_start = getattr(node, "col_offset", None)
    col_end = getattr(node, "end_col_offset", None)
    if not isinstance(col_start, int) or not isinstance(col_end, int):
        return line_start, line_end, None, None
    start = _byte_offset_to_character_column(lines[line_start - 1], col_start, endpoint=False)
    end = _byte_offset_to_character_column(lines[line_end - 1], col_end, endpoint=True)
    if start is None or end is None:
        return line_start, line_end, None, None
    return line_start, line_end, start, end


def _empty_file_result(path: str, status: str, reason: str, *, truncated: bool = False) -> dict[str, Any]:
    return {
        "file_path": path,
        "module_name": _module_name(path),
        "status": status,
        "reason": reason,
        "symbols": [],
        "imports": [],
        "parameters": [],
        "decorators": [],
        "annotations": [],
        "calls": [],
        "diagnostics": [],
        "record_count": 0,
        "truncated": truncated,
        "uncertainty": [reason],
    }


def unavailable_python_file(path: str, reason: str, *, status: str = "unavailable") -> dict[str, Any]:
    """Represent a bounded-read failure without inventing source evidence."""
    return _empty_file_result(path, status, reason, truncated=True)


def analyze_python_source(
    path: str,
    source: str,
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> dict[str, Any]:
    """Parse one already-bounded UTF-8 source string and emit syntax-only records."""
    path = validate_repository_path(path)
    if not isinstance(source, str):
        return _empty_file_result(path, "unavailable", "source_not_text", truncated=True)
    if len(source) > MAX_SOURCE_BYTES:
        return _empty_file_result(path, "unavailable", "source_byte_limit", truncated=True)
    if not isinstance(max_records, int) or isinstance(max_records, bool) or not 1 <= max_records <= DEFAULT_MAX_RECORDS:
        raise ValueError("invalid_ast_record_limit")
    try:
        source_bytes = source.encode("utf-8")
    except UnicodeEncodeError:
        return _empty_file_result(path, "unavailable", "invalid_utf8_source", truncated=True)
    if len(source_bytes) > MAX_SOURCE_BYTES:
        return _empty_file_result(path, "unavailable", "source_byte_limit", truncated=True)
    if not source:
        return _empty_file_result(path, "empty_source", "empty_source")

    lines = source.splitlines()
    if not lines:
        lines = [""]
    try:
        tree = ast.parse(source, filename=path, mode="exec", type_comments=True)
    except SyntaxError as exc:
        line = exc.lineno if isinstance(exc.lineno, int) else None
        diagnostics: list[dict[str, Any]] = []
        if line is not None and 1 <= line <= len(lines):
            start = exc.offset if isinstance(exc.offset, int) and exc.offset > 0 else None
            end = getattr(exc, "end_offset", None)
            if not isinstance(end, int) or end < (start or 1):
                end = start
            diagnostics.append({
                "path": path,
                "kind": "syntax_error",
                "rule_id": "python.ast.syntax_error",
                "rule_version": RULE_VERSION,
                "parser_id": PARSER_ID,
                "parser_version": PARSER_VERSION,
                "evidence_type": "parser_diagnostic",
                "line_start": line,
                "line_end": line,
                "column_start": start,
                "column_end": end,
                "symbol_id": _bounded_identifier(f"{path}::syntax_error@{line}:{start or 1}"),
                "truncated": False,
                "uncertainty": [],
            })
        result = _empty_file_result(path, "syntax_error", "syntax_error")
        result["diagnostics"] = diagnostics
        result["record_count"] = len(diagnostics)
        return result
    except (ValueError, RecursionError):
        return _empty_file_result(path, "parse_error", "parser_rejected_source", truncated=True)

    node_count = 0
    for _ in ast.walk(tree):
        node_count += 1
        if node_count > MAX_AST_NODES:
            return _empty_file_result(path, "node_limit", "ast_node_limit", truncated=True)

    module_name = _module_name(path)
    module_id = _bounded_identifier(f"{path}::module:{module_name}@1:1")
    output: dict[str, Any] = {
        "file_path": path,
        "module_name": module_name,
        "status": "parsed",
        "symbols": [],
        "imports": [],
        "parameters": [],
        "decorators": [],
        "annotations": [],
        "calls": [],
        "diagnostics": [],
        "record_count": 0,
        "truncated": False,
        "uncertainty": [],
    }
    record_count = 0

    def add_record(collection: str, record: dict[str, Any]) -> None:
        nonlocal record_count
        if record_count >= max_records:
            raise _RecordLimitReached
        output[collection].append(record)
        record_count += 1

    def common_record(
        node: ast.AST | None,
        *,
        symbol_id: str,
        parent_id: str | None,
        parent_kind: str,
        rule_id: str,
        evidence_type: str,
        location: tuple[int, int, int | None, int | None] | None = None,
    ) -> dict[str, Any]:
        line_start, line_end, column_start, column_end = location or _node_location(node, lines)  # type: ignore[arg-type]
        uncertainty = []
        if column_start is None or column_end is None:
            uncertainty.append("column_range_unavailable")
        return {
            "path": path,
            "line_start": line_start,
            "line_end": line_end,
            "column_start": column_start,
            "column_end": column_end,
            "symbol_id": _bounded_identifier(symbol_id),
            "parent_id": parent_id,
            "parent_kind": parent_kind,
            "parser_id": PARSER_ID,
            "parser_version": PARSER_VERSION,
            "rule_id": rule_id,
            "rule_version": RULE_VERSION,
            "evidence_type": evidence_type,
            "truncated": False,
            "uncertainty": uncertainty,
        }

    # A module covers multiple lines and has no AST-defined token columns.
    module_location = (1, len(lines), None, None)
    module_record = common_record(
        None,
        symbol_id=module_id,
        parent_id=None,
        parent_kind="",
        rule_id="python.ast.module",
        evidence_type="ast_module",
        location=module_location,
    )
    module_record.update({"kind": "module", "name": module_name, "qualname": module_name})
    try:
        add_record("symbols", module_record)
    except _RecordLimitReached:
        return _empty_file_result(path, "record_limit", "ast_record_limit", truncated=True)

    def annotation_record(
        node: ast.AST,
        *,
        parent_id: str,
        parent_kind: str,
        qualname: str,
        role: str,
    ) -> str:
        location = _node_location(node, lines)
        line, _, col, _ = location
        identity = _bounded_identifier(f"{path}::{role}:{qualname}@{line}:{col or 1}")
        record = common_record(
            node,
            symbol_id=identity,
            parent_id=parent_id,
            parent_kind=parent_kind,
            rule_id=f"python.ast.{role}",
            evidence_type="ast_annotation",
            location=location,
        )
        record.update({"kind": role, "annotation": _annotation_text(node), "name": role})
        add_record("annotations", record)
        return identity

    def add_decorators(node: ast.AST, *, owner_id: str, owner_kind: str, qualname: str) -> list[str]:
        identifiers: list[str] = []
        for index, decorator in enumerate(getattr(node, "decorator_list", [])):
            location = _node_location(decorator, lines)
            line, _, col, _ = location
            identifier = _bounded_identifier(f"{path}::decorator:{qualname}:{index}@{line}:{col or 1}")
            record = common_record(
                decorator,
                symbol_id=identifier,
                parent_id=owner_id,
                parent_kind=owner_kind,
                rule_id="python.ast.decorator",
                evidence_type="ast_decorator",
                location=location,
            )
            record.update({"kind": "decorator", "name": _expression_label(decorator), "qualname": qualname})
            add_record("decorators", record)
            identifiers.append(identifier)
        return identifiers

    def add_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda, *, owner_id: str, qualname: str) -> list[str]:
        arguments = node.args
        positional = list(arguments.posonlyargs) + list(arguments.args)
        defaults_start = len(positional) - len(arguments.defaults)
        specifications: list[tuple[ast.arg, str, bool]] = []
        for index, argument in enumerate(arguments.posonlyargs):
            specifications.append((argument, "positional_only", index >= defaults_start))
        offset = len(arguments.posonlyargs)
        for index, argument in enumerate(arguments.args):
            specifications.append((argument, "positional_or_keyword", offset + index >= defaults_start))
        if arguments.vararg is not None:
            specifications.append((arguments.vararg, "variadic_positional", False))
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
            specifications.append((argument, "keyword_only", default is not None))
        if arguments.kwarg is not None:
            specifications.append((arguments.kwarg, "variadic_keyword", False))

        identifiers: list[str] = []
        for argument, parameter_kind, has_default in specifications:
            location = _node_location(argument, lines)
            line, _, col, _ = location
            identifier = _bounded_identifier(f"{path}::parameter:{qualname}:{argument.arg}@{line}:{col or 1}")
            record = common_record(
                argument,
                symbol_id=identifier,
                parent_id=owner_id,
                parent_kind="function",
                rule_id="python.ast.parameter",
                evidence_type="ast_parameter",
                location=location,
            )
            annotation = _annotation_text(argument.annotation)
            record.update({"kind": "parameter", "name": argument.arg[:256], "parameter_kind": parameter_kind,
                           "annotation": annotation, "has_default": has_default, "qualname": qualname})
            add_record("parameters", record)
            identifiers.append(identifier)
            if argument.annotation is not None:
                annotation_id = annotation_record(
                    argument.annotation,
                    parent_id=identifier,
                    parent_kind="parameter",
                    qualname=f"{qualname}.{argument.arg}",
                    role="parameter_annotation",
                )
                record["annotation_id"] = annotation_id
        return identifiers

    def visit(node: ast.AST, parent_id: str, parent_kind: str, scope_names: tuple[str, ...]) -> None:
        if isinstance(node, ast.Module):
            for child in node.body:
                visit(child, module_id, "module", scope_names)
            return

        if isinstance(node, ast.ClassDef):
            qualname = ".".join((*scope_names, node.name))
            location = _node_location(node, lines)
            line, _, col, _ = location
            identifier = _bounded_identifier(f"{path}::class:{qualname}@{line}:{col or 1}")
            record = common_record(node, symbol_id=identifier, parent_id=parent_id, parent_kind=parent_kind,
                                   rule_id="python.ast.class", evidence_type="ast_class", location=location)
            record.update({"kind": "class", "name": node.name[:256], "name_truncated": len(node.name) > 256,
                           "qualname": qualname[:512], "qualname_truncated": len(qualname) > 512,
                           "bases": [_annotation_text(base) for base in node.bases[:16]], "decorator_ids": []})
            if len(node.name) > 256 or len(qualname) > 512 or len(node.bases) > 16:
                record["truncated"] = True
                record["uncertainty"].append("class_metadata_truncated")
            add_record("symbols", record)
            decorator_ids = add_decorators(node, owner_id=identifier, owner_kind="class", qualname=qualname)
            record["decorator_ids"] = decorator_ids
            for child in node.body:
                visit(child, identifier, "class", (*scope_names, node.name))
            for child in (*node.decorator_list, *node.bases, *node.keywords):
                visit(child, identifier, "class", (*scope_names, node.name))
            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join((*scope_names, node.name))
            kind = "method" if parent_kind == "class" else ("async_function" if isinstance(node, ast.AsyncFunctionDef) else "function")
            if kind == "method" and isinstance(node, ast.AsyncFunctionDef):
                kind = "async_method"
            location = _node_location(node, lines)
            line, _, col, _ = location
            identifier = _bounded_identifier(f"{path}::{kind}:{qualname}@{line}:{col or 1}")
            return_annotation = _annotation_text(node.returns)
            record = common_record(node, symbol_id=identifier, parent_id=parent_id, parent_kind=parent_kind,
                                   rule_id=f"python.ast.{kind}", evidence_type="ast_function", location=location)
            record.update({"kind": kind, "name": node.name[:256], "name_truncated": len(node.name) > 256,
                           "qualname": qualname[:512], "qualname_truncated": len(qualname) > 512,
                           "is_async": isinstance(node, ast.AsyncFunctionDef), "return_annotation": return_annotation,
                           "parameter_ids": [], "decorator_ids": [], "return_annotation_id": None,
                           "type_comment": (node.type_comment or "")[:256]})
            if len(node.name) > 256 or len(qualname) > 512:
                record["truncated"] = True
                record["uncertainty"].append("symbol_name_truncated")
            add_record("symbols", record)
            decorator_ids = add_decorators(node, owner_id=identifier, owner_kind=kind, qualname=qualname)
            parameter_ids = add_parameters(node, owner_id=identifier, qualname=qualname)
            record["decorator_ids"] = decorator_ids
            record["parameter_ids"] = parameter_ids
            if node.returns is not None:
                record["return_annotation_id"] = annotation_record(
                    node.returns,
                    parent_id=identifier,
                    parent_kind=kind,
                    qualname=qualname,
                    role="return_annotation",
                )
            for child in node.body:
                visit(child, identifier, kind, (*scope_names, node.name))
            # Calls inside decorators/defaults/annotations are syntax records too,
            # but their expressions are never evaluated.
            for decorator in node.decorator_list:
                visit(decorator, identifier, kind, (*scope_names, node.name))
            parameter_annotations = tuple(
                argument.annotation
                for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                if argument.annotation is not None
            )
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                parameter_annotations += (node.args.vararg.annotation,)
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                parameter_annotations += (node.args.kwarg.annotation,)
            for expression in (*node.args.defaults, *(item for item in node.args.kw_defaults if item is not None), *parameter_annotations, node.returns):
                if expression is not None:
                    visit(expression, identifier, kind, (*scope_names, node.name))
            return

        if isinstance(node, ast.Lambda):
            location = _node_location(node, lines)
            line, _, col, _ = location
            qualname = ".".join((*scope_names, f"<lambda>@{line}:{col or 1}"))
            identifier = _bounded_identifier(f"{path}::lambda:{qualname}")
            record = common_record(node, symbol_id=identifier, parent_id=parent_id, parent_kind=parent_kind,
                                   rule_id="python.ast.lambda", evidence_type="ast_lambda", location=location)
            record.update({"kind": "lambda", "name": "<lambda>", "qualname": qualname[:512],
                           "parameter_ids": [], "decorator_ids": [], "return_annotation": None})
            add_record("symbols", record)
            record["parameter_ids"] = add_parameters(node, owner_id=identifier, qualname=qualname)
            visit(node.body, identifier, "lambda", scope_names)
            annotations = tuple(argument.annotation for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                                if argument.annotation is not None)
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                annotations += (node.args.vararg.annotation,)
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                annotations += (node.args.kwarg.annotation,)
            for expression in (*node.args.defaults, *(item for item in node.args.kw_defaults if item is not None), *annotations):
                visit(expression, identifier, "lambda", scope_names)
            return

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            location = _node_location(node, lines)
            line, _, col, _ = location
            identity = _bounded_identifier(f"{path}::import@{line}:{col or 1}")
            record = common_record(node, symbol_id=identity, parent_id=parent_id, parent_kind=parent_kind,
                                   rule_id="python.ast.import", evidence_type="ast_import", location=location)
            if isinstance(node, ast.Import):
                record.update({"kind": "import", "module": ", ".join(alias.name for alias in node.names[:32])[:256],
                               "aliases": [{"name": alias.name[:256], "asname": alias.asname[:256] if alias.asname else None}
                                           for alias in node.names[:32]], "level": 0})
            else:
                record.update({"kind": "import_from", "module": (node.module or "")[:256],
                               "aliases": [{"name": alias.name[:256], "asname": alias.asname[:256] if alias.asname else None}
                                           for alias in node.names[:32]], "level": min(node.level, 16)})
            if len(node.names) > 32:
                record["truncated"] = True
                record["uncertainty"].append("import_alias_limit")
            add_record("imports", record)
            return

        if isinstance(node, ast.Call):
            location = _node_location(node, lines)
            line, _, col, _ = location
            identity = _bounded_identifier(f"{path}::call@{line}:{col or 1}")
            target, dynamic = _call_target(node.func)
            record = common_record(node, symbol_id=identity, parent_id=parent_id, parent_kind=parent_kind,
                                   rule_id="python.ast.call", evidence_type="ast_call", location=location)
            record.update({"kind": "call", "target": target, "dynamic_target": dynamic})
            add_record("calls", record)

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.arg):
                # Parameters have already been emitted with their owner identity.
                continue
            visit(child, parent_id, parent_kind, scope_names)

    try:
        visit(tree, module_id, "module", ())
    except _RecordLimitReached:
        output["truncated"] = True
        output["status"] = "parsed_truncated"
        output["uncertainty"] = ["ast_record_limit"]
    except (RecursionError, ValueError):
        return _empty_file_result(path, "resource_limited", "ast_walk_limit", truncated=True)

    output["record_count"] = record_count
    return output
