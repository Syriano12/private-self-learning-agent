"""Static evidence extraction only: manifest parsing and syntax analysis, never imports."""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import PurePosixPath
from typing import Any, Callable

from private_agent.tools.github_access import RepositoryAccessError
from private_agent.tools.python_ast import (
    DEFAULT_MAX_RECORDS,
    PARSER_ID,
    PARSER_VERSION,
    analyze_python_source,
    unavailable_python_file,
)

MANIFESTS = {"pyproject.toml", "requirements.txt", "package.json", "Cargo.toml", "go.mod", "composer.json"}
SOURCE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java"}
ENTRY_NAMES = {"main.py", "app.py", "cli.py", "__main__.py", "index.js", "index.ts", "server.js", "main.go", "main.rs"}


def dependencies(files: list[dict[str, Any]], read: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    manifests = [item["path"] for item in files if PurePosixPath(item["path"]).name in MANIFESTS]
    entries: list[dict[str, Any]] = []
    entry_points: list[dict[str, Any]] = []
    limitations: list[str] = []
    for path in manifests[:8]:
        try:
            evidence = read(path)
            text = evidence["content"]
            name = PurePosixPath(path).name
            if name == "pyproject.toml":
                payload = tomllib.loads(text)
                project = payload.get("project", {})
                groups = {"runtime": project.get("dependencies", [])}
                groups.update(project.get("optional-dependencies", {}))
                for group, values in groups.items():
                    if isinstance(values, list):
                        entries.extend({"path": path, "group": group, "dependency": str(value)[:512]} for value in values[:100])
                scripts = project.get("scripts", {})
                if isinstance(scripts, dict):
                    entry_points.extend({"path": path, "name": key, "target": str(value)[:512]} for key, value in list(scripts.items())[:20])
            elif name in {"package.json", "composer.json"}:
                payload = json.loads(text)
                for group in ("dependencies", "devDependencies", "peerDependencies", "require", "require-dev"):
                    values = payload.get(group, {})
                    if isinstance(values, dict):
                        entries.extend({"path": path, "group": group, "dependency": key, "version": str(value)[:512]} for key, value in list(values.items())[:100])
                for key in ("main", "module"):
                    if isinstance(payload.get(key), str):
                        entry_points.append({"path": path, "name": key, "target": payload[key][:512]})
                # scripts are reported as declarations, never executed.
                if isinstance(payload.get("scripts"), dict):
                    entry_points.extend({"path": path, "name": key, "target": str(value)[:512]} for key, value in list(payload["scripts"].items())[:20])
            elif name == "Cargo.toml":
                payload = tomllib.loads(text)
                for group in ("dependencies", "dev-dependencies", "build-dependencies"):
                    values = payload.get(group, {})
                    if isinstance(values, dict):
                        entries.extend({"path": path, "group": group, "dependency": key, "version": str(value)[:512]} for key, value in list(values.items())[:100])
            else:
                for number, line in enumerate(text.splitlines(), 1):
                    stripped = line.strip()
                    if stripped and not stripped.startswith(("#", "//", "module ", "go ", "require (", ")")):
                        entries.append({"path": path, "line": number, "group": "declared", "dependency": stripped[:512]})
                        if len(entries) >= 100:
                            break
        except (RepositoryAccessError, ValueError, TypeError, AttributeError) as exc:
            limitations.append(f"manifest_unavailable_or_unparsed:{path}:{type(exc).__name__}")
    if not manifests:
        limitations.append("No supported manifest found in the bounded tree; dependencies are unknown.")
    if len(manifests) > 8:
        limitations.append("Manifest reading limited to eight files.")
    return {"manifests": manifests[:8], "dependencies": entries[:100], "entry_points": entry_points,
            "limitations": limitations, "lockfiles_resolved": False}


def structure(files: list[dict[str, Any]], read: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """Tree -> manifests -> declared entry points -> selected source modules."""
    deps = dependencies(files, read)
    paths = {item["path"] for item in files}
    declared_targets: set[str] = set()
    for entry in deps["entry_points"]:
        target = entry["target"].split(":", 1)[0]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", target):
            module = target.replace(".", "/") + ".py"
            declared_targets.update({module, "src/" + module})
        elif target in paths:
            declared_targets.add(target)
    candidates = [path for path in paths if PurePosixPath(path).suffix in SOURCE_SUFFIXES]
    candidates.sort(key=lambda path: (
        path not in declared_targets,
        PurePosixPath(path).name not in ENTRY_NAMES,
        path.startswith(("tests/", "test/")), len(PurePosixPath(path).parts), path,
    ))
    findings = []
    hypotheses = []
    limitations = list(deps["limitations"])
    selected = candidates[:8]
    python_candidates = [path for path in candidates if PurePosixPath(path).suffix == ".py"]
    selected_python = {path for path in selected if PurePosixPath(path).suffix == ".py"}
    excluded_python = [{"file_path": path, "reason": "source_file_limit"}
                       for path in python_candidates if path not in selected_python]
    python_ast: dict[str, Any] = {
        "parser_id": PARSER_ID,
        "parser_version": PARSER_VERSION,
        "files": [],
        "excluded_files": excluded_python[:40],
        "excluded_files_truncated": len(excluded_python) > 40,
        "record_count": 0,
        "truncated": len(excluded_python) > 0,
        "complete": len(excluded_python) == 0,
    }
    if excluded_python:
        limitations.append("python_ast_source_file_limit")

    for path in selected:
        is_python = PurePosixPath(path).suffix == ".py"
        try:
            evidence = read(path)
        except (RepositoryAccessError, ValueError) as exc:
            limitations.append(f"source_not_read:{path}")
            if is_python:
                reason = exc.code if isinstance(exc, RepositoryAccessError) else "read_failed"
                python_ast["files"].append(unavailable_python_file(path, reason))
                python_ast["truncated"] = True
                python_ast["complete"] = False
            continue
        text = evidence["content"]
        if is_python:
            if evidence.get("redacted"):
                parsed = unavailable_python_file(path, "redacted_source_not_parsed", status="redacted_source_not_parsed")
            elif python_ast["record_count"] >= DEFAULT_MAX_RECORDS:
                parsed = unavailable_python_file(path, "ast_record_limit", status="record_limit")
            else:
                parsed = analyze_python_source(path, text, max_records=DEFAULT_MAX_RECORDS - python_ast["record_count"])
            python_ast["files"].append(parsed)
            python_ast["record_count"] += parsed["record_count"]
            if parsed["truncated"]:
                python_ast["truncated"] = True
                python_ast["complete"] = False
            if parsed["status"] not in {"parsed", "empty_source"}:
                python_ast["complete"] = False
                limitations.append(f"python_ast_{parsed['status']}:{path}")
            if parsed["status"] not in {"parsed", "parsed_truncated"}:
                continue
            module_record = next((item for item in parsed["symbols"] if item["kind"] == "module"), None)
            module_id = module_record["symbol_id"] if module_record else None
            for symbol in parsed["symbols"]:
                if symbol.get("parent_id") != module_id or symbol.get("kind") not in {"class", "function", "async_function"}:
                    continue
                symbol_type = {"class": "ClassDef", "function": "FunctionDef", "async_function": "AsyncFunctionDef"}[symbol["kind"]]
                findings.append({"kind": "declared_symbol", "path": path, "line": symbol["line_start"],
                                 "symbol": symbol["name"], "symbol_type": symbol_type})
            for imported in parsed["imports"]:
                if imported.get("parent_id") != module_id:
                    continue
                names = [item["name"] for item in imported["aliases"]]
                module = imported.get("module") or ", ".join(names)
                findings.append({"kind": "declared_import", "path": path, "line": imported["line_start"],
                                 "module": module or "relative", "names": names[:10]})
            if path in declared_targets:
                hypotheses.append({"claim": "This module is a likely executable entry boundary based on the manifest declaration.",
                                   "path": path, "basis": "declared_entry_point", "confidence": "hypothesis"})
        else:
            for number, line in enumerate(text.splitlines(), 1):
                if re.search(r"\b(class|function|export|func|fn|import|require)\b", line):
                    findings.append({"kind": "source_declaration_candidate", "path": path, "line": number,
                                     "excerpt": line[:240]})
                    if len(findings) >= 80:
                        break
            hypotheses.append({"claim": "Non-Python declarations are lexical candidates, not a complete semantic model.",
                               "path": path, "confidence": "hypothesis"})
    directories = sorted({path.split("/", 1)[0] for path in paths if "/" in path})[:30]
    limitations.extend([
        "Static declarations do not establish runtime behavior or correctness.",
        "Only selected files were read; unseen modules and dynamic imports remain unknown.",
        "Dependency constraints are manifest declarations, not installed or resolved versions.",
    ])
    return {"main_directories": directories, "entry_points": deps["entry_points"],
            "dependencies": deps["dependencies"], "manifests": deps["manifests"],
            "verified_findings": [item for item in findings if item["kind"] != "source_declaration_candidate"][:80],
            "lexical_candidates": [item for item in findings if item["kind"] == "source_declaration_candidate"][:80],
            "hypotheses": hypotheses[:12], "limitations": limitations, "python_ast": python_ast}
