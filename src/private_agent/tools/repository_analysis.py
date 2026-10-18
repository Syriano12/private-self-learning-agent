"""Static evidence extraction only: manifest parsing and syntax analysis, never imports."""
from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import PurePosixPath
from typing import Any, Callable

from private_agent.tools.github_access import RepositoryAccessError

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
    for path in candidates[:8]:
        try:
            evidence = read(path)
        except (RepositoryAccessError, ValueError):
            limitations.append(f"source_not_read:{path}")
            continue
        text = evidence["content"]
        if PurePosixPath(path).suffix == ".py":
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError):
                limitations.append(f"source_syntax_unparsed:{path}")
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    findings.append({"kind": "declared_symbol", "path": path, "line": node.lineno,
                                     "symbol": node.name, "symbol_type": type(node).__name__})
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [alias.name for alias in node.names]
                    module = node.module if isinstance(node, ast.ImportFrom) else ", ".join(names)
                    findings.append({"kind": "declared_import", "path": path, "line": node.lineno,
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
            "hypotheses": hypotheses[:12], "limitations": limitations}
