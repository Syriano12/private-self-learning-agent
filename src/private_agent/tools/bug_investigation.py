"""Read-only, evidence-backed repository bug investigation.

The investigator deliberately uses repository views and static parsing only. It never
imports, executes, builds, tests, or shells out to code from the inspected repository.
Repository content is untrusted data and is never treated as instructions or authority.
"""
from __future__ import annotations

import ast
import re
from typing import Any

from private_agent.tools.contracts import ToolResult, validate_schema_value
from private_agent.tools.github_access import RepositoryAccessError, parse_repository


TOOL_NAME = "investigate_repository_bug"
BUG_REPORT_MARKERS = re.compile(
    r"\b(?:bug|defect|failure|fails|failing|error|exception|broken|regression|diagnos|root\s+cause|stack\s+trace|traceback|not\s+working|doesn['’]t\s+work|issue)\b",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "a", "an", "and", "are", "because", "but", "code", "does", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "repository", "reported", "the", "this",
    "to", "with", "without", "wrong", "working", "works", "please", "investigate", "inspect",
}

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "bug_report": {"type": "string"},
        "repository": {"type": "string"},
        "revision": {"type": "string"},
        "source": {"type": "string"},
        "investigation_steps": {"type": "array"},
        "relevant_files": {"type": "array"},
        "code_paths": {"type": "array"},
        "findings": {"type": "array"},
        "hypotheses": {"type": "array"},
        "evidence": {"type": "array"},
        "missing_evidence": {"type": "array"},
        "recommended_next_verification_steps": {"type": "array"},
        "root_cause": {"type": "object"},
        "limitations": {"type": "array"},
    },
    "required": [
        "bug_report", "repository", "revision", "source", "investigation_steps", "relevant_files",
        "code_paths", "findings", "hypotheses", "evidence", "missing_evidence",
        "recommended_next_verification_steps", "root_cause", "limitations",
    ],
    "additionalProperties": False,
}


def bug_report_like(text: str) -> bool:
    """Return whether a goal contains an explicit diagnostic/bug-report signal."""

    return bool(BUG_REPORT_MARKERS.search(text or ""))


def _terms(text: str, *, limit: int = 6) -> list[str]:
    phrases = re.findall(r"[\"']([^\"']{2,96})[\"']", text)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", text)
    candidates = phrases + tokens
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = candidate.strip().strip(".,;:()[]{}")
        if not value or value.lower() in _STOP_WORDS or value.lower() in seen:
            continue
        if value.lower().startswith(("http.", "https.", "github.")):
            continue
        seen.add(value.lower())
        result.append(value)
        if len(result) >= limit:
            break
    if not result:
        result = ["error", "exception"]
    return result


def _line_end(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))


class BugInvestigator:
    """Coordinate bounded repository-tool operations and static evidence extraction."""

    def __init__(self, inspector: Any) -> None:
        self.inspector = inspector

    def _redact(self, value: str) -> str:
        redact = getattr(self.inspector, "_redact", None)
        return redact(value) if callable(redact) else value

    def _call(self, operation: str, inputs: dict[str, Any]) -> dict[str, Any]:
        # RepositoryInspector.inspect is the existing bounded access mechanism. No
        # repository path or credential is accepted from repository content.
        return self.inspector.inspect(operation, inputs)

    @staticmethod
    def _evidence_ref(
        evidence: dict[str, Any],
        *,
        repository: str,
        revision: str,
        line_start: int,
        line_end: int,
        excerpt: str = "",
    ) -> dict[str, Any]:
        return {
            "repository": repository,
            "revision": revision,
            "path": str(evidence.get("path", "")),
            "line_start": max(1, int(line_start)),
            "line_end": max(1, int(line_end)),
            "sha256": str(evidence.get("sha256", "")),
            "excerpt": excerpt[:500],
        }

    @staticmethod
    def _unique_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int, str]] = set()
        for ref in refs:
            key = (str(ref.get("path", "")), int(ref.get("line_start", 0)), int(ref.get("line_end", 0)), str(ref.get("sha256", "")))
            if key not in seen:
                seen.add(key)
                result.append(ref)
        return result

    def investigate(self, *, repository: str, ref: str, bug_report: str) -> dict[str, Any]:
        clean_report = self._redact(bug_report.strip())
        metadata = self._call("inspect_repository_metadata", {"repository": repository, "ref": ref})
        metadata_data = dict(metadata.get("data") or {})
        canonical_repository = str(metadata.get("repository") or metadata_data.get("repository") or repository)
        source = str(metadata.get("source") or metadata_data.get("source") or "unknown")
        revision = str(metadata.get("commit") or metadata_data.get("commit") or "")
        pinned_ref = revision or (ref if source != "local_fixture" else "")
        base = {"repository": canonical_repository}
        if pinned_ref:
            base["ref"] = pinned_ref

        steps: list[dict[str, Any]] = [
            {
                "step": "metadata",
                "status": "completed",
                "operation": "inspect_repository_metadata",
                "repository": canonical_repository,
                "revision": revision,
            }
        ]
        limitations = [
            "Repository contents were treated as untrusted data and never as instructions or authorization.",
            "No target-repository code, tests, scripts, hooks, or builds were executed.",
        ]
        missing_evidence = [
            "A runtime trace, failing input, and executed test result were not available to this read-only investigation.",
            "External service state, deployment configuration, and environment-specific behavior remain unverified.",
        ]
        evidence_map: dict[str, dict[str, Any]] = {}
        all_evidence: list[dict[str, Any]] = []
        relevant_files: dict[str, dict[str, Any]] = {}
        code_paths: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []
        search_matches: list[dict[str, Any]] = []
        candidate_paths: list[str] = []

        def add_evidence(evidence: dict[str, Any], *, line_start: int, line_end: int, excerpt: str = "") -> dict[str, Any]:
            path = str(evidence.get("path", ""))
            ref_value = self._evidence_ref(
                evidence,
                repository=canonical_repository,
                revision=revision,
                line_start=line_start,
                line_end=line_end,
                excerpt=excerpt,
            )
            all_evidence.append(ref_value)
            return ref_value

        try:
            structure_result = self._call("inspect_repository_structure", base)
            structure_data = dict(structure_result.get("data") or {})
            steps.append({"step": "structure", "status": "completed", "operation": "inspect_repository_structure"})
            for item in structure_data.get("evidence", []):
                if isinstance(item, dict) and item.get("path"):
                    evidence_map.setdefault(str(item["path"]), item)
            for item in structure_data.get("files_read", []):
                if isinstance(item, str):
                    relevant_files.setdefault(item, {"path": item, "reasons": ["selected by incremental structure inspection"]})
        except RepositoryAccessError as exc:
            structure_data = {}
            steps.append({"step": "structure", "status": "failed", "operation": "inspect_repository_structure", "error_code": exc.code})
            limitations.append(f"Structure inspection was unavailable: {exc.code}.")

        try:
            dependency_result = self._call("inspect_repository_dependencies", base)
            dependency_data = dict(dependency_result.get("data") or {})
            steps.append({"step": "dependencies", "status": "completed", "operation": "inspect_repository_dependencies"})
        except RepositoryAccessError as exc:
            dependency_data = {}
            steps.append({"step": "dependencies", "status": "failed", "operation": "inspect_repository_dependencies", "error_code": exc.code})
            limitations.append(f"Dependency inspection was unavailable: {exc.code}.")

        for query in _terms(clean_report):
            try:
                search_result = self._call("search_repository_code", {**base, "query": query, "limit": 12})
                data = dict(search_result.get("data") or {})
                matches = [item for item in data.get("matches", []) if isinstance(item, dict)]
                search_matches.extend(matches[:12])
                steps.append({"step": "literal_search", "status": "completed", "operation": "search_repository_code", "query": query, "match_count": len(matches)})
                for match in matches:
                    path = str(match.get("path", ""))
                    if path:
                        candidate_paths.append(path)
                        relevant_files.setdefault(path, {"path": path, "reasons": []})["reasons"].append(f"literal match for {query!r}")
            except RepositoryAccessError as exc:
                steps.append({"step": "literal_search", "status": "failed", "operation": "search_repository_code", "query": query, "error_code": exc.code})
                limitations.append(f"Literal search for {query!r} was unavailable: {exc.code}.")

        # Prefer search hits, then the source files selected by the existing structure tool.
        for item in structure_data.get("files_read", []) if isinstance(structure_data, dict) else []:
            if isinstance(item, str):
                candidate_paths.append(item)
        for item in (dependency_data.get("manifests", []) if isinstance(dependency_data, dict) else []):
            if isinstance(item, str):
                candidate_paths.append(item)
        ordered_candidates: list[str] = []
        for path in candidate_paths:
            if path and path not in ordered_candidates:
                ordered_candidates.append(path)
        ordered_candidates = ordered_candidates[:8]

        file_payloads: dict[str, dict[str, Any]] = {}
        for path in ordered_candidates:
            try:
                read_result = self._call("read_repository_file", {**base, "path": path, "line_count": 400, "max_chars": 16000})
                data = dict(read_result.get("data") or {})
                if data.get("path"):
                    file_payloads[path] = data
                    evidence_map[path] = data
                    ref_value = add_evidence(data, line_start=int(data.get("start_line", 1)), line_end=max(1, int(data.get("start_line", 1)) + len(str(data.get("content", "")).splitlines()) - 1), excerpt=str(data.get("content", ""))[:500])
                    relevant_files.setdefault(path, {"path": path, "reasons": []})["file_evidence"] = ref_value
                    steps.append({"step": "read_candidate", "status": "completed", "operation": "read_repository_file", "path": path})
            except RepositoryAccessError as exc:
                missing_evidence.append(f"Candidate file {path} was not readable ({exc.code}).")
                steps.append({"step": "read_candidate", "status": "failed", "operation": "read_repository_file", "path": path, "error_code": exc.code})

        # Record directly observed literal matches, each tied back to a full-file hash when available.
        for match in search_matches[:16]:
            path = str(match.get("path", ""))
            line = int(match.get("line", 1)) if str(match.get("line", "1")).isdigit() else 1
            evidence = evidence_map.get(path)
            if evidence:
                ref_value = add_evidence(evidence, line_start=line, line_end=line, excerpt=str(match.get("excerpt", "")))
                findings.append({
                    "kind": "observed_search_match",
                    "claim_type": "directly_observed_fact",
                    "claim": f"The literal investigation term appears in repository source at {path}:{line}.",
                    "path": path,
                    "line_start": line,
                    "line_end": line,
                    "evidence_refs": [ref_value],
                })
            else:
                # A search result without a bounded full-file read is not an
                # auditable evidence reference. Keep it explicit as an
                # unverified lead instead of emitting an empty hash.
                findings.append({
                    "kind": "unverified_search_match",
                    "claim_type": "unverified_observation",
                    "claim": f"A bounded literal search returned a candidate at {path}:{line}, but the file was not read within the evidence budget.",
                    "path": path,
                    "line_start": line,
                    "line_end": line,
                    "evidence_refs": [],
                    "missing_evidence": ["A bounded full-file read and SHA-256 for this candidate are unavailable."],
                })
                missing_evidence.append(f"Search candidate {path}:{line} was not promoted to evidence because its full-file hash was unavailable.")

        static_mismatches: list[dict[str, Any]] = []
        for path, payload in file_payloads.items():
            if not path.endswith(".py"):
                continue
            content = str(payload.get("content", ""))
            try:
                tree = ast.parse(content, filename=path)
            except (SyntaxError, ValueError):
                limitations.append(f"Static Python parsing was unavailable for {path}; no runtime claim was made.")
                continue
            definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions.setdefault(node.name, []).append(node)
                    code_paths.append({
                        "path": path,
                        "symbol": node.name,
                        "kind": "function",
                        "line_start": int(node.lineno),
                        "line_end": _line_end(node),
                        "evidence_refs": [add_evidence(payload, line_start=int(node.lineno), line_end=_line_end(node), excerpt="")],
                    })
                elif isinstance(node, ast.ClassDef):
                    code_paths.append({
                        "path": path,
                        "symbol": node.name,
                        "kind": "class",
                        "line_start": int(node.lineno),
                        "line_end": _line_end(node),
                        "evidence_refs": [add_evidence(payload, line_start=int(node.lineno), line_end=_line_end(node), excerpt="")],
                    })
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                candidates = definitions.get(node.func.id, [])
                if len(candidates) != 1:
                    if candidates:
                        limitations.append(
                            f"ambiguous_static_symbol_resolution:{path}:{node.func.id}:multiple_definitions"
                        )
                    continue
                definition = candidates[0]
                parameters = {arg.arg for arg in (*definition.args.posonlyargs, *definition.args.args, *definition.args.kwonlyargs)}
                accepts_arbitrary = definition.args.kwarg is not None
                for keyword in node.keywords:
                    if keyword.arg and keyword.arg not in parameters and not accepts_arbitrary:
                        static_mismatches.append({
                            "path": path,
                            "call_name": node.func.id,
                            "keyword": keyword.arg,
                            "call_line": int(node.lineno),
                            "definition_line": int(definition.lineno),
                            "parameters": sorted(parameters),
                            "payload": payload,
                        })

        terms = {term.lower() for term in _terms(clean_report)}
        for mismatch in static_mismatches[:8]:
            relevant = mismatch["call_name"].lower() in terms or mismatch["keyword"].lower() in terms or any(term in clean_report.lower() for term in ("keyword", "argument", "parameter", "signature"))
            if not relevant:
                continue
            payload = mismatch["payload"]
            call_ref = add_evidence(payload, line_start=mismatch["call_line"], line_end=mismatch["call_line"], excerpt="")
            definition_ref = add_evidence(payload, line_start=mismatch["definition_line"], line_end=mismatch["definition_line"], excerpt="")
            findings.append({
                "kind": "static_call_signature_mismatch",
                "claim_type": "directly_observed_fact",
                "claim": (
                    f"The local static definition of {mismatch['call_name']} does not declare keyword "
                    f"{mismatch['keyword']!r}, but a call passes that keyword."
                ),
                "path": mismatch["path"],
                "line_start": mismatch["call_line"],
                "line_end": mismatch["call_line"],
                "related_line_start": mismatch["definition_line"],
                "related_line_end": mismatch["definition_line"],
                "evidence_refs": [call_ref, definition_ref],
            })

        code_paths = code_paths[:40]
        unique_evidence = self._unique_refs(all_evidence)[:60]
        mismatch_findings = [item for item in findings if item["kind"] == "static_call_signature_mismatch"]
        if mismatch_findings:
            root_cause = {
                "status": "ESTABLISHED",
                "explanation": "A deterministic static signature mismatch directly matches the reported keyword/argument failure. This is a source-level finding; runtime reproduction was intentionally not attempted.",
                "evidence_refs": mismatch_findings[0]["evidence_refs"],
            }
        else:
            root_cause = {
                "status": "UNRESOLVED",
                "explanation": "The bounded static evidence identifies relevant code but does not establish a single root cause without runtime behavior or a reproducible failing input.",
                "evidence_refs": unique_evidence[:4],
            }

        hypotheses: list[dict[str, Any]] = []
        if mismatch_findings:
            hypotheses.append({
                "id": "H1",
                "status": "supported",
                "statement": "The reported failure is caused by a call-site keyword/signature mismatch.",
                "supporting_evidence": mismatch_findings[0]["evidence_refs"],
                "contradicting_evidence": [],
                "missing_evidence": ["A permitted runtime reproduction or failing test would confirm the observed source mismatch reaches the reported path."],
                "confidence": 0.88,
                "confidence_justification": "The definition and call are both present in the same pinned/read-only source evidence.",
                "next_safe_step": "Review or add a focused test in an approved disposable environment; do not execute the target repository during this investigation.",
            })
        if len(relevant_files) >= 2 or search_matches:
            supporting = [item["evidence_refs"][0] for item in findings[:4] if item.get("evidence_refs")]
            hypotheses.append({
                "id": "H2",
                "status": "plausible",
                "statement": "An upstream input-propagation or boundary mismatch may also produce the reported symptom.",
                "supporting_evidence": supporting[:4] or unique_evidence[:2],
                "contradicting_evidence": [],
                "missing_evidence": ["A request-level trace showing the value at each boundary is missing."],
                "confidence": 0.34,
                "confidence_justification": "Multiple relevant files or literal matches were observed, but static evidence does not establish runtime propagation.",
                "next_safe_step": "Trace the reported input through the cited symbols in a separately approved, non-production reproduction.",
            })
        if dependency_data.get("manifests"):
            manifest_refs = [item.get("file_evidence") for item in relevant_files.values() if item.get("path") in set(dependency_data.get("manifests", [])) and item.get("file_evidence")]
            hypotheses.append({
                "id": "H3",
                "status": "unverified",
                "statement": "A dependency or environment interaction remains possible but is not established by declarations alone.",
                "supporting_evidence": manifest_refs[:3],
                "contradicting_evidence": [],
                "missing_evidence": ["Resolved dependency versions, runtime environment, and a failing execution are missing."],
                "confidence": 0.18,
                "confidence_justification": "A manifest was observed, but declared dependencies are not proof of runtime behavior.",
                "next_safe_step": "Compare resolved versions and environment data only after explicit approval and within a read-only diagnostic sandbox.",
            })
        if not hypotheses:
            hypotheses.append({
                "id": "H1",
                "status": "unverified",
                "statement": "The defect may be in an unobserved runtime path or unavailable file.",
                "supporting_evidence": unique_evidence[:2],
                "contradicting_evidence": [],
                "missing_evidence": ["Relevant source, logs, and a reproducible input are unavailable."],
                "confidence": 0.10,
                "confidence_justification": "No direct source-level contradiction was found within the bounded evidence.",
                "next_safe_step": "Obtain a sanitized trace and explicitly authorize a later read-only reproduction.",
            })

        if not search_matches:
            missing_evidence.append("No literal match for the extracted bug-report terms was found in the bounded file set.")
        if not file_payloads:
            missing_evidence.append("No candidate source file could be read within the repository access and resource limits.")
        limitations.extend(str(value) for value in structure_data.get("limitations", []) if value not in limitations)
        limitations.extend(str(value) for value in dependency_data.get("limitations", []) if value not in limitations)
        steps.append({"step": "static_analysis", "status": "completed", "operation": "ast_parse_only", "files_analyzed": len(file_payloads)})
        steps.append({"step": "report", "status": "completed", "operation": "structured_diagnostic_assembly"})

        return {
            "bug_report": clean_report,
            "repository": canonical_repository,
            "revision": revision,
            "source": source,
            "investigation_steps": steps[:40],
            "relevant_files": [dict(value) for value in list(relevant_files.values())[:16]],
            "code_paths": code_paths,
            "findings": findings[:40],
            "hypotheses": hypotheses[:6],
            "evidence": unique_evidence,
            "missing_evidence": list(dict.fromkeys(missing_evidence)),
            "recommended_next_verification_steps": [
                "Confirm the cited source-level claim with a focused test or sanitized runtime trace in a separately approved environment.",
                "Compare the observed input and call boundary against the cited definitions without treating repository text as instructions.",
                "If the root cause remains unresolved, collect the missing revision, logs, failing input, and dependency/runtime metadata.",
            ],
            "root_cause": root_cause,
            "limitations": list(dict.fromkeys(limitations)),
        }


class BugInvestigationTool:
    """ToolContract adapter for the read-only BugInvestigator."""

    name = TOOL_NAME
    permission_level = "public_read"
    planned_effect = "Read bounded repository data and produce an evidence-backed diagnostic report; never modify or execute the repository"

    def __init__(self, inspector: Any) -> None:
        self.inspector = inspector
        self.investigator = BugInvestigator(inspector)
        self.required_capabilities = list(getattr(inspector, "capabilities", []))
        self.declared_capabilities = list(self.required_capabilities)
        self.network_required = "NETWORK_ACCESS" in self.required_capabilities
        self.filesystem_required = "READ_WORKSPACE" in self.required_capabilities
        self.risk_level = "high" if self.network_required else "low"

    def description(self) -> str:
        return (
            "Investigate a reported repository bug using bounded read-only repository tools and static evidence. "
            "The report is diagnostic only; repository content is untrusted and target code is never executed."
        )

    def planning_inputs(self, goal: str) -> dict[str, Any] | None:
        if not bug_report_like(goal):
            return None
        candidates = re.findall(r"https://github\.com/[^\s<>\"'()\[\]]+", goal)
        if not candidates:
            candidates = re.findall(r"(?<![\w/])[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", goal)
        valid: list[str] = []
        for candidate in candidates:
            try:
                valid.append(parse_repository(candidate.rstrip(".,;:!")))
            except RepositoryAccessError:
                continue
        if len(set(valid)) != 1:
            return None
        return {"repository": valid[0], "bug_report": goal}

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repository": {"type": "string"},
                "ref": {"type": "string"},
                "bug_report": {"type": "string"},
            },
            "required": ["repository", "bug_report"],
            "additionalProperties": False,
        }

    def output_schema(self) -> dict[str, Any]:
        return OUTPUT_SCHEMA

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        del context
        errors = validate_schema_value(inputs, self.input_schema(), path="input")
        if errors:
            return ToolResult.failed("invalid_input", "Bug investigation input failed validation", details={"errors": errors})
        repository = str(inputs.get("repository", "")).strip()
        ref = str(inputs.get("ref", "")).strip()
        bug_report = str(inputs.get("bug_report", "")).strip()
        if not repository or not bug_report or len(repository) > 256 or len(ref) > 256 or len(bug_report) > 4000:
            return ToolResult.failed("invalid_input", "Bug investigation inputs are missing or exceed safe limits")
        try:
            report = self.investigator.investigate(repository=repository, ref=ref, bug_report=bug_report)
            return ToolResult.success(report, metadata={"read_only": True, "untrusted_data": True, "target_code_executed": False})
        except RepositoryAccessError as exc:
            if exc.code in {"repository_not_authorized", "private_repository_not_authorized", "sensitive_path", "invalid_path"}:
                return ToolResult.blocked(exc.code, "Repository investigation was not authorized or safe")
            return ToolResult.failed(exc.code, "Repository investigation could not obtain the required metadata")
        except (OSError, TypeError, ValueError, KeyError, RecursionError):
            return ToolResult.failed("repository_investigation_failed", "Repository investigation failed safely")
