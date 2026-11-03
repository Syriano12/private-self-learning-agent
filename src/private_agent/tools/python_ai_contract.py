"""Provider-neutral, advisory-only contract for future AI review of static analysis."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from private_agent.tools.python_call_graph import EvidenceRegistry, evidence_key
from private_agent.tools.repository_snapshot import CodeEvidenceRef, RepositorySnapshot

REQUEST_SCHEMA_VERSION = 1
HYPOTHESIS_SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 32768
MAX_RESPONSE_BYTES = 131072
MAX_RESPONSE_NODES = 4096
MAX_RESPONSE_DEPTH = 16
MAX_HYPOTHESES = 16
MAX_TEXT = 1200
FORBIDDEN_AUTHORITY_FIELDS = {
    "tool_calls", "tools", "actions", "action", "approved", "approval",
    "permissions", "capabilities", "finding_status", "validation_status",
    "validated", "authorization", "execute", "exec", "command",
}

AI_HYPOTHESIS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "integer", "const": HYPOTHESIS_SCHEMA_VERSION},
        "hypotheses": {
            "type": "array", "maxItems": MAX_HYPOTHESES,
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string", "minLength": 1, "maxLength": 64},
                    "statement": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "supporting_evidence_ids": {"type": "array", "maxItems": 16, "items": {"type": "string", "maxLength": 64}},
                    "uncertainties": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 256}},
                    "missing_evidence": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 256}},
                },
                "required": ["hypothesis_id", "statement", "confidence", "supporting_evidence_ids", "uncertainties", "missing_evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["schema_version", "hypotheses"],
    "additionalProperties": False,
}


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _content_request_id(request: dict[str, Any], *, prefix: str = "ai-request:") -> str:
    content = dict(request)
    content.pop("request_id", None)
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:32]
    return prefix + digest


def _bind_bundle_identity(request: dict[str, Any], descriptor: dict[str, Any]) -> dict[str, Any]:
    request["bundle_digest"] = descriptor["bundle_digest"]
    request["bundle_request_id"] = descriptor["request_id"]
    request.pop("request_id", None)
    request["request_id"] = _content_request_id(request)
    if _json_size(request) > MAX_REQUEST_BYTES:
        raise ValueError("ai_analysis_request_limit")
    return request


def _compact_reference(evidence_id: str, ref: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "snapshot_id", "repository_identity", "source_revision", "commit", "file_path",
        "file_sha256", "line_start", "line_end", "column_start", "column_end",
        "symbol_id", "parser_id", "parser_version", "rule_id", "rule_version",
        "evidence_type", "truncated", "uncertainty",
    )
    return {"evidence_id": evidence_id, **{key: ref[key] for key in fields if key in ref}}


def _records(value: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
    return {key: value[key] for key in allowed if key in value}


def _reference_set_digest(references: dict[str, dict[str, Any]]) -> str:
    canonical = [
        {"evidence_id": evidence_id, "evidence_ref": references[evidence_id]}
        for evidence_id in sorted(references)
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_ai_analysis_request(
    snapshot: RepositorySnapshot,
    python_ast: dict[str, Any],
    call_graph: dict[str, Any],
    data_flow: dict[str, Any],
    registry: EvidenceRegistry | dict[str, Any],
) -> dict[str, Any]:
    """Build structured, content-free AI input. It never calls any provider."""
    snapshot.validate()
    registry_value = registry.to_dict() if isinstance(registry, EvidenceRegistry) else registry
    if not isinstance(registry_value, dict) or registry_value.get("schema_version") != 1:
        raise ValueError("analysis_evidence_registry_invalid")
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in registry_value.get("references", []):
        if not isinstance(item, dict) or not isinstance(item.get("evidence_ref"), dict):
            raise ValueError("analysis_evidence_registry_invalid")
        raw_ref = item["evidence_ref"]
        ref = CodeEvidenceRef.from_dict(raw_ref, snapshot)
        evidence_id = evidence_key(ref)
        if item.get("evidence_id") != evidence_id or evidence_id in seen:
            raise ValueError("analysis_evidence_registry_invalid")
        seen.add(evidence_id)
        refs.append(_compact_reference(evidence_id, raw_ref))
    if len(refs) > 160:
        raise ValueError("analysis_evidence_registry_limit")

    def require_issued_references(records: Any) -> None:
        if not isinstance(records, list):
            raise ValueError("ai_analysis_evidence_reference_invalid")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("ai_analysis_evidence_reference_invalid")
            evidence_ids = record.get("evidence_ids")
            if not isinstance(evidence_ids, list) or not evidence_ids or any(
                not isinstance(item, str) or item not in seen for item in evidence_ids
            ):
                raise ValueError("ai_analysis_evidence_reference_invalid")

    require_issued_references(call_graph.get("nodes"))
    require_issued_references(call_graph.get("edges"))
    require_issued_references(data_flow.get("nodes"))
    require_issued_references(data_flow.get("edges"))
    known_flow_nodes = {
        item.get("node_id") for item in data_flow.get("nodes", []) if isinstance(item, dict)
    }
    for field in ("source_candidates", "sanitizer_candidates", "sink_candidates"):
        values = data_flow.get(field, [])
        if not isinstance(values, list) or any(item not in known_flow_nodes for item in values):
            raise ValueError("ai_analysis_flow_node_reference_invalid")

    files = [item for item in python_ast.get("files", []) if isinstance(item, dict)]
    parsed_files = [item for item in files if item.get("status") == "parsed"]
    summary = {
        "python_files_seen": len(files),
        "python_files_parsed": len(parsed_files),
        "module_count": len({str(item.get("module_name", "")) for item in parsed_files}),
        "symbol_count": sum(len(item.get("symbols", [])) for item in parsed_files),
        "import_count": sum(len(item.get("imports", [])) for item in parsed_files),
        "call_site_count": sum(len(item.get("calls", [])) for item in parsed_files),
        "coverage_truncated": bool(python_ast.get("truncated", False)),
        "coverage_complete": (
            python_ast.get("complete") is True
            and call_graph.get("complete") is True
            and data_flow.get("complete") is True
        ),
    }
    graph = {
        "schema_version": call_graph.get("schema_version", 1),
        "nodes": [
            _records(item, ("node_id", "kind", "name", "qualname", "path", "line_start", "line_end", "evidence_ids", "dynamic_target"))
            for item in call_graph.get("nodes", []) if isinstance(item, dict)
        ],
        "edges": [
            _records(item, ("source_symbol_id", "target_symbol_id", "callsite_id", "kind", "resolution", "certainty", "target_origin", "runtime_target_confirmed", "evidence_ids"))
            for item in call_graph.get("edges", []) if isinstance(item, dict)
        ],
        "complete": call_graph.get("complete") is True,
        "truncated": bool(call_graph.get("truncated", False)),
        "limitations": [str(item)[:256] for item in call_graph.get("limitations", [])[:16]],
    }
    flow = {
        "schema_version": data_flow.get("schema_version", 1),
        "nodes": [
            _records(item, ("node_id", "kind", "name", "path", "line_start", "line_end", "role_candidate", "sink_target_candidate", "evidence_ids", "uncertainty"))
            for item in data_flow.get("nodes", []) if isinstance(item, dict)
        ],
        "edges": [
            _records(item, ("source_node_id", "target_node_id", "kind", "certainty", "runtime_value_verified", "evidence_ids"))
            for item in data_flow.get("edges", []) if isinstance(item, dict)
        ],
        "source_candidates": list(data_flow.get("source_candidates", [])),
        "sanitizer_candidates": list(data_flow.get("sanitizer_candidates", [])),
        "sink_candidates": list(data_flow.get("sink_candidates", [])),
        "complete": data_flow.get("complete") is True,
        "unresolved": [
            _records(item, ("path", "line_start", "kind", "reason", "name"))
            for item in data_flow.get("unresolved", [])[:32] if isinstance(item, dict)
        ],
        "truncated": bool(data_flow.get("truncated", False)),
        "limitations": [str(item)[:256] for item in data_flow.get("limitations", [])[:16]],
    }
    request: dict[str, Any] = {
        "contract_id": "vulnora.static_analysis.ai_request",
        "schema_version": REQUEST_SCHEMA_VERSION,
        "provider_independent": True,
        "repository": {
            "repository_identity": snapshot.repository_identity,
            "snapshot_id": snapshot.snapshot_id,
            "source_revision": snapshot.source_revision,
            "commit": snapshot.commit,
            "source": snapshot.source,
        },
        "repository_summary": summary,
        "call_graph": graph,
        "data_flow": flow,
        "evidence_catalog": refs,
        "source_code_included": False,
        "source_strings_are_untrusted": True,
        "execution_authority": "none",
        "finding_validation_authority": "none",
    }
    request["request_id"] = "ai-input:" + hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:32]
    if _json_size(request) > MAX_REQUEST_BYTES:
        raise ValueError("ai_analysis_request_limit")
    return request


def _validate_string_list(value: Any, *, max_items: int) -> list[str]:
    if not isinstance(value, list) or len(value) > max_items:
        raise ValueError("ai_hypothesis_list_invalid")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or len(item) > 256:
            raise ValueError("ai_hypothesis_list_invalid")
        result.append(item)
    return result


def _check_forbidden_keys(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited_containers: set[int] = set()
    visited = 0
    text_size = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_RESPONSE_NODES:
            raise ValueError("ai_hypothesis_response_limit")
        if depth > MAX_RESPONSE_DEPTH:
            raise ValueError("ai_hypothesis_response_depth_limit")
        if isinstance(current, dict):
            identity = id(current)
            if identity in visited_containers:
                raise ValueError("ai_hypothesis_response_invalid")
            visited_containers.add(identity)
            if len(current) > MAX_RESPONSE_NODES:
                raise ValueError("ai_hypothesis_response_limit")
            for key, child in current.items():
                if not isinstance(key, str) or len(key) > 256:
                    raise ValueError("ai_hypothesis_response_limit")
                if key.lower() in FORBIDDEN_AUTHORITY_FIELDS:
                    raise ValueError("ai_hypothesis_forbidden_authority_field")
                text_size += len(key)
                stack.append((child, depth + 1))
        elif isinstance(current, list):
            identity = id(current)
            if identity in visited_containers:
                raise ValueError("ai_hypothesis_response_invalid")
            visited_containers.add(identity)
            if len(current) > MAX_RESPONSE_NODES:
                raise ValueError("ai_hypothesis_response_limit")
            stack.extend((child, depth + 1) for child in current)
        elif isinstance(current, str):
            if len(current) > MAX_RESPONSE_BYTES:
                raise ValueError("ai_hypothesis_response_limit")
            text_size += len(current.encode("utf-8"))
        elif current is not None and not isinstance(current, (bool, int, float)):
            raise ValueError("ai_hypothesis_response_invalid")
        if text_size > MAX_RESPONSE_BYTES:
            raise ValueError("ai_hypothesis_response_limit")
    try:
        if _json_size(value) > MAX_RESPONSE_BYTES:
            raise ValueError("ai_hypothesis_response_limit")
    except (TypeError, RecursionError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc) == "ai_hypothesis_response_limit":
            raise
        raise ValueError("ai_hypothesis_response_invalid") from None


def validate_ai_hypothesis_response(
    response: Any,
    request: dict[str, Any],
    evidence_registry: dict[str, Any],
    snapshot_value: RepositorySnapshot | dict[str, Any],
    inspector: Any,
) -> dict[str, Any]:
    """Validate citations and freshness, then keep every model claim advisory-only."""
    _check_forbidden_keys(response)
    if not isinstance(response, dict) or set(response) != {"schema_version", "hypotheses"}:
        raise ValueError("ai_hypothesis_response_schema_invalid")
    if response.get("schema_version") != HYPOTHESIS_SCHEMA_VERSION or isinstance(response.get("schema_version"), bool):
        raise ValueError("ai_hypothesis_schema_version_unsupported")
    base_request_keys = {
        "contract_id", "schema_version", "provider_independent", "repository", "repository_summary",
        "call_graph", "data_flow", "evidence_catalog", "source_code_included",
        "source_strings_are_untrusted", "execution_authority", "finding_validation_authority", "request_id",
    }
    allowed_request_keys = {frozenset(base_request_keys), frozenset(base_request_keys | {"bundle_digest", "bundle_request_id"})}
    if (
        not isinstance(request, dict)
        or frozenset(request) not in allowed_request_keys
        or request.get("contract_id") != "vulnora.static_analysis.ai_request"
        or request.get("schema_version") != REQUEST_SCHEMA_VERSION
    ):
        raise ValueError("ai_analysis_request_invalid")
    if (
        request.get("provider_independent") is not True
        or request.get("source_code_included") is not False
        or request.get("source_strings_are_untrusted") is not True
        or request.get("execution_authority") != "none"
        or request.get("finding_validation_authority") != "none"
    ):
        raise ValueError("ai_analysis_authority_claims_invalid")
    if "bundle_digest" in request:
        bundle_digest = request.get("bundle_digest")
        bundle_request_id = request.get("bundle_request_id")
        hex_chars = set("0123456789abcdef")
        if (
            not isinstance(bundle_digest, str) or len(bundle_digest) != 64
            or any(character not in hex_chars for character in bundle_digest)
            or not isinstance(bundle_request_id, str) or not bundle_request_id.startswith("ai-bundle:")
            or len(bundle_request_id) != len("ai-bundle:") + 32
            or any(character not in hex_chars for character in bundle_request_id[len("ai-bundle:"):])
        ):
            raise ValueError("ai_analysis_request_invalid")
    request_id = request.get("request_id")
    prefix = "ai-request:" if isinstance(request_id, str) and request_id.startswith("ai-request:") else "ai-input:"
    if request_id != _content_request_id(request, prefix=prefix):
        raise ValueError("ai_analysis_request_digest_mismatch")
    snapshot = snapshot_value if isinstance(snapshot_value, RepositorySnapshot) else RepositorySnapshot.from_dict(snapshot_value)
    request_repository = request.get("repository")
    if not isinstance(request_repository, dict) or request_repository.get("snapshot_id") != snapshot.snapshot_id or request_repository.get("repository_identity") != snapshot.repository_identity:
        raise ValueError("ai_analysis_snapshot_mismatch")
    if not callable(getattr(inspector, "verify_snapshot", None)):
        raise ValueError("snapshot_verifier_required")
    inspector.verify_snapshot(snapshot)

    raw_references = evidence_registry.get("references") if isinstance(evidence_registry, dict) else None
    if not isinstance(evidence_registry, dict) or evidence_registry.get("schema_version") != 1 or not isinstance(raw_references, list):
        raise ValueError("analysis_evidence_registry_invalid")
    references: dict[str, CodeEvidenceRef] = {}
    for item in raw_references:
        if not isinstance(item, dict) or not isinstance(item.get("evidence_ref"), dict):
            raise ValueError("analysis_evidence_registry_invalid")
        ref = CodeEvidenceRef.from_dict(item["evidence_ref"], snapshot)
        key = evidence_key(ref)
        if item.get("evidence_id") != key or key in references:
            raise ValueError("analysis_evidence_registry_invalid")
        references[key] = ref
    request_refs = request.get("evidence_catalog")
    if not isinstance(request_refs, list) or any(not isinstance(item, dict) or not isinstance(item.get("evidence_id"), str) for item in request_refs) or {item["evidence_id"] for item in request_refs} != set(references):
        raise ValueError("ai_analysis_evidence_set_mismatch")
    expected_catalog = [
        _compact_reference(key, references[key].to_dict())
        for key in sorted(references)
    ]
    if request_refs != expected_catalog:
        raise ValueError("ai_analysis_evidence_catalog_mismatch")

    for section_name in ("call_graph", "data_flow"):
        section = request.get(section_name)
        if not isinstance(section, dict):
            raise ValueError("ai_analysis_evidence_reference_invalid")
        for record_kind in ("nodes", "edges"):
            records = section.get(record_kind)
            if not isinstance(records, list):
                raise ValueError("ai_analysis_evidence_reference_invalid")
            for record in records:
                evidence_ids = record.get("evidence_ids") if isinstance(record, dict) else None
                if not isinstance(evidence_ids, list) or not evidence_ids or any(
                    not isinstance(item, str) or item not in references for item in evidence_ids
                ):
                    raise ValueError("ai_analysis_evidence_reference_invalid")
    data_flow = request["data_flow"]
    flow_node_ids: set[str] = set()
    for item in data_flow["nodes"]:
        node_id = item.get("node_id") if isinstance(item, dict) else None
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("ai_analysis_flow_node_reference_invalid")
        flow_node_ids.add(node_id)
    for field in ("source_candidates", "sanitizer_candidates", "sink_candidates"):
        candidates = data_flow.get(field)
        if not isinstance(candidates, list) or any(not isinstance(item, str) or item not in flow_node_ids for item in candidates):
            raise ValueError("ai_analysis_flow_node_reference_invalid")

    hypotheses = response.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) > MAX_HYPOTHESES:
        raise ValueError("ai_hypothesis_count_invalid")
    accepted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    allowed = {"hypothesis_id", "statement", "confidence", "supporting_evidence_ids", "uncertainties", "missing_evidence"}
    for item in hypotheses:
        if not isinstance(item, dict) or set(item) != allowed:
            raise ValueError("ai_hypothesis_schema_invalid")
        hypothesis_id = item["hypothesis_id"]
        statement = item["statement"]
        confidence = item["confidence"]
        if not isinstance(hypothesis_id, str) or not hypothesis_id or len(hypothesis_id) > 64 or hypothesis_id in seen_ids:
            raise ValueError("ai_hypothesis_id_invalid")
        if not isinstance(statement, str) or not statement.strip() or len(statement) > MAX_TEXT:
            raise ValueError("ai_hypothesis_statement_invalid")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("ai_hypothesis_confidence_invalid")
        citations = _validate_string_list(item["supporting_evidence_ids"], max_items=16)
        if len(citations) != len(set(citations)) or any(citation not in references for citation in citations):
            raise ValueError("ai_hypothesis_evidence_reference_invalid")
        uncertainties = _validate_string_list(item["uncertainties"], max_items=8)
        missing = _validate_string_list(item["missing_evidence"], max_items=8)
        seen_ids.add(hypothesis_id)
        accepted.append({
            "hypothesis_id": hypothesis_id,
            "statement": statement.strip(),
            "confidence": float(confidence),
            "confidence_origin": "untrusted_provider_estimate",
            "supporting_evidence_ids": citations,
            "uncertainties": uncertainties,
            "missing_evidence": missing,
            "status": "UNVERIFIED",
            "evidence_state": "CITATIONS_PRESENT_NOT_SEMANTICALLY_VERIFIED" if citations else "NO_SUPPORTING_EVIDENCE",
            "finding_promotion_allowed": False,
        })
    return {
        "schema_version": HYPOTHESIS_SCHEMA_VERSION,
        "trusted": False,
        "authority": "ADVISORY_ONLY",
        "hypotheses": accepted,
    }


def _bundle_components(
    snapshot: RepositorySnapshot, python_ast: dict[str, Any], call_graph: dict[str, Any],
    data_flow: dict[str, Any], registry: EvidenceRegistry | dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry_value = registry.to_dict() if isinstance(registry, EvidenceRegistry) else registry
    summary = {
        "python_files_seen": len([item for item in python_ast.get("files", []) if isinstance(item, dict)]),
        "python_files_parsed": sum(1 for item in python_ast.get("files", []) if isinstance(item, dict) and item.get("status") == "parsed"),
        "symbol_count": sum(len(item.get("symbols", [])) for item in python_ast.get("files", []) if isinstance(item, dict)),
        "import_count": sum(len(item.get("imports", [])) for item in python_ast.get("files", []) if isinstance(item, dict)),
        "call_site_count": sum(len(item.get("calls", [])) for item in python_ast.get("files", []) if isinstance(item, dict)),
    }
    components = {
        "snapshot_id": snapshot.snapshot_id,
        "repository_identity": snapshot.repository_identity,
        "source_revision": snapshot.source_revision,
        "python_summary": summary,
        "call_graph": call_graph,
        "data_flow": data_flow,
        "evidence_registry": registry_value,
    }
    if registry_value.get("registry_mode") == "python_ast_base_plus_additional":
        base = _references_from_python_ast(python_ast, snapshot)
        if (
            registry_value.get("base_reference_count") != len(base)
            or registry_value.get("base_reference_digest") != _reference_set_digest(base)
        ):
            raise ValueError("analysis_evidence_base_mismatch")
    return registry_value, components


def build_ai_analysis_contract(
    snapshot: RepositorySnapshot, python_ast: dict[str, Any], call_graph: dict[str, Any],
    data_flow: dict[str, Any], registry: EvidenceRegistry | dict[str, Any],
) -> dict[str, Any]:
    """Return a compact API descriptor; the graph stays in sibling bundle sections."""
    snapshot.validate()
    registry_value, components = _bundle_components(snapshot, python_ast, call_graph, data_flow, registry)
    if not isinstance(registry_value, dict) or registry_value.get("schema_version") != 1:
        raise ValueError("analysis_evidence_registry_invalid")
    bundle_digest = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    descriptor: dict[str, Any] = {
        "contract_id": "vulnora.static_analysis.ai_bundle_ref",
        "schema_version": REQUEST_SCHEMA_VERSION,
        "provider_independent": True,
        "request_materialization": "python_ai_contract.materialize_ai_analysis_request",
        "repository_identity": snapshot.repository_identity,
        "snapshot_id": snapshot.snapshot_id,
        "source_revision": snapshot.source_revision,
        "bundle_digest": bundle_digest,
        "sections": {
            "repository_summary": {"section": "python_ast", "sha256": hashlib.sha256(json.dumps(components["python_summary"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
            "call_graph": {"section": "call_graph", "sha256": hashlib.sha256(json.dumps(call_graph, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()},
            "data_flow": {"section": "data_flow", "sha256": hashlib.sha256(json.dumps(data_flow, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()},
            "evidence": {"section": "analysis_evidence_registry", "sha256": hashlib.sha256(json.dumps(registry_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()},
        },
        "source_code_included": False,
        "excluded_paths_included": False,
        "execution_authority": "none",
        "finding_validation_authority": "none",
    }
    descriptor["request_id"] = "ai-bundle:" + hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:32]
    return descriptor


def materialize_ai_analysis_request(
    analysis_data: dict[str, Any], snapshot_value: RepositorySnapshot | dict[str, Any],
) -> dict[str, Any]:
    """Build the bounded provider-neutral payload from a verified API bundle, without calling a provider."""
    snapshot = snapshot_value if isinstance(snapshot_value, RepositorySnapshot) else RepositorySnapshot.from_dict(snapshot_value)
    registry_value = analysis_data.get("analysis_evidence_registry") if isinstance(analysis_data, dict) else None
    if isinstance(registry_value, dict) and registry_value.get("registry_mode") == "python_ast_base_plus_additional":
        from private_agent.tools.python_analysis_bundle import materialize_ai_analysis_request as materialize_compact

        return materialize_compact(analysis_data, snapshot)
    required = {"python_ast", "call_graph", "data_flow", "analysis_evidence_registry", "ai_analysis_contract"}
    if not isinstance(analysis_data, dict) or not required.issubset(analysis_data):
        raise ValueError("ai_analysis_bundle_incomplete")
    expected = build_ai_analysis_contract(
        snapshot, analysis_data["python_ast"], analysis_data["call_graph"],
        analysis_data["data_flow"], analysis_data["analysis_evidence_registry"],
    )
    if analysis_data["ai_analysis_contract"] != expected:
        raise ValueError("ai_analysis_bundle_digest_mismatch")
    request = build_ai_analysis_request(
        snapshot, analysis_data["python_ast"], analysis_data["call_graph"],
        analysis_data["data_flow"], analysis_data["analysis_evidence_registry"],
    )
    return _bind_bundle_identity(request, expected)


def _references_from_python_ast(python_ast: dict[str, Any], snapshot: RepositorySnapshot) -> dict[str, dict[str, Any]]:
    from private_agent.tools.python_call_graph import evidence_key

    found: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            raw = value.get("evidence_ref")
            if isinstance(raw, dict):
                ref = CodeEvidenceRef.from_dict(raw, snapshot)
                key = evidence_key(ref)
                rendered = ref.to_dict()
                previous = found.get(key)
                if previous is not None and previous != rendered:
                    raise ValueError("analysis_evidence_registry_collision")
                found[key] = rendered
            for key, child in value.items():
                if key != "evidence_ref":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(python_ast)
    return found


def compact_analysis_evidence_registry(
    registry: dict[str, Any], python_ast: dict[str, Any], snapshot: RepositorySnapshot,
) -> dict[str, Any]:
    """Avoid duplicating Phase 2 references already serialized within its AST records."""
    from private_agent.tools.python_call_graph import evidence_key

    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise ValueError("analysis_evidence_registry_invalid")
    base = _references_from_python_ast(python_ast, snapshot)
    additional: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in registry.get("references", []):
        if not isinstance(item, dict) or not isinstance(item.get("evidence_ref"), dict):
            raise ValueError("analysis_evidence_registry_invalid")
        ref = CodeEvidenceRef.from_dict(item["evidence_ref"], snapshot)
        key = evidence_key(ref)
        if key != item.get("evidence_id") or key in seen:
            raise ValueError("analysis_evidence_registry_invalid")
        seen.add(key)
        rendered = ref.to_dict()
        if key in base:
            if base[key] != rendered:
                raise ValueError("analysis_evidence_registry_collision")
        else:
            additional.append({"evidence_id": key, "evidence_ref": rendered})
    return {
        "schema_version": 1,
        "registry_mode": "python_ast_base_plus_additional",
        "snapshot_id": snapshot.snapshot_id,
        "base_reference_digest": _reference_set_digest(base),
        "references": additional,
        "base_reference_count": len(base),
    }


def expand_analysis_evidence_registry(
    registry: dict[str, Any], python_ast: dict[str, Any], snapshot: RepositorySnapshot,
) -> dict[str, Any]:
    """Reconstitute the canonical evidence catalogue for internal validation/materialization."""
    from private_agent.tools.python_call_graph import evidence_key

    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise ValueError("analysis_evidence_registry_invalid")
    if registry.get("registry_mode") not in {None, "python_ast_base_plus_additional"}:
        raise ValueError("analysis_evidence_registry_invalid")
    if registry.get("registry_mode") == "python_ast_base_plus_additional" and registry.get("snapshot_id") != snapshot.snapshot_id:
        raise ValueError("analysis_evidence_snapshot_mismatch")
    combined = _references_from_python_ast(python_ast, snapshot)
    if registry.get("registry_mode") == "python_ast_base_plus_additional" and (
        registry.get("base_reference_count") != len(combined)
        or registry.get("base_reference_digest") != _reference_set_digest(combined)
    ):
        raise ValueError("analysis_evidence_base_mismatch")
    for item in registry.get("references", []):
        if not isinstance(item, dict) or not isinstance(item.get("evidence_ref"), dict):
            raise ValueError("analysis_evidence_registry_invalid")
        ref = CodeEvidenceRef.from_dict(item["evidence_ref"], snapshot)
        key = evidence_key(ref)
        if item.get("evidence_id") != key or key in combined:
            raise ValueError("analysis_evidence_registry_invalid")
        combined[key] = ref.to_dict()
    if len(combined) > 160:
        raise ValueError("analysis_evidence_registry_limit")
    return {
        "schema_version": 1,
        "references": [
            {"evidence_id": key, "evidence_ref": combined[key]}
            for key in sorted(combined)
        ],
    }
