"""Deterministic analysis of caller-supplied local evidence.

This module intentionally has no filesystem, network, subprocess, shell, or URL-fetch
capability. Inputs are text artifacts supplied through the authenticated API. They are
redacted and bounded before persistence; all output is observation-only and never
constitutes authorization or a vulnerability finding.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping
from .models import safe_json
from .passive_research import normalize_passive_source_requests

MAX_ARTIFACTS = 8
MAX_ARTIFACT_BYTES = 64_000
MAX_TOTAL_INPUT_BYTES = 128_000
MAX_NAME_LENGTH = 256
MAX_CONTENT_TYPE_LENGTH = 128
MAX_OUTPUT_ITEMS = 10_000
MAX_OUTPUT_ITEM_LENGTH = 512

_SECRET_PATTERNS = (
    re.compile(r"(?im)^(\s*(?:authorization|proxy-authorization|cookie|set-cookie)\s*:\s*)[^\r\n]+$"),
    re.compile(r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|passwd|secret)\b\s*[:=]\s*)(['\"]?)[^\s,'\"}]+\2"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----"),
)
_SENSITIVE_JSON_KEYS = frozenset(
    {
        "authorization", "proxyauthorization", "cookie", "setcookie", "apikey", "accesstoken",
        "refreshtoken", "token", "password", "passwd", "secret", "clientsecret", "privatekey",
    }
)
_SENSITIVE_METADATA_RE = re.compile(r"(?i)(?:authorization|cookie|api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|passwd|secret|private[_-]?key)")
_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_ROUTE_RE = re.compile(r"['\"](\/[-A-Za-z0-9._~:/?&=%+#]+)['\"]")
_CALL_ROUTE_RE = re.compile(r"(?:fetch|axios\.(?:get|post|put|patch|delete)|request)\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
_SENSITIVE_QUERY_KEY_RE = re.compile(r"(?i)^(?:authorization|auth|api[_-]?key|access[_-]?key|access[_-]?token|refresh[_-]?token|token|session|cookie|password|passwd|secret|client[_-]?secret|private[_-]?key)$")
_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
_DYNAMIC_ROUTE_RE = re.compile(r"(?:\{[^}]+\}|:[A-Za-z0-9_-]+)")
_REMOTE_ROUTE_RE = re.compile(r"(?<![A-Za-z0-9_])/(?:api|v[0-9]+)(?:/[A-Za-z0-9._~:/?&=%+#{}-]*)?")


def _bounded_text(value: Any, *, limit: int, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}_required")
    value = value.strip()
    if len(value.encode("utf-8")) > limit:
        raise ValueError(f"{field}_too_large")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ValueError(f"{field}_contains_control_character")
    return value


def _bounded_metadata(value: Any, *, limit: int, field: str) -> str:
    text = _bounded_text(value, limit=limit, field=field)
    if any(ord(char) < 32 for char in text):
        raise ValueError(f"{field}_contains_control_character")
    if _SENSITIVE_METADATA_RE.search(text):
        return "[REDACTED_METADATA]"
    return text


def _redact_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
            result[key] = "[REDACTED]" if normalized_key in _SENSITIVE_JSON_KEYS else _redact_json_value(raw_value)
        return result
    if isinstance(value, list):
        return [_redact_json_value(item) for item in value]
    if isinstance(value, str):
        return _redact_scalar_text(value)
    return value


def _redact_scalar_text(value: str) -> str:
    sanitized = _SECRET_PATTERNS[0].sub(r"\1[REDACTED]", value)
    sanitized = _SECRET_PATTERNS[1].sub(r"\1\2[REDACTED]\2", sanitized)
    sanitized = _SECRET_PATTERNS[2].sub("Bearer [REDACTED]", sanitized)
    sanitized = _SECRET_PATTERNS[3].sub("[REDACTED_PRIVATE_KEY]", sanitized)
    return _URL_RE.sub(lambda match: _redact_url_like(match.group(0)), sanitized)


def redact_local_evidence(value: str) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return _redact_scalar_text(value)
    return json.dumps(_redact_json_value(parsed), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _redact_url_like(value: str) -> str:
    """Remove URL userinfo and sensitive query values without parsing/fetching URLs."""
    value = value.rstrip(".,);]")
    scheme_end = value.find("://")
    if scheme_end >= 0:
        authority_start = scheme_end + 3
        authority_end = len(value)
        for marker in "/?#":
            position = value.find(marker, authority_start)
            if position >= 0:
                authority_end = min(authority_end, position)
        authority = value[authority_start:authority_end]
        if "@" in authority:
            value = value[:authority_start] + authority.rsplit("@", 1)[-1] + value[authority_end:]
    query_start = value.find("?")
    if query_start < 0:
        return value
    fragment_start = value.find("#", query_start)
    query_end = fragment_start if fragment_start >= 0 else len(value)
    pairs = value[query_start + 1:query_end].split("&")
    redacted_pairs: list[str] = []
    for pair in pairs:
        key, separator, query_value = pair.partition("=")
        if _SENSITIVE_QUERY_KEY_RE.fullmatch(key.strip()):
            redacted_pairs.append(f"{key}={separator and '[REDACTED]' or ''}")
        else:
            redacted_pairs.append(pair)
    return value[:query_start + 1] + "&".join(redacted_pairs) + value[query_end:]


@dataclass(frozen=True)
class LocalArtifact:
    name: str
    content_type: str
    content: str
    content_sha256: str


def normalize_local_artifacts(value: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(value or {})
    if set(payload) - {"artifacts"}:
        raise ValueError("stage_input_payload_field_invalid")
    raw_artifacts = payload.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise ValueError("stage_artifacts_list_required")
    if len(raw_artifacts) > MAX_ARTIFACTS:
        raise ValueError("stage_artifacts_too_many")
    artifacts: list[dict[str, str]] = []
    total = 0
    seen: set[str] = set()
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise ValueError("stage_artifact_object_required")
        if set(raw) - {"name", "content_type", "content", "content_sha256"}:
            raise ValueError("stage_artifact_field_invalid")
        name = _bounded_metadata(raw.get("name"), limit=MAX_NAME_LENGTH, field="artifact_name")
        content_type = _bounded_metadata(raw.get("content_type", "text/plain"), limit=MAX_CONTENT_TYPE_LENGTH, field="artifact_content_type")
        if name in seen:
            raise ValueError("stage_artifact_name_duplicate")
        seen.add(name)
        content = redact_local_evidence(_bounded_text(raw.get("content"), limit=MAX_ARTIFACT_BYTES, field="artifact_content"))
        total += len(content.encode("utf-8"))
        if total > MAX_TOTAL_INPUT_BYTES:
            raise ValueError("stage_artifacts_too_large")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        provided_digest = raw.get("content_sha256")
        if provided_digest is not None and provided_digest != digest:
            raise ValueError("stage_artifact_digest_invalid")
        artifacts.append(
            {
                "name": name,
                "content_type": content_type,
                "content": content,
                "content_sha256": digest,
            }
        )
    return {"artifacts": artifacts}


def normalize_research_input(stage_type: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize the closed input contract for one research stage type."""
    payload = dict(value or {})
    if stage_type != "PASSIVE_RESEARCH":
        return normalize_local_artifacts(payload)
    if set(payload) - {"artifacts", "sources"}:
        raise ValueError("stage_input_payload_field_invalid")
    normalized = normalize_local_artifacts({"artifacts": payload.get("artifacts", [])})
    normalized["sources"] = normalize_passive_source_requests({"sources": payload.get("sources", [])})
    return normalized


def _add_bounded(items: list[str], value: str, *, limit: int = MAX_OUTPUT_ITEMS) -> None:
    value = value.strip()
    if len(value) > MAX_OUTPUT_ITEM_LENGTH:
        value = value[:MAX_OUTPUT_ITEM_LENGTH - 1] + "…"
    if value and value not in items and len(items) < limit:
        items.append(value)


def _inventory_artifact(artifact: LocalArtifact, endpoints: list[str], urls: list[str], packages: list[str], files: list[dict[str, str]], limitations: list[str]) -> None:
    files.append({"name": artifact.name, "content_type": artifact.content_type, "content_sha256": artifact.content_sha256})
    content = artifact.content
    for match in _URL_RE.findall(content):
        _add_bounded(urls, _redact_url_like(match))
    for match in _CALL_ROUTE_RE.findall(content):
        _add_bounded(endpoints, _redact_url_like(match))
    for match in _ROUTE_RE.findall(content):
        if match.startswith("/"):
            _add_bounded(endpoints, _redact_url_like(match))
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, Mapping):
        paths = parsed.get("paths")
        if isinstance(paths, Mapping):
            for path, methods in paths.items():
                if not isinstance(path, str) or not path.startswith("/") or not isinstance(methods, Mapping):
                    continue
                for method in methods:
                    if isinstance(method, str) and method.lower() in _HTTP_METHODS:
                        _add_bounded(endpoints, f"{method.upper()} {path}")
        for dependency_key in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
            dependencies = parsed.get(dependency_key)
            if isinstance(dependencies, Mapping):
                for package_name in dependencies:
                    if isinstance(package_name, str):
                        _add_bounded(packages, package_name)
        if parsed.get("openapi") or parsed.get("swagger"):
            if not isinstance(paths, Mapping):
                _add_bounded(limitations, f"{artifact.name}:openapi_paths_missing")
    if len(endpoints) >= MAX_OUTPUT_ITEMS or len(urls) >= MAX_OUTPUT_ITEMS:
        _add_bounded(limitations, "inventory_output_item_limit_reached")


def _fit_inventory_output(output: dict[str, Any], max_output_bytes: int) -> dict[str, Any]:
    def size() -> int:
        return len(json.dumps(output, ensure_ascii=False, sort_keys=True).encode("utf-8"))

    if size() <= max_output_bytes:
        return output
    output["limitations"] = sorted(set([*output["limitations"], "stage_output_limit_reached"]))
    collections = ("endpoints", "urls_observed", "packages", "files")
    while size() > max_output_bytes:
        candidates = [name for name in collections if output[name]]
        if not candidates:
            break
        largest = max(candidates, key=lambda name: len(output[name]))
        output[largest].pop()
    output["artifact_count"] = len(output["files"])
    if size() > max_output_bytes:
        return {
            "kind": "LOCAL_INVENTORY",
            "network_accessed": False,
            "files": [],
            "endpoints": [],
            "urls_observed": [],
            "packages": [],
            "limitations": ["stage_output_limit_reached"],
            "artifact_count": 0,
        }
    return output


def build_local_inventory(input_payload: Mapping[str, Any], *, max_items: int, max_output_bytes: int = 32_000) -> dict[str, Any]:
    normalized = normalize_local_artifacts(input_payload)
    endpoints: list[str] = []
    urls: list[str] = []
    packages: list[str] = []
    files: list[dict[str, str]] = []
    limitations: list[str] = []
    for raw in normalized["artifacts"]:
        _inventory_artifact(LocalArtifact(**raw), endpoints, urls, packages, files, limitations)
    truncated = False
    for collection in (endpoints, urls, packages, files):
        if len(collection) > max_items:
            del collection[max_items:]
            truncated = True
    if truncated:
        _add_bounded(limitations, "stage_max_items_reached")
    return _fit_inventory_output({
        "kind": "LOCAL_INVENTORY",
        "network_accessed": False,
        "files": files,
        "endpoints": endpoints,
        "urls_observed": urls,
        "packages": packages,
        "limitations": sorted(set(limitations)),
        "artifact_count": len(files),
    }, max_output_bytes)


def build_local_passive_observations(input_payload: Mapping[str, Any], *, max_items: int, max_output_bytes: int = 32_000) -> dict[str, Any]:
    inventory = build_local_inventory(input_payload, max_items=max_items, max_output_bytes=max_output_bytes)
    observations: list[dict[str, Any]] = []
    for file_info in inventory["files"][:max_items]:
        observations.append({"kind": "local_artifact_present", "source": file_info["name"], "content_sha256": file_info["content_sha256"]})
    if inventory["endpoints"]:
        observations.append({"kind": "local_routes_observed", "count": len(inventory["endpoints"]), "routes": inventory["endpoints"][:max_items]})
    if inventory["packages"]:
        observations.append({"kind": "local_dependencies_observed", "count": len(inventory["packages"]), "packages": inventory["packages"][:max_items]})
    output = {
        "kind": "LOCAL_PASSIVE_OBSERVATIONS",
        "network_accessed": False,
        "observations": observations[:max_items],
        "limitations": inventory["limitations"],
        "artifact_count": inventory["artifact_count"],
    }
    while len(json.dumps(output, ensure_ascii=False, sort_keys=True).encode("utf-8")) > max_output_bytes and output["observations"]:
        output["observations"].pop()
    if len(json.dumps(output, ensure_ascii=False, sort_keys=True).encode("utf-8")) > max_output_bytes:
        output["observations"] = []
        output["limitations"] = ["stage_output_limit_reached"]
        output["artifact_count"] = 0
    return output


def build_local_hypotheses(input_payload: Mapping[str, Any], *, max_items: int, max_output_bytes: int = 32_000) -> dict[str, Any]:
    normalized = normalize_local_artifacts(input_payload)
    inventory = build_local_inventory(normalized, max_items=max_items, max_output_bytes=max_output_bytes)
    artifact_refs = [
        {"artifact_name": artifact["name"], "content_sha256": artifact["content_sha256"]}
        for artifact in normalized["artifacts"][:max_items]
    ]
    hypotheses: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    network_accessed = False
    passive_source_evidence = False

    def add(
        category: str,
        title: str,
        observation: str,
        conditions: list[str],
        alternatives: list[str],
        *,
        evidence_refs: list[dict[str, Any]] | None = None,
    ) -> None:
        key = (category, observation)
        if key in seen or len(hypotheses) >= max_items:
            return
        seen.add(key)
        evidence = [
            {**reference, "observation": observation}
            for reference in (artifact_refs if evidence_refs is None else evidence_refs)
        ]
        hypothesis_id = "HYP-LOCAL-" + hashlib.sha256(f"{category}:{observation}:{json.dumps(evidence, sort_keys=True)}".encode()).hexdigest()[:16]
        hypotheses.append(
            {
                "hypothesis_id": hypothesis_id,
                "category": category,
                "title": title,
                "status": "UNVERIFIED",
                "scope_required": True,
                "evidence": evidence,
                "reasoning": f"Sanitized evidence contains {observation}; this is a candidate indicator for {category}, not proof of a vulnerability.",
                "confidence": 0.35,
                "confidence_justification": "Deterministic evidence-derived indicator only; no independent verification or model inference was performed.",
                "verification_conditions": conditions,
                "safe_verification_plan": conditions,
                "alternative_explanations": alternatives,
            }
        )

    for endpoint in inventory["endpoints"]:
        lower = endpoint.lower()
        if _DYNAMIC_ROUTE_RE.search(endpoint):
            add(
                "IDOR_BOLA",
                "Object-level authorization should be verified for a dynamic resource route",
                endpoint,
                ["Use only an approved local fixture or explicitly authorized test identity pair", "Compare same-owner and different-owner resource access without mutating data"],
                ["The route may enforce object authorization server-side", "The placeholder may be documentation-only or resolved by a non-user identifier"],
            )
        if any(marker in lower for marker in ("/admin", "/internal", "/staff", "role")):
            add(
                "BROKEN_ACCESS_CONTROL",
                "Privileged-looking API route requires role and authorization review",
                endpoint,
                ["Verify role checks with local fixtures for allowed and denied identities", "Keep the test inside the written scope and do not infer access from route naming"],
                ["The route may be protected by middleware", "The path name may not indicate privileged behavior"],
            )
        if endpoint.startswith(("GET ", "POST ", "PUT ", "PATCH ", "DELETE ")) or "/api/" in lower:
            add(
                "API_SURFACE_REVIEW",
                "API surface requires method, authentication, and error-boundary review",
                endpoint,
                ["Use a local fixture to compare documented and observed method behavior", "Check that errors do not disclose secrets or cross-owner data"],
                ["The endpoint may be non-sensitive or unreachable", "The artifact may describe a stale or generated route"],
            )
    for artifact in normalized["artifacts"][:max_items]:
        lowered = artifact["content"].lower()
        if any(marker in lowered for marker in ("password", "secret", "api_key", "client_secret", "private_key")):
            add(
                "SENSITIVE_DATA_EXPOSURE",
                "Secret-like field marker requires disclosure review; values were redacted",
                f"secret-like field marker in {artifact['name']}",
                ["Inspect only the sanitized local artifact", "Verify that values are not returned to unauthorized identities in a local fixture"],
                ["The marker may be a schema label or documentation example", "Redaction prevents determining whether a real secret was present"],
            )
        try:
            parsed = json.loads(artifact["content"])
        except (TypeError, ValueError):
            parsed = None
        if not isinstance(parsed, Mapping) or parsed.get("kind") != "PASSIVE_RESEARCH":
            continue
        network_accessed = network_accessed or bool(parsed.get("network_accessed"))
        retrievals = parsed.get("retrievals") if isinstance(parsed.get("retrievals"), list) else []
        for retrieval in retrievals[:max_items]:
            if not isinstance(retrieval, Mapping):
                continue
            excerpt = retrieval.get("text_excerpt") if isinstance(retrieval.get("text_excerpt"), str) else ""
            routes = list(dict.fromkeys(_REMOTE_ROUTE_RE.findall(excerpt)))[:max_items]
            if not routes:
                continue
            passive_source_evidence = True
            source_type = str(retrieval.get("source_type", "public_source"))[:128]
            source_url = str(retrieval.get("source_url", ""))[:2048]
            source_digest = str(retrieval.get("content_sha256", ""))[:64]
            evidence_refs = [
                {
                    **reference,
                    "source_type": source_type,
                    "source_url": source_url,
                    "source_content_sha256": source_digest,
                }
                for reference in artifact_refs
            ]
            for route in routes:
                add(
                    "PUBLIC_SOURCE_API_SURFACE",
                    "Public-source API route candidate requires scope-aware review",
                    f"{source_type} observed {route}",
                    [
                        "Treat public documentation as an observation only; do not infer authorization",
                        "Verify only with an approved local fixture or separately authorized target",
                    ],
                    [
                        "The public page may be stale or describe a different deployment",
                        "The route may be documentation-only or protected by controls not visible in the excerpt",
                    ],
                    evidence_refs=evidence_refs,
                )
    output = {
        "kind": "LOCAL_HYPOTHESES",
        "generation_mode": "deterministic_evidence_rules",
        "llm_inference": {"used": False, "reason": "This bounded stage does not call an LLM; configured planner inference is a separate path."},
        "network_accessed": network_accessed,
        "hypotheses": hypotheses,
        "limitations": [
            "hypotheses_are_unverified",
            "llm_inference_not_used",
            *(("public_source_only", "public_source_is_not_authorization", "active_testing_not_performed")
              if passive_source_evidence or network_accessed
              else ("no_external_target_access",)),
        ],
        "artifact_count": len(artifact_refs),
    }
    while len(json.dumps(output, ensure_ascii=False, sort_keys=True).encode("utf-8")) > max_output_bytes and output["hypotheses"]:
        output["hypotheses"].pop()
    if len(json.dumps(output, ensure_ascii=False, sort_keys=True).encode("utf-8")) > max_output_bytes:
        output["hypotheses"] = []
        output["limitations"] = ["hypotheses_are_unverified", "stage_output_limit_reached"]
        output["artifact_count"] = 0
    return output


def build_local_verification(input_payload: Mapping[str, Any], *, max_items: int, max_output_bytes: int = 32_000) -> dict[str, Any]:
    normalized = normalize_local_artifacts(input_payload)
    cases: list[dict[str, Any]] = []
    limitations = ["local_fixture_only", "fixture_comparison_is_not_external_confirmation"]
    artifact_count = 0
    for artifact in normalized["artifacts"]:
        try:
            parsed = json.loads(artifact["content"])
        except (TypeError, ValueError):
            continue
        if not isinstance(parsed, Mapping) or parsed.get("fixture_type") != "local_verification":
            continue
        raw_cases = parsed.get("cases")
        if not isinstance(raw_cases, list):
            limitations.append(f"{artifact['name']}:fixture_cases_missing")
            continue
        artifact_count += 1
        for index, raw_case in enumerate(raw_cases[:max_items]):
            if not isinstance(raw_case, Mapping):
                limitations.append(f"{artifact['name']}:fixture_case_invalid")
                continue
            case_id = raw_case.get("case_id", f"case-{index + 1}")
            if not isinstance(case_id, str) or not case_id.strip() or len(case_id) > 128:
                limitations.append(f"{artifact['name']}:fixture_case_id_invalid")
                continue
            expected = safe_json(raw_case.get("expected"))
            observed = safe_json(raw_case.get("observed"))
            expected_digest = hashlib.sha256(json.dumps(expected, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            observed_digest = hashlib.sha256(json.dumps(observed, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            cases.append(
                {
                    "case_id": case_id.strip(),
                    "matched": expected == observed,
                    "expected_sha256": expected_digest,
                    "observed_sha256": observed_digest,
                    "artifact_name": artifact["name"],
                    "artifact_sha256": artifact["content_sha256"],
                }
            )
            if len(cases) >= max_items:
                break
    output = {
        "kind": "LOCAL_FIXTURE_VERIFICATION",
        "network_accessed": False,
        "finding_confirmation_allowed": False,
        "cases": cases,
        "limitations": sorted(set(limitations)),
        "fixture_artifact_count": artifact_count,
    }
    while len(json.dumps(output, ensure_ascii=False, sort_keys=True).encode("utf-8")) > max_output_bytes and output["cases"]:
        output["cases"].pop()
    if len(json.dumps(output, ensure_ascii=False, sort_keys=True).encode("utf-8")) > max_output_bytes:
        output["cases"] = []
        output["limitations"] = ["local_fixture_only", "stage_output_limit_reached"]
        output["fixture_artifact_count"] = 0
    return output


def build_local_report(input_payload: Mapping[str, Any], *, max_items: int, max_output_bytes: int = 32_000) -> dict[str, Any]:
    normalized = normalize_local_artifacts(input_payload)
    artifact_refs = [
        {"artifact_name": artifact["name"], "content_sha256": artifact["content_sha256"]}
        for artifact in normalized["artifacts"][:max_items]
    ]
    observed_endpoints: list[str] = []
    hypothesis_summaries: list[dict[str, Any]] = []
    verification_summaries: list[dict[str, Any]] = []
    source_retrievals: list[dict[str, Any]] = []
    limitations = ["local_only", "external_target_not_tested", "hypotheses_are_not_findings"]
    network_accessed = False
    recognized = 0
    for artifact in normalized["artifacts"]:
        try:
            parsed = json.loads(artifact["content"])
        except (TypeError, ValueError):
            continue
        if not isinstance(parsed, Mapping):
            continue
        kind = parsed.get("kind")
        if kind in {"LOCAL_INVENTORY", "LOCAL_PASSIVE_OBSERVATIONS"}:
            recognized += 1
            for endpoint in parsed.get("endpoints", []) if isinstance(parsed.get("endpoints"), list) else []:
                if isinstance(endpoint, str):
                    _add_bounded(observed_endpoints, endpoint, limit=max_items)
            for observation in parsed.get("observations", []) if isinstance(parsed.get("observations"), list) else []:
                if isinstance(observation, Mapping):
                    value = observation.get("routes")
                    if isinstance(value, list):
                        for endpoint in value:
                            if isinstance(endpoint, str):
                                _add_bounded(observed_endpoints, endpoint, limit=max_items)
            if isinstance(parsed.get("limitations"), list):
                limitations.extend(str(item)[:128] for item in parsed["limitations"][:max_items])
        elif kind == "PASSIVE_RESEARCH":
            recognized += 1
            network_accessed = network_accessed or bool(parsed.get("network_accessed"))
            limitations.extend(["public_source_only", "public_source_is_not_authorization", "active_testing_not_performed"])
            raw_retrievals = parsed.get("retrievals") if isinstance(parsed.get("retrievals"), list) else []
            for retrieval in raw_retrievals[:max_items]:
                if not isinstance(retrieval, Mapping):
                    continue
                source_retrievals.append(
                    {
                        "source_type": str(retrieval.get("source_type", ""))[:128],
                        "source_url": str(retrieval.get("source_url", ""))[:2048],
                        "retrieved_at": str(retrieval.get("retrieved_at", ""))[:64],
                        "status_code": retrieval.get("status_code") if isinstance(retrieval.get("status_code"), int) else None,
                        "content_sha256": str(retrieval.get("content_sha256", ""))[:64],
                    }
                )
                excerpt = retrieval.get("text_excerpt") if isinstance(retrieval.get("text_excerpt"), str) else ""
                for endpoint in _REMOTE_ROUTE_RE.findall(excerpt)[:max_items]:
                    _add_bounded(observed_endpoints, endpoint, limit=max_items)
        elif kind == "LOCAL_HYPOTHESES":
            recognized += 1
            network_accessed = network_accessed or bool(parsed.get("network_accessed"))
            raw_hypotheses = parsed.get("hypotheses") if isinstance(parsed.get("hypotheses"), list) else []
            for hypothesis in raw_hypotheses[:max_items]:
                if not isinstance(hypothesis, Mapping):
                    continue
                hypothesis_summaries.append(
                    {
                        "hypothesis_id": str(hypothesis.get("hypothesis_id", ""))[:128],
                        "category": str(hypothesis.get("category", ""))[:128],
                        "title": str(hypothesis.get("title", ""))[:512],
                        "status": "UNVERIFIED",
                        "evidence_count": len(hypothesis.get("evidence", [])) if isinstance(hypothesis.get("evidence"), list) else 0,
                    }
                )
            limitations.append("hypotheses_require_scope_and_fixture_verification")
        elif kind == "LOCAL_FIXTURE_VERIFICATION":
            recognized += 1
            raw_cases = parsed.get("cases") if isinstance(parsed.get("cases"), list) else []
            for case in raw_cases[:max_items]:
                if isinstance(case, Mapping):
                    verification_summaries.append(
                        {
                            "case_id": str(case.get("case_id", ""))[:128],
                            "matched": bool(case.get("matched", False)),
                            "expected_sha256": str(case.get("expected_sha256", ""))[:64],
                            "observed_sha256": str(case.get("observed_sha256", ""))[:64],
                        }
                    )
            limitations.append("finding_confirmation_allowed=false")
    limitations = sorted(set(limitations))
    output = {
        "kind": "LOCAL_REPORT",
        "network_accessed": network_accessed,
        "report_status": "PUBLIC_PASSIVE_UNVERIFIED" if network_accessed else "LOCAL_ONLY_UNVERIFIED",
        "finding_confirmation_allowed": False,
        "artifact_refs": artifact_refs[:max_items],
        "source_retrievals": source_retrievals[:max_items],
        "observed_endpoints": observed_endpoints[:max_items],
        "hypotheses": hypothesis_summaries[:max_items],
        "verification_cases": verification_summaries[:max_items],
        "limitations": limitations[:max_items],
        "recognized_stage_results": recognized,
        "report_markdown": "",
    }
    def render_markdown() -> str:
        lines = [
            "# Vulnora local research report",
            f"status={output['report_status']}",
            f"network_accessed={str(output['network_accessed']).lower()}",
            "finding_confirmation_allowed=false",
            "",
            "## Evidence artifacts",
        ]
        lines.extend(f"- {ref['artifact_name']} sha256={ref['content_sha256']}" for ref in output["artifact_refs"])
        lines.append("\n## Observed API surface")
        lines.extend(f"- {endpoint}" for endpoint in output["observed_endpoints"])
        lines.append("\n## Public passive sources")
        lines.extend(
            f"- {item['source_type']} status={item['status_code']} retrieved_at={item['retrieved_at']} sha256={item['content_sha256']}"
            for item in output["source_retrievals"]
        )
        lines.append("\n## Hypotheses (unverified)")
        lines.extend(f"- {item['hypothesis_id']} [{item['status']}] {item['category']}: {item['title']} (evidence={item['evidence_count']})" for item in output["hypotheses"])
        lines.append("\n## Local fixture verification")
        lines.extend(f"- {item['case_id']} matched={str(item['matched']).lower()} expected_sha256={item['expected_sha256']} observed_sha256={item['observed_sha256']}" for item in output["verification_cases"])
        lines.append("\n## Limitations")
        lines.extend(f"- {limitation}" for limitation in output["limitations"])
        return "\n".join(lines)

    output["report_markdown"] = render_markdown()
    while len(json.dumps(output, ensure_ascii=False, sort_keys=True).encode("utf-8")) > max_output_bytes:
        if output["hypotheses"]:
            output["hypotheses"].pop()
        elif output["verification_cases"]:
            output["verification_cases"].pop()
        elif output["observed_endpoints"]:
            output["observed_endpoints"].pop()
        elif output["artifact_refs"]:
            output["artifact_refs"].pop()
        else:
            output["limitations"] = ["local_only", "stage_output_limit_reached"]
            output["report_markdown"] = render_markdown()
            if len(json.dumps(output, ensure_ascii=False, sort_keys=True).encode("utf-8")) > max_output_bytes:
                return {
                    "kind": "LOCAL_REPORT",
                    "network_accessed": network_accessed,
                    "report_status": "PUBLIC_PASSIVE_UNVERIFIED" if network_accessed else "LOCAL_ONLY_UNVERIFIED",
                    "finding_confirmation_allowed": False,
                    "limitations": ["local_only", "stage_output_limit_reached"],
                }
            break
        output["report_markdown"] = render_markdown()
    return output
