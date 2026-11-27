"""Bounded passive reconnaissance and deterministic authorization testing."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from private_agent.bugbounty.fixtures import DeterministicFixtureApp, FixtureEndpoint, ScopeEnforcedFixtureClient
from private_agent.bugbounty.models import (
    AuthorizationObservation,
    Evidence,
    Finding,
    FindingState,
    PolicyDecision,
    ScopeDefinition,
    ScopeDecision,
    Severity,
    TestCaseResult,
    semantic_normalize,
    stable_digest,
    utc_now,
)


@dataclass(frozen=True)
class ReconObservation:
    observation_id: str
    kind: str
    source: str
    asset: str
    details: dict[str, Any]
    confidence: float
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "kind": self.kind,
            "source": self.source,
            "asset": self.asset,
            "details": self.details,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }


@dataclass
class AttackSurfaceInventory:
    assets: list[str]
    endpoints: list[dict[str, Any]]
    observations: list[ReconObservation]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "assets": self.assets[:100],
            "endpoints": self.endpoints[:100],
            "observations": [item.to_dict() for item in self.observations[:100]],
            "limitations": self.limitations[:100],
        }


class PassiveRecon:
    """Parse supplied local artifacts only; never fetches a URL."""

    MAX_URL_INPUTS = 100
    MAX_SOURCE_DOCUMENTS = 50
    MAX_SOURCE_BYTES = 2_000_000
    MAX_RESEARCH_STAGE_RESULTS = 4
    MAX_RESEARCH_STAGE_ENDPOINTS = 100
    _URL_RE = re.compile(r"https?://[A-Za-z0-9._:-]{1,255}(?:/[A-Za-z0-9._~:/?#[\]@!$&'()*+,;=%{}-]*)?")
    _ROUTE_RE = re.compile(r"(?:app|router|route)\.(?:get|post|put|patch|delete)\(['\"]([^'\"]+)")

    @staticmethod
    def _inventory_candidate_path(value: Any) -> str | None:
        if not isinstance(value, str) or not value.strip() or len(value) > 2048:
            return None
        candidate = value.strip()
        if any(ord(char) < 32 for char in candidate):
            return None
        if candidate.startswith(("http://", "https://")):
            try:
                parsed = urlparse(candidate)
            except ValueError:
                return None
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or any(char.isspace() for char in parsed.hostname):
                return None
            candidate = parsed.path or "/"
        elif " " in candidate:
            method, candidate = candidate.split(None, 1)
            if method.upper() not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE"}:
                return None
        path = candidate.split("?", 1)[0].split("#", 1)[0]
        if not path.startswith("/") or len(path) > 512 or any(ord(char) < 32 for char in path):
            return None
        return path

    def analyze(
        self,
        *,
        urls: Sequence[str] = (),
        openapi: Mapping[str, Any] | None = None,
        source_documents: Mapping[str, str] | None = None,
        fixture: DeterministicFixtureApp | None = None,
        research_stage_results: Sequence[Mapping[str, Any]] = (),
    ) -> AttackSurfaceInventory:
        assets: set[str] = set()
        endpoints: list[dict[str, Any]] = []
        observations: list[ReconObservation] = []
        limitations: list[str] = []

        def add(kind: str, source: str, asset: str, details: dict[str, Any], confidence: float = 0.7) -> None:
            raw_id = stable_digest([kind, source, asset, details])[:16]
            observations.append(ReconObservation("O-" + raw_id, kind, source, asset, details, confidence, {"mode": "passive", "source": source}))
            if asset:
                assets.add(asset)

        for index, raw in enumerate(urls):
            if index >= self.MAX_URL_INPUTS:
                limitations.append("url_input_limit_reached")
                break
            if not isinstance(raw, str) or len(raw) > 2048:
                limitations.append("invalid_or_oversized_url_ignored")
                continue
            parsed = urlparse(raw)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                limitations.append("non_http_url_ignored")
                continue
            asset = parsed.hostname.lower()
            endpoints.append({"method": "GET", "path": parsed.path or "/", "source": "supplied_url", "asset": asset})
            add("url", raw, asset, {"method": "GET", "path": parsed.path or "/"}, 0.8)

        stage_endpoint_count = 0
        if len(research_stage_results) > self.MAX_RESEARCH_STAGE_RESULTS:
            limitations.append("research_inventory_stage_result_limit_reached")
        for stage_source in research_stage_results[:self.MAX_RESEARCH_STAGE_RESULTS]:
            if not isinstance(stage_source, Mapping) or stage_source.get("stage_type") != "INVENTORY":
                continue
            stage_id = stage_source.get("stage_id")
            result_digest = stage_source.get("result_sha256")
            result = stage_source.get("result")
            if (
                not isinstance(stage_id, str) or not stage_id or len(stage_id) > 128
                or not isinstance(result_digest, str) or not re.fullmatch(r"[a-f0-9]{64}", result_digest)
                or not isinstance(result, Mapping) or result.get("kind") != "LOCAL_INVENTORY"
                or result.get("network_accessed") is not False
            ):
                limitations.append("research_inventory_stage_invalid_or_not_local")
                continue
            stage_entries: list[tuple[str, str]] = []
            malformed_collection = False
            for field_name in ("endpoints", "urls_observed"):
                entries = result.get(field_name, [])
                if not isinstance(entries, list):
                    malformed_collection = True
                    break
                if len(entries) > self.MAX_RESEARCH_STAGE_ENDPOINTS:
                    limitations.append("research_inventory_endpoint_limit_reached")
                stage_entries.extend((field_name, entry) for entry in entries[:self.MAX_RESEARCH_STAGE_ENDPOINTS]
                                     if isinstance(entry, str))
                if any(not isinstance(entry, str) for entry in entries[:self.MAX_RESEARCH_STAGE_ENDPOINTS]):
                    limitations.append("research_inventory_non_text_candidate_ignored")
            if malformed_collection:
                limitations.append("research_inventory_stage_shape_invalid")
                continue
            if len(stage_entries) > self.MAX_RESEARCH_STAGE_ENDPOINTS:
                stage_entries = stage_entries[:self.MAX_RESEARCH_STAGE_ENDPOINTS]
                limitations.append("research_inventory_endpoint_limit_reached")
            seen_stage_paths: set[str] = set()
            for field_name, raw_candidate in stage_entries:
                if stage_endpoint_count >= self.MAX_RESEARCH_STAGE_ENDPOINTS:
                    limitations.append("research_inventory_endpoint_limit_reached")
                    break
                path = self._inventory_candidate_path(raw_candidate)
                if path is None:
                    limitations.append("research_inventory_invalid_path_ignored")
                    continue
                if path in seen_stage_paths:
                    continue
                seen_stage_paths.add(path)
                stage_endpoint_count += 1
                provenance = {
                    "mode": "passive",
                    "source": "completed_local_inventory_stage",
                    "source_stage_id": stage_id,
                    "result_sha256": result_digest,
                    "source_field": field_name,
                    "trust": "unverified_candidate",
                }
                endpoint = {
                    "method": "UNKNOWN",
                    "path": path,
                    "asset": "",
                    "source": "research_stage_inventory",
                    "candidate_only": True,
                    "confidence": 0.2,
                    "provenance": provenance,
                }
                endpoints.append(endpoint)
                add(
                    "research_stage_route_candidate",
                    "research_stage_inventory",
                    "",
                    {"path": path, "trust": "unverified_candidate", "source_stage_id": stage_id,
                     "result_sha256": result_digest, "source_field": field_name},
                    0.2,
                )

        if openapi is not None:
            paths = openapi.get("paths") if isinstance(openapi, Mapping) else None
            if isinstance(paths, Mapping):
                servers = openapi.get("servers", [])
                server_url = ""
                if isinstance(servers, list) and servers and isinstance(servers[0], Mapping):
                    server_url = str(servers[0].get("url", ""))
                server_host = urlparse(server_url).hostname.lower() if urlparse(server_url).hostname else "fixture.local"
                for path_index, (path, operations) in enumerate(paths.items()):
                    if path_index >= 500:
                        limitations.append("openapi_path_limit_reached")
                        break
                    if not isinstance(path, str) or not path.startswith("/") or len(path) > 512 or not isinstance(operations, Mapping):
                        continue
                    for method, operation in operations.items():
                        method_upper = str(method).upper()
                        if method_upper not in {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"}:
                            continue
                        entry = {"method": method_upper, "path": path, "asset": server_host, "source": "openapi"}
                        if isinstance(operation, Mapping):
                            entry["operation_id"] = str(operation.get("operationId", ""))[:256]
                            entry["security"] = bool(operation.get("security"))
                        endpoints.append(entry)
                        add("openapi_endpoint", "openapi", server_host, entry, 0.75)
            else:
                limitations.append("openapi_paths_missing")

        source_bytes = 0
        for source_index, (source_name, document) in enumerate((source_documents or {}).items()):
            if source_index >= self.MAX_SOURCE_DOCUMENTS:
                limitations.append("source_document_limit_reached")
                break
            if not isinstance(source_name, str) or not isinstance(document, str) or len(document.encode("utf-8")) > 512_000:
                limitations.append("source_document_invalid_or_oversized")
                continue
            source_bytes += len(document.encode("utf-8"))
            if source_bytes > self.MAX_SOURCE_BYTES:
                limitations.append("source_document_bytes_limit_reached")
                break
            for match in self._URL_RE.finditer(document):
                raw = match.group(0)[:2048]
                parsed = urlparse(raw)
                if parsed.hostname:
                    endpoints.append({"method": "GET", "path": parsed.path or "/", "asset": parsed.hostname.lower(), "source": source_name})
                    add("source_url", source_name, parsed.hostname.lower(), {"url": raw}, 0.45)
            for match in self._ROUTE_RE.finditer(document):
                path = match.group(1)
                if path.startswith("/") and len(path) <= 512:
                    endpoints.append({"method": "UNKNOWN", "path": path, "asset": "local-source", "source": source_name})
                    add("source_route", source_name, "local-source", {"path": path}, 0.4)
            if "authorization" in document.lower() or "permission" in document.lower():
                add("auth_boundary_hint", source_name, "local-source", {"reason": "authorization-related token found"}, 0.35)

        if fixture is not None:
            for endpoint in fixture.inventory():
                endpoints.append({**endpoint, "asset": urlparse(fixture.base_url).hostname, "source": "deterministic_fixture"})
                add("fixture_endpoint", "deterministic_fixture", urlparse(fixture.base_url).hostname or "", endpoint, 1.0)

        if not endpoints:
            limitations.append("no_local_attack_surface_supplied")
        return AttackSurfaceInventory(sorted(assets), endpoints, observations, limitations)


class DeterministicPlanner:
    """No-LLM planner used for reproducible local evaluation."""

    def build(self, inventory: AttackSurfaceInventory, scope: ScopeDefinition) -> list[dict[str, Any]]:
        plan: list[dict[str, Any]] = []
        for index, endpoint in enumerate(inventory.endpoints[:100], start=1):
            method = str(endpoint.get("method", "GET"))
            step = {
                "step_id": f"plan-{index:03d}",
                "action": "passive_inventory_candidate" if endpoint.get("candidate_only") else "passive_inventory" if method == "UNKNOWN" else "authorization_matrix_candidate",
                "method": method,
                "path": str(endpoint.get("path", "")),
                "asset": str(endpoint.get("asset", "")),
                "authorization_required": True,
                "active": False if endpoint.get("candidate_only") else method in {"GET", "HEAD", "OPTIONS"} and scope.active_testing,
                "reason": "unverified local inventory candidate; not authorization and never an active request" if endpoint.get("candidate_only") else "deterministic plan from supplied evidence; no unbounded scan",
            }
            if endpoint.get("candidate_only"):
                step["confidence"] = endpoint.get("confidence", 0.2)
                step["provenance"] = endpoint.get("provenance", {})
            plan.append(step)
        return plan


class IDORBOLAEngine:
    """Test only controlled fixture identities/resources through an explicit scope."""

    def __init__(self, scope: ScopeDefinition) -> None:
        self.scope = scope

    @staticmethod
    def _policy_observation(fixture: Any, identity: str, method: str, url: str, resource_id: str) -> AuthorizationObservation:
        probe = getattr(fixture, "authorization_decision", None)
        if not callable(probe):
            return AuthorizationObservation(PolicyDecision.UNKNOWN, "policy_unknown", "application_policy_adapter_unavailable")
        try:
            observation = probe(identity, method, url, resource_id=resource_id)
        except Exception as exc:
            return AuthorizationObservation(PolicyDecision.UNKNOWN, "policy_unknown", f"policy_adapter_error:{type(exc).__name__}")
        if not isinstance(observation, AuthorizationObservation):
            return AuthorizationObservation(PolicyDecision.UNKNOWN, "policy_unknown", "policy_adapter_return_type_invalid")
        return observation

    def scan(self, fixture: DeterministicFixtureApp) -> tuple[list[Finding], list[TestCaseResult], list[str]]:
        findings: list[Finding] = []
        tests: list[TestCaseResult] = []
        limitations: list[str] = []
        client = ScopeEnforcedFixtureClient(fixture, self.scope)
        allowed_ids = [identity for identity in self.scope.permitted_identities if identity in fixture.identities]
        allowed_resources = [resource for resource in self.scope.permitted_resources if resource in fixture.resources]
        if len(allowed_ids) < 2 or not allowed_resources:
            return [], [], ["idor_matrix_requires_two_permitted_test_identities_and_one_resource"]
        for endpoint in fixture.endpoints:
            resource_ids = [rid for rid in allowed_resources if fixture.resources[rid].resource_type == endpoint.resource_type]
            for resource_id in resource_ids:
                resource = fixture.resources[resource_id]
                owner_first = [resource.owner_id] if resource.owner_id in allowed_ids else []
                candidate_ids = owner_first + [identity for identity in allowed_ids if identity not in owner_first]
                baseline_url = fixture.url_for(endpoint, resource_id, parent_id=resource.parent_id)
                baseline_identity = candidate_ids[0]
                baseline_policy = self._policy_observation(fixture, baseline_identity, endpoint.method, baseline_url, resource_id)
                for candidate in candidate_ids:
                    candidate_policy = self._policy_observation(fixture, candidate, endpoint.method, baseline_url, resource_id)
                    if candidate_policy.decision is PolicyDecision.ALLOW:
                        baseline_identity = candidate
                        baseline_policy = candidate_policy
                        break
                if baseline_policy.decision is not PolicyDecision.ALLOW:
                    limitations.append("authorized_baseline_policy_not_established")
                scope_decision, scope_reason = self.scope.authorize(
                    baseline_url, method=endpoint.method, mode="active", identity=baseline_identity, resource_id=resource_id
                )
                if scope_decision is not ScopeDecision.ALLOWED:
                    tests.append(TestCaseResult("T-" + stable_digest([endpoint.endpoint_id, baseline_identity, resource_id])[:12], endpoint.endpoint_id, baseline_identity, resource_id, "SKIPPED", "authorized baseline", scope_decision.value, scope_reason))
                    continue
                try:
                    baseline = client.request(baseline_identity, endpoint.method, baseline_url, resource_id=resource_id)
                except Exception as exc:
                    limitations.append(f"baseline_scope_boundary_failed:{type(exc).__name__}")
                    continue
                for identity in allowed_ids:
                    if identity == baseline_identity or fixture.identities[identity].role == "admin":
                        continue
                    test_id = "T-" + stable_digest([endpoint.endpoint_id, identity, resource_id])[:12]
                    decision, reason = self.scope.authorize(
                        baseline_url, method=endpoint.method, mode="active", identity=identity, resource_id=resource_id
                    )
                    if decision is not ScopeDecision.ALLOWED:
                        tests.append(TestCaseResult(test_id, endpoint.endpoint_id, identity, resource_id, "SKIPPED", "deny unauthorized identity", decision.value, reason))
                        continue
                    try:
                        observed = client.request(identity, endpoint.method, baseline_url, resource_id=resource_id)
                    except Exception as exc:
                        tests.append(TestCaseResult(test_id, endpoint.endpoint_id, identity, resource_id, "SKIPPED", "authorized scope", "scope_denied", str(exc)))
                        continue
                    policy = self._policy_observation(fixture, identity, endpoint.method, baseline_url, resource_id)
                    if observed.status in {401, 403, 404}:
                        tests.append(TestCaseResult(test_id, endpoint.endpoint_id, identity, resource_id, "SECURE", policy.decision.value, str(observed.status), "server denied the request"))
                        continue
                    if observed.status in {429, 500, 502, 503, 504}:
                        tests.append(TestCaseResult(test_id, endpoint.endpoint_id, identity, resource_id, "INCONCLUSIVE", policy.decision.value, str(observed.status), "fixture unavailable or rate-limited"))
                        continue
                    baseline_semantic = semantic_normalize(baseline.body)
                    observed_semantic = semantic_normalize(observed.body)
                    protected_keys = {"owner_id", "tenant_id", "sensitivity", "classification", "amount", "details"}
                    protected_content = (
                        observed_semantic == baseline_semantic
                        or bool(protected_keys.intersection(observed_semantic))
                        or str(observed_semantic.get("value", "")).startswith("controlled-")
                    )
                    if observed.status >= 300 or baseline.status >= 300 or not protected_content:
                        tests.append(TestCaseResult(test_id, endpoint.endpoint_id, identity, resource_id, "SECURE", policy.decision.value, str(observed.status), "no defensible protected-content authorization failure"))
                        continue
                    evidence = Evidence(
                        kind="controlled_authorization_comparison",
                        source="deterministic_fixture",
                        expected={"policy_decision": policy.decision.value, "status": "403_or_equivalent_denial", "body": baseline_semantic},
                        observed={"status": observed.status, "body": observed_semantic},
                        provenance={"fixture_source": "deterministic_local_boundary", "policy_model": policy.model, "policy_basis": policy.basis, "authorization_decision": policy.decision.value, "owner_identity": baseline_identity, "test_identity": identity, "resource_id": resource_id, "endpoint_id": endpoint.endpoint_id, "scope_reference": self.scope.authorization_reference},
                        request={"method": endpoint.method, "url": baseline_url, "identity": identity},
                        response={"status": observed.status, "body": observed.body, "headers": observed.headers},
                    )
                    if policy.decision is PolicyDecision.ALLOW:
                        tests.append(TestCaseResult(test_id, endpoint.endpoint_id, identity, resource_id, "SECURE", "ALLOW", str(observed.status), "explicit application policy permits this identity", evidence))
                        continue
                    confirmed = policy.decision is PolicyDecision.DENY
                    status = "VULNERABLE" if confirmed else "SUSPECTED"
                    title = ("Unauthorized" if confirmed else "Potential unauthorized") + f" {endpoint.resource_type} access by controlled identity"
                    finding = Finding(
                        finding_id=Finding.deterministic_id("IDOR/BOLA", fixture.base_url, endpoint.path_template, title),
                        vulnerability_class="IDOR/BOLA",
                        title=title,
                        asset=fixture.base_url,
                        endpoint=endpoint.path_template,
                        state=FindingState.OBSERVED,
                        severity=Severity.HIGH if resource.sensitivity in {"private", "restricted"} else Severity.MEDIUM,
                        confidence=0.98 if confirmed else 0.55,
                        scope_reference=self.scope.authorization_reference,
                        preconditions=["two controlled test identities", f"resource {resource_id}", f"policy decision {policy.decision.value}"],
                        identities=[baseline_identity, identity],
                        reproduction_steps=[
                            f"Use controlled identity {baseline_identity} to establish a permitted baseline for {baseline_url}.",
                            f"Repeat the same GET as controlled identity {identity} without changing authorization context.",
                            "Compare the explicit policy observation with the semantically normalized response; do not treat status alone as proof.",
                        ],
                        evidence=[evidence],
                        expected_behavior="The explicitly observed application policy denies access to this identity.",
                        observed_behavior=f"The server returned {observed.status} and protected content to {identity}.",
                        root_cause_hypothesis="The endpoint returned protected content despite the observed authorization decision." if confirmed else "Protected content was observed, but the application policy adapter did not establish whether this identity was allowed.",
                        impact="Controlled cross-identity access to protected resource data or an unresolved authorization concern.",
                        remediation="Reconcile the application policy decision with the resource/operation authorization check before reporting a confirmed issue.",
                        regression="Keep a matrix test covering owner, authorized non-owner, and unauthorized identities for the exact operation.",
                        limitations=["Confirmed or suspected only against the supplied deterministic local adapter; no external target was accessed."],
                    )
                    if confirmed:
                        finding.transition(FindingState.HYPOTHESIS)
                        finding.transition(FindingState.VALIDATION_PENDING)
                        finding.transition(FindingState.CONFIRMED)
                        finding.transition(FindingState.REPORT_READY)
                    else:
                        finding.transition(FindingState.HYPOTHESIS)
                    if finding.finding_id not in {item.finding_id for item in findings}:
                        findings.append(finding)
                    tests.append(TestCaseResult(test_id, endpoint.endpoint_id, identity, resource_id, status, policy.decision.value, str(observed.status), "explicit deny with protected content" if confirmed else "protected content observed while policy remains unknown", evidence))
        if not tests:
            limitations.append("no_eligible_idor_cases")
        return findings, tests, limitations


class StaticSecurityAnalyzer:
    """Advisory source analysis; findings remain hypotheses until validated."""

    PATTERNS = (
        (re.compile(r"subprocess\.[A-Za-z_]+\([^\n]{0,500}shell\s*=\s*True", re.I), "OS command injection risk", "Injection", Severity.HIGH),
        (re.compile(r"(?:SELECT|UPDATE|DELETE|INSERT)[^\n]{0,180}(?:f['\"]|\.format\(|%\s*\w+)", re.I), "Potential SQL injection sink", "Injection", Severity.HIGH),
        (re.compile(r"(?:pickle\.loads|yaml\.load|marshal\.loads)\s*\(", re.I), "Unsafe deserialization sink", "Unsafe Deserialization", Severity.HIGH),
        (re.compile(r"(?:requests|httpx|urllib)\.[A-Za-z_]+\([^\n]{0,300}(?:request|query|param|user|input|url)", re.I), "Potential SSRF-capable URL sink", "SSRF", Severity.HIGH),
        (re.compile(r"(?:innerHTML|outerHTML|document\.write)\s*=", re.I), "Potential client-side HTML injection sink", "Cross-Site Scripting", Severity.MEDIUM),
        (re.compile(r"(?:open|Path)\([^\n]{0,220}(?:request|query|param|user|input)", re.I), "Potential path traversal file sink", "Path Traversal", Severity.HIGH),
        (re.compile(r"jwt\.decode\([^\n]{0,300}(?:verify_signature\s*=\s*False|algorithms\s*=\s*\[\s*['\"]none['\"])", re.I), "Weak JWT validation configuration", "Authentication", Severity.HIGH),
        (re.compile(r"(?:redirect|location)\s*\([^\n]{0,220}(?:request|query|param|next|return_url|url)", re.I), "Potential open redirect sink", "Open Redirect", Severity.MEDIUM),
        (re.compile(r"allow_origins\s*=\s*\[[^\]]*['\"]\*['\"][^\]]*\].{0,200}allow_credentials\s*=\s*True", re.I | re.S), "Permissive CORS with credentials", "Security Misconfiguration", Severity.MEDIUM),
        (re.compile(r"authorization\s*:\s*bearer\s+[A-Za-z0-9._~-]{8,}", re.I), "Credential-shaped authorization value in source", "Sensitive Data Exposure", Severity.HIGH),
    )

    def analyze(self, documents: Mapping[str, str], scope_reference: str) -> list[Finding]:
        findings: list[Finding] = []
        for source, text in documents.items():
            if not isinstance(source, str) or not isinstance(text, str) or len(text.encode("utf-8")) > 512_000:
                continue
            for pattern, title, vuln_class, severity in self.PATTERNS:
                match = pattern.search(text)
                if not match:
                    continue
                line = text[:match.start()].count("\n") + 1
                evidence = Evidence(
                    kind="static_source_observation",
                    source=source,
                    expected={"condition": "security-sensitive sink is guarded or parameterized"},
                    observed={"line": line, "snippet": text[max(0, match.start() - 120):match.end() + 120]},
                    provenance={"mode": "static", "source": source, "line": line},
                )
                finding = Finding(
                    finding_id=Finding.deterministic_id(vuln_class, source, str(line), title),
                    vulnerability_class=vuln_class,
                    title=title,
                    asset="local-source",
                    endpoint=source,
                    state=FindingState.OBSERVED,
                    severity=severity,
                    confidence=0.45,
                    scope_reference=scope_reference,
                    reproduction_steps=[f"Review {source} at line {line} in the authorized local source artifact."],
                    evidence=[evidence],
                    expected_behavior="The sink is constrained by parameterization, validation, and explicit authorization.",
                    observed_behavior="A suspicious source pattern was observed; runtime exploitability was not tested.",
                    root_cause_hypothesis="Static pattern may indicate an unsafe sink, but context and reachability are unresolved.",
                    impact="Potential security weakness requiring authorized validation.",
                    remediation="Review data flow, parameterization, authorization, and tests before changing production behavior.",
                    limitations=["Static observation only; not a confirmed vulnerability and no target code was executed."],
                )
                finding.transition(FindingState.HYPOTHESIS)
                findings.append(finding)
        return findings
