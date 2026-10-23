"""Independent local validation corpus for the bounded Bug Bounty engine.

The corpus is a research harness, not a target connector.  Each fixture exposes
only observable identities, resources, endpoints, and responses to the engine.
The expected outcome is held by a separate oracle object owned by this module's
harness and is never passed to :class:`IDORBOLAEngine`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Sequence
from urllib.parse import urlparse

from private_agent.bugbounty.engines import IDORBOLAEngine
from private_agent.bugbounty.fixtures import (
    FixtureEndpoint,
    FixtureIdentity,
    FixtureResource,
    FixtureResponse,
    ScopeEnforcedFixtureClient,
)
from private_agent.bugbounty.models import (
    AuthorizationObservation,
    Evidence,
    Finding,
    FindingState,
    PolicyDecision,
    ScopeDefinition,
    ScopeDecision,
    validate_confirmed_evidence,
)

EXPECTED_OUTCOMES = frozenset({"VULNERABLE", "SUSPECTED", "SAFE", "INCONCLUSIVE", "ERROR"})
DETERMINATE_OUTCOMES = frozenset({"VULNERABLE", "SAFE"})
Outcome = Literal["VULNERABLE", "SUSPECTED", "SAFE", "INCONCLUSIVE", "ERROR"]


@dataclass(frozen=True)
class CorpusOracle:
    """Ground truth owned by the harness and never exposed to the engine."""

    case_id: str
    split: Literal["train", "validation", "held_out"]
    expected_outcome: Outcome
    vulnerability_class: str
    rationale: str


class IndependentAuthorizationFixture:
    """Black-box local application model with neutral case identity.

    The engine sees this object's normal application surface only.  It does not
    receive a ``CorpusOracle`` or a label-bearing fixture name.  ``_policy`` is
    an internal test-double implementation detail, not ground truth.
    """

    base_url = "https://fixture.local"

    def __init__(self, policy: str, *, endpoint_kind: str = "order", path_variant: str = "v1") -> None:
        self._policy = policy
        self.identities = {
            "alice": FixtureIdentity("alice", "user", "tenant-a"),
            "bob": FixtureIdentity("bob", "user", "tenant-b"),
            "admin": FixtureIdentity("admin", "admin", "tenant-a"),
            "anonymous": FixtureIdentity("anonymous", "anonymous", ""),
            "expired-token": FixtureIdentity("expired-token", "user", "tenant-b"),
        }
        self.resources = {
            "order-alice": FixtureResource("order-alice", "order", "alice", "tenant-a", sensitivity="private"),
            "order-bob": FixtureResource("order-bob", "order", "bob", "tenant-b", sensitivity="private"),
            "comment-bob": FixtureResource("comment-bob", "comment", "bob", "tenant-b", parent_id="order-alice", sensitivity="private"),
            "report-admin": FixtureResource("report-admin", "admin_report", "admin", "tenant-a", sensitivity="restricted"),
            "record-201": FixtureResource("record-201", "record", "alice", "tenant-a", sensitivity="private"),
        }
        path_root = "/api/" + path_variant
        if endpoint_kind == "admin":
            self.endpoints = (FixtureEndpoint("admin-report-read", "GET", f"{path_root}/admin/reports/{{report_id}}", "admin_report", "admin", "Read a role-protected report."),)
        elif endpoint_kind == "nested":
            self.endpoints = (FixtureEndpoint("nested-comment-read", "GET", f"{path_root}/orders/{{order_id}}/comments/{{comment_id}}", "comment", "owner_and_parent", "Read a nested comment."),)
        elif endpoint_kind == "record":
            self.endpoints = (FixtureEndpoint("record-read", "GET", f"{path_root}/records/{{record_id}}", "record", "owner", "Read a record with a changed identifier."),)
        else:
            self.endpoints = (FixtureEndpoint("order-read", "GET", f"{path_root}/orders/{{order_id}}", "order", "owner", "Read an object owned by an identity."),)

    def inventory(self) -> list[dict[str, str]]:
        return [endpoint.to_dict() for endpoint in self.endpoints]

    def url_for(self, endpoint: FixtureEndpoint, resource_id: str, *, parent_id: str = "") -> str:
        return (
            self.base_url
            + endpoint.path_template.replace("{order_id}", parent_id or resource_id)
            .replace("{comment_id}", resource_id)
            .replace("{report_id}", resource_id)
            .replace("{record_id}", resource_id)
        )

    def _find_endpoint(self, path: str, method: str) -> FixtureEndpoint | None:
        for endpoint in self.endpoints:
            if endpoint.method != method.upper():
                continue
            expected = endpoint.path_template.strip("/").split("/")
            actual = path.strip("/").split("/")
            if len(expected) != len(actual):
                continue
            if all(left.startswith("{") and left.endswith("}") or left == right for left, right in zip(expected, actual)):
                return endpoint
        return None

    @staticmethod
    def _ids(endpoint: FixtureEndpoint, path: str) -> tuple[str, str]:
        parts = path.strip("/").split("/")
        if endpoint.endpoint_id == "nested-comment-read":
            return parts[-1], parts[-3]
        return parts[-1], ""

    @staticmethod
    def _authorized(identity: FixtureIdentity, endpoint: FixtureEndpoint, resource: FixtureResource, parent: FixtureResource | None) -> bool:
        if endpoint.access_level == "admin":
            return identity.role == "admin"
        if endpoint.access_level == "owner":
            return identity.identity_id == resource.owner_id or identity.role == "admin"
        if endpoint.access_level == "owner_and_parent":
            return (parent is not None and parent.owner_id == identity.identity_id) or identity.role == "admin"
        return False

    def authorization_decision(self, identity_id: str, method: str, url: str, *, resource_id: str) -> AuthorizationObservation:
        if self._policy in {"p11", "p16"}:
            return AuthorizationObservation(PolicyDecision.UNKNOWN, "phase19_local_policy", "policy_observation_unavailable")
        identity = self.identities.get(identity_id)
        parsed = urlparse(url)
        endpoint = self._find_endpoint(parsed.path, method)
        resource = self.resources.get(resource_id)
        if identity is None or endpoint is None or resource is None:
            return AuthorizationObservation(PolicyDecision.DENY, "phase19_local_policy", "credential_or_resource_boundary_denied")
        _, parent_id = self._ids(endpoint, parsed.path)
        parent = self.resources.get(parent_id) if parent_id else None
        allowed = self._authorized(identity, endpoint, resource, parent)
        if self._policy in {"p07", "p15"} and identity_id != resource.owner_id and identity.role != "admin":
            allowed = True
        return AuthorizationObservation(
            PolicyDecision.ALLOW if allowed else PolicyDecision.DENY,
            "phase19_local_policy",
            "explicit_local_authorization_rule",
        )

    def request(self, identity_id: str, method: str, url: str) -> FixtureResponse:
        identity = self.identities.get(identity_id)
        if identity is None:
            return FixtureResponse(401, {"error": "unknown_identity"}, {"content-type": "application/json"})
        if self._policy == "p16":
            raise RuntimeError("local_fixture_observation_error")
        if self._policy == "p11":
            return FixtureResponse(503, {"error": "fixture_unavailable"}, {"retry-after": "1"})
        if self._policy == "p09" and identity_id == "anonymous":
            return FixtureResponse(401, {"error": "authentication_required"}, {"content-type": "application/json"})
        if self._policy == "p10" and identity_id == "expired-token":
            return FixtureResponse(401, {"error": "credential_expired"}, {"content-type": "application/json"})
        parsed = urlparse(url)
        endpoint = self._find_endpoint(parsed.path, method)
        if endpoint is None:
            return FixtureResponse(404, {"error": "not_found"}, {"content-type": "application/json"})
        resource_id, parent_id = self._ids(endpoint, parsed.path)
        resource = self.resources.get(resource_id)
        parent = self.resources.get(parent_id) if parent_id else None
        if resource is None or resource.resource_type != endpoint.resource_type:
            return FixtureResponse(404, {"error": "resource_not_found"}, {"content-type": "application/json"})
        allowed = self._authorized(identity, endpoint, resource, parent)
        if self._policy in {"p01", "p06", "p12", "p13"}:
            allowed = allowed or (endpoint.endpoint_id in {"order-read", "record-read"} and identity_id == "bob" and resource.owner_id == "alice")
        elif self._policy in {"p02"}:
            allowed = allowed or (endpoint.endpoint_id == "admin-report-read" and identity.role != "admin")
        elif self._policy in {"p03", "p17"}:
            allowed = allowed or (endpoint.endpoint_id == "nested-comment-read" and identity_id == "bob" and parent is not None and parent.owner_id == "alice")
        if not allowed:
            if self._policy == "p08":
                return FixtureResponse(404, {"error": "not_found"}, {"content-type": "application/json"})
            if self._policy in {"p07", "p15"} and identity_id != resource.owner_id and identity.role != "admin":
                return FixtureResponse(200, {"resource_id": resource.resource_id, "value": "public-summary"}, {"content-type": "application/json"})
            return FixtureResponse(403, {"error": "forbidden"}, {"content-type": "application/json"})
        return FixtureResponse(
            200,
            {
                "resource_id": resource.resource_id,
                "resource_type": resource.resource_type,
                "owner_id": resource.owner_id,
                "tenant_id": resource.tenant_id,
                "sensitivity": resource.sensitivity,
                "value": f"controlled-{resource.resource_id}",
            },
            {"content-type": "application/json", "x-fixture": "independent-local"},
        )


@dataclass(frozen=True)
class CorpusSample:
    """A fixture plus a separate oracle; only ``fixture`` enters the engine."""

    fixture: IndependentAuthorizationFixture
    scope: ScopeDefinition
    oracle: CorpusOracle


def _scope_for(fixture: IndependentAuthorizationFixture) -> ScopeDefinition:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ScopeDefinition.from_dict(
        {
            "program_id": "phase19-independent-local",
            "authorization_reference": "phase19-local-authorization",
            "issued_at": now.isoformat(),
            "expires_at": "2099-01-01T00:00:00+00:00",
            "assets": [{"asset_id": "independent-fixture", "host": "fixture.local", "path_prefix": "/", "methods": ["GET"], "active_allowed": True}],
            "permitted_identities": list(fixture.identities),
            "permitted_resources": list(fixture.resources),
            "permitted_methods": ["GET"],
            "prohibited_methods": ["POST", "PUT", "PATCH", "DELETE"],
            "active_testing": True,
        }
    )


def _sample(case_id: str, split: Literal["train", "validation", "held_out"], policy: str, expected: Outcome, vulnerability_class: str, *, endpoint_kind: str = "order", path_variant: str = "v1", rationale: str = "") -> CorpusSample:
    fixture = IndependentAuthorizationFixture(policy, endpoint_kind=endpoint_kind, path_variant=path_variant)
    return CorpusSample(fixture, _scope_for(fixture), CorpusOracle(case_id, split, expected, vulnerability_class, rationale or expected.lower()))


def build_independent_corpus() -> tuple[CorpusSample, ...]:
    """Return a fixed corpus with no oracle fields on the fixture objects."""
    return (
        _sample("T-001", "train", "p01", "VULNERABLE", "horizontal-IDOR"),
        _sample("T-002", "train", "p02", "VULNERABLE", "vertical-BOLA", endpoint_kind="admin"),
        _sample("T-003", "train", "p03", "VULNERABLE", "nested-BOLA", endpoint_kind="nested"),
        _sample("T-004", "train", "p04", "SAFE", "object-authorization"),
        _sample("T-005", "train", "p05", "SAFE", "role-authorization", endpoint_kind="admin"),
        _sample("T-006", "train", "p06", "VULNERABLE", "same-status-protected-content"),
        _sample("T-007", "train", "p07", "SAFE", "same-status-public-summary"),
        _sample("T-008", "train", "p08", "SAFE", "status-only-negative-control"),
        _sample("V-001", "validation", "p09", "SAFE", "missing-authentication"),
        _sample("V-002", "validation", "p10", "SAFE", "expired-credential"),
        _sample("V-003", "validation", "p11", "INCONCLUSIVE", "insufficient-evidence"),
        _sample("V-004", "validation", "p12", "VULNERABLE", "cross-tenant-IDOR"),
        _sample("H-001", "held_out", "p13", "VULNERABLE", "changed-resource-ID", endpoint_kind="record"),
        _sample("H-002", "held_out", "p14", "SAFE", "changed-path-safe-control", path_variant="v2"),
        _sample("H-003", "held_out", "p15", "SAFE", "response-format-safe-control"),
        _sample("H-004", "held_out", "p16", "ERROR", "observation-error"),
        _sample("H-005", "held_out", "p17", "VULNERABLE", "nested-path-authorization", endpoint_kind="nested", path_variant="v3"),
        _sample("H-006", "held_out", "p18", "SAFE", "cross-tenant-safe-control"),
    )


def _classify(findings: Sequence[Finding], tests: Sequence[Any], limitations: Sequence[str]) -> Outcome:
    if any(item.state in {FindingState.CONFIRMED, FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED} for item in findings):
        return "VULNERABLE"
    if findings or any(getattr(item, "status", "") == "SUSPECTED" for item in tests):
        return "SUSPECTED"
    if any(getattr(item, "status", "") == "INCONCLUSIVE" for item in tests):
        return "INCONCLUSIVE"
    if any(str(item).startswith("baseline_scope_boundary_failed:") for item in limitations):
        return "ERROR"
    return "SAFE"


def _case_record(sample: CorpusSample) -> dict[str, Any]:
    findings, tests, limitations = IDORBOLAEngine(sample.scope).scan(sample.fixture)
    predicted = _classify(findings, tests, limitations)
    evidence_errors = sorted({error for finding in findings for error in validate_confirmed_evidence(finding)})
    status_only_proof = bool(findings) and not any(
        evidence.kind == "controlled_authorization_comparison" and isinstance(evidence.expected, dict) and "body" in evidence.expected and isinstance(evidence.observed, dict) and "body" in evidence.observed
        for finding in findings
        for evidence in finding.evidence
    )
    evidence_quality = "observed_authorization_comparison" if findings and not evidence_errors and not status_only_proof else ("invalid" if findings else "none")
    return {
        "case_id": sample.oracle.case_id,
        "split": sample.oracle.split,
        "expected_outcome": sample.oracle.expected_outcome,
        "predicted_outcome": predicted,
        "expected_class": sample.oracle.vulnerability_class,
        "finding_classes": sorted({finding.vulnerability_class for finding in findings}),
        "finding_count": len(findings),
        "test_count": len(tests),
        "test_statuses": sorted(str(getattr(item, "status", "")) for item in tests),
        "evidence_count": sum(len(finding.evidence) for finding in findings),
        "evidence_quality": evidence_quality,
        "evidence_errors": evidence_errors,
        "status_only_proof": status_only_proof,
        "limitations": sorted(set(str(item) for item in limitations)),
    }


def _metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for item in records if item["expected_outcome"] == "VULNERABLE" and item["predicted_outcome"] == "VULNERABLE")
    fp = sum(1 for item in records if item["expected_outcome"] == "SAFE" and item["predicted_outcome"] == "VULNERABLE")
    fn = sum(1 for item in records if item["expected_outcome"] == "VULNERABLE" and item["predicted_outcome"] == "SAFE")
    tn = sum(1 for item in records if item["expected_outcome"] == "SAFE" and item["predicted_outcome"] == "SAFE")
    inconclusive = sum(1 for item in records if item["predicted_outcome"] == "INCONCLUSIVE")
    errors = sum(1 for item in records if item["predicted_outcome"] == "ERROR")
    suspected = sum(1 for item in records if item["predicted_outcome"] == "SUSPECTED")
    overclaims = sum(1 for item in records if item["expected_outcome"] in {"INCONCLUSIVE", "ERROR"} and item["predicted_outcome"] == "VULNERABLE")
    unresolved = sum(1 for item in records if item["expected_outcome"] == "VULNERABLE" and item["predicted_outcome"] in {"INCONCLUSIVE", "ERROR"})
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = tp / precision_denominator if precision_denominator else None
    recall = tp / recall_denominator if recall_denominator else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    return {
        "total": len(records),
        "determinate_total": tp + fp + fn + tn,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "inconclusive": inconclusive,
        "errors": errors,
        "suspected": suspected,
        "overclaims": overclaims,
        "unresolved_expected_vulnerable": unresolved,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inconclusive_excluded_from_metrics": True,
    }


def evaluate_independent_corpus(samples: Sequence[CorpusSample] | None = None) -> dict[str, Any]:
    corpus = tuple(samples or build_independent_corpus())
    records = [_case_record(sample) for sample in corpus]
    splits = {split: _metrics([item for item in records if item["split"] == split]) for split in ("train", "validation", "held_out")}
    classes: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        classes.setdefault(item["expected_class"], []).append(item)
    per_class = {name: _metrics(group) for name, group in sorted(classes.items())}
    return {
        "corpus_version": "phase19-independent-v1",
        "ground_truth_accessible_to_engine": False,
        "external_targets_contacted": 0,
        "split_policy": "train/validation/held_out are fixed before execution; held_out is not used to tune logic",
        "limitations": [
            "Synthetic local fixtures are not representative of arbitrary external applications.",
            "Metrics are case-level and apply only to the explicit harness oracle.",
            "Inconclusive and error cases are reported and excluded from precision/recall/F1.",
        ],
        "cases": records,
        "overall": _metrics(records),
        "splits": splits,
        "per_class": per_class,
    }
