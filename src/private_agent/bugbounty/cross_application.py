"""Cross-application local generalization harness for the IDOR/BOLA engine.

The applications in this module are independent deterministic in-memory models,
not renamed instances of the original fixture.  They share only the narrow
observable adapter contract consumed by the existing engine.  Oracle metadata is
kept in ``CrossApplicationOracle`` and is never attached to an application.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Sequence
from urllib.parse import urlparse

from private_agent.bugbounty.engines import IDORBOLAEngine
from private_agent.bugbounty.fixtures import FixtureResponse
from private_agent.bugbounty.models import AuthorizationObservation, Finding, FindingState, PolicyDecision, ScopeDefinition
from private_agent.bugbounty.validation import validate_confirmed_evidence

Outcome = Literal["VULNERABLE", "SUSPECTED", "SAFE", "INCONCLUSIVE", "ERROR"]


@dataclass(frozen=True)
class ApplicationIdentity:
    identity_id: str
    role: str
    tenant_id: str


@dataclass(frozen=True)
class ApplicationResource:
    resource_id: str
    resource_type: str
    owner_id: str
    tenant_id: str
    parent_id: str = ""
    sensitivity: str = "private"


@dataclass(frozen=True)
class ApplicationEndpoint:
    endpoint_id: str
    method: str
    path_template: str
    resource_type: str
    access_level: str
    description: str


@dataclass(frozen=True)
class CrossApplicationOracle:
    case_id: str
    application_id: str
    split: Literal["train", "validation", "held_out"]
    expected_outcome: Outcome
    vulnerability_class: str
    rationale: str


@dataclass(frozen=True)
class CrossApplicationSample:
    application_id: str
    application: Any
    scope: ScopeDefinition
    oracle: CrossApplicationOracle


class LedgerApplication:
    """Resource-layer ownership model with a 404 denial convention."""

    base_url = "https://ledger.local"

    def __init__(self, rule: int) -> None:
        self._rule = rule
        self.identities = {
            "alice": ApplicationIdentity("alice", "user", "tenant-a"),
            "bob": ApplicationIdentity("bob", "user", "tenant-b"),
            "anonymous": ApplicationIdentity("anonymous", "anonymous", ""),
            "expired": ApplicationIdentity("expired", "user", "tenant-b"),
        }
        self.resources = {
            "invoice-a": ApplicationResource("invoice-a", "invoice", "alice", "tenant-a"),
            "invoice-b": ApplicationResource("invoice-b", "invoice", "bob", "tenant-b"),
        }
        self.endpoints = (
            ApplicationEndpoint("invoice-read", "GET", "/v1/invoices/{invoice_id}", "invoice", "owner", "Read an invoice."),
        )

    def url_for(self, endpoint: ApplicationEndpoint, resource_id: str, *, parent_id: str = "") -> str:
        return self.base_url + endpoint.path_template.replace("{invoice_id}", resource_id)

    def authorization_decision(self, identity_id: str, method: str, url: str, *, resource_id: str) -> AuthorizationObservation:
        identity = self.identities.get(identity_id)
        resource = self.resources.get(resource_id)
        if identity is None or resource is None or identity_id in {"anonymous", "expired"}:
            return AuthorizationObservation(PolicyDecision.DENY, "ledger_resource_owner", "credential_or_resource_denied")
        allowed = identity.identity_id == resource.owner_id
        return AuthorizationObservation(
            PolicyDecision.ALLOW if allowed else PolicyDecision.DENY,
            "ledger_resource_owner",
            "resource_owner_check",
        )

    def request(self, identity_id: str, method: str, url: str) -> FixtureResponse:
        identity = self.identities.get(identity_id)
        if identity is None or identity_id in {"anonymous", "expired"}:
            return FixtureResponse(401, {"error": "authentication_required"}, {"content-type": "application/json"})
        resource_id = urlparse(url).path.rstrip("/").split("/")[-1]
        resource = self.resources.get(resource_id)
        if resource is None:
            return FixtureResponse(404, {"error": "not_found"}, {"content-type": "application/json"})
        allowed = identity.identity_id == resource.owner_id
        if self._rule == 1 and identity.identity_id == "bob" and resource.owner_id == "alice":
            allowed = True
        if not allowed:
            return FixtureResponse(404, {"error": "not_found"}, {"content-type": "application/json"})
        return FixtureResponse(
            200,
            {"invoice_id": resource.resource_id, "owner_id": resource.owner_id, "tenant_id": resource.tenant_id, "amount": "protected"},
            {"content-type": "application/json", "x-app": "ledger"},
        )


class ConsoleApplication:
    """Operation-layer role model with explicit 403 denial and 401 credentials."""

    base_url = "https://console.local"

    def __init__(self, rule: int) -> None:
        self._rule = rule
        self.identities = {
            "root": ApplicationIdentity("root", "admin", "tenant-a"),
            "analyst": ApplicationIdentity("analyst", "analyst", "tenant-a"),
            "guest": ApplicationIdentity("guest", "anonymous", ""),
            "expired": ApplicationIdentity("expired", "analyst", "tenant-a"),
        }
        self.resources = {
            "audit-7": ApplicationResource("audit-7", "audit_report", "root", "tenant-a", sensitivity="restricted"),
        }
        self.endpoints = (
            ApplicationEndpoint("audit-read", "GET", "/console/audits/{audit_id}", "audit_report", "admin", "Read an administrator audit report."),
        )

    def url_for(self, endpoint: ApplicationEndpoint, resource_id: str, *, parent_id: str = "") -> str:
        return self.base_url + endpoint.path_template.replace("{audit_id}", resource_id)

    def authorization_decision(self, identity_id: str, method: str, url: str, *, resource_id: str) -> AuthorizationObservation:
        identity = self.identities.get(identity_id)
        resource = self.resources.get(resource_id)
        if identity is None or resource is None or identity_id in {"guest", "expired"}:
            return AuthorizationObservation(PolicyDecision.DENY, "console_operation_role", "credential_or_resource_denied")
        allowed = identity.role == "admin" or (self._rule == 2 and identity.role == "analyst" and method.upper() == "GET")
        return AuthorizationObservation(
            PolicyDecision.ALLOW if allowed else PolicyDecision.DENY,
            "console_operation_role",
            "admin_operation_role_check",
        )

    def request(self, identity_id: str, method: str, url: str) -> FixtureResponse:
        identity = self.identities.get(identity_id)
        if identity is None or identity_id in {"guest", "expired"}:
            return FixtureResponse(401, {"error": "invalid_credentials"}, {"content-type": "application/json"})
        resource_id = urlparse(url).path.rstrip("/").split("/")[-1]
        resource = self.resources.get(resource_id)
        if resource is None:
            return FixtureResponse(404, {"error": "not_found"}, {"content-type": "application/json"})
        allowed = identity.role == "admin"
        if self._rule == 1 and identity.role != "admin":
            allowed = True
        if self._rule == 2 and identity.role == "analyst" and method.upper() == "GET":
            allowed = True
        if not allowed:
            return FixtureResponse(403, {"error": "role_required"}, {"content-type": "application/json"})
        return FixtureResponse(
            200,
            {"audit_id": resource.resource_id, "tenant_id": resource.tenant_id, "classification": "restricted"},
            {"content-type": "application/json", "x-app": "console"},
        )


class TenantVaultApplication:
    """Tenant-layer policy: same-tenant members may read; cross-tenant users may not."""

    base_url = "https://vault.local"

    def __init__(self, rule: int) -> None:
        self._rule = rule
        self.identities = {
            "alice": ApplicationIdentity("alice", "member", "tenant-a"),
            "carol": ApplicationIdentity("carol", "member", "tenant-a"),
            "bob": ApplicationIdentity("bob", "member", "tenant-b"),
            "anonymous": ApplicationIdentity("anonymous", "anonymous", ""),
            "expired": ApplicationIdentity("expired", "member", "tenant-b"),
        }
        self.resources = {
            "vault-a": ApplicationResource("vault-a", "vault_document", "alice", "tenant-a", sensitivity="restricted"),
        }
        self.endpoints = (
            ApplicationEndpoint("vault-read", "GET", "/vault/documents/{document_id}", "vault_document", "owner", "Read a tenant document."),
        )

    def url_for(self, endpoint: ApplicationEndpoint, resource_id: str, *, parent_id: str = "") -> str:
        return self.base_url + endpoint.path_template.replace("{document_id}", resource_id)

    def authorization_decision(self, identity_id: str, method: str, url: str, *, resource_id: str) -> AuthorizationObservation:
        identity = self.identities.get(identity_id)
        resource = self.resources.get(resource_id)
        if identity is None or resource is None or identity_id in {"anonymous", "expired"}:
            return AuthorizationObservation(PolicyDecision.DENY, "vault_tenant_membership", "credential_or_resource_denied")
        allowed = identity.tenant_id == resource.tenant_id
        return AuthorizationObservation(
            PolicyDecision.ALLOW if allowed else PolicyDecision.DENY,
            "vault_tenant_membership",
            "tenant_membership_check",
        )

    def request(self, identity_id: str, method: str, url: str) -> FixtureResponse:
        identity = self.identities.get(identity_id)
        if identity is None or identity_id in {"anonymous", "expired"}:
            return FixtureResponse(401, {"error": "credential_required"}, {"content-type": "application/json"})
        resource_id = urlparse(url).path.rstrip("/").split("/")[-1]
        resource = self.resources.get(resource_id)
        if resource is None:
            return FixtureResponse(404, {"error": "missing"}, {"content-type": "application/json"})
        allowed = identity.tenant_id == resource.tenant_id
        if self._rule == 1 and identity.identity_id == "bob":
            allowed = True
        if not allowed:
            return FixtureResponse(404, {"error": "missing"}, {"content-type": "application/json"})
        return FixtureResponse(
            200,
            {"document_id": resource.resource_id, "tenant_id": resource.tenant_id, "classification": "restricted", "value": "tenant-protected"},
            {"content-type": "application/json", "x-app": "vault"},
        )


class ProjectApplication:
    """Membership-based nested-resource model, independent of owner checks."""

    base_url = "https://projects.local"

    def __init__(self, rule: int) -> None:
        self._rule = rule
        self.identities = {
            "alice": ApplicationIdentity("alice", "member", "tenant-a"),
            "carol": ApplicationIdentity("carol", "member", "tenant-a"),
            "bob": ApplicationIdentity("bob", "member", "tenant-b"),
            "anonymous": ApplicationIdentity("anonymous", "anonymous", ""),
        }
        self.resources = {
            "project-a": ApplicationResource("project-a", "project", "alice", "tenant-a", sensitivity="restricted"),
            "task-42": ApplicationResource("task-42", "task", "carol", "tenant-a", parent_id="project-a", sensitivity="restricted"),
        }
        self.endpoints = (
            ApplicationEndpoint("task-read", "GET", "/projects/{project_id}/tasks/{task_id}", "task", "owner_and_parent", "Read a project task."),
        )

    def url_for(self, endpoint: ApplicationEndpoint, resource_id: str, *, parent_id: str = "") -> str:
        return self.base_url + endpoint.path_template.replace("{project_id}", parent_id).replace("{task_id}", resource_id)

    def authorization_decision(self, identity_id: str, method: str, url: str, *, resource_id: str) -> AuthorizationObservation:
        identity = self.identities.get(identity_id)
        resource = self.resources.get(resource_id)
        if identity is None or resource is None or identity_id == "anonymous":
            return AuthorizationObservation(PolicyDecision.DENY, "projects_membership", "credential_or_resource_denied")
        allowed = identity.identity_id in {"alice", "carol"}
        return AuthorizationObservation(
            PolicyDecision.ALLOW if allowed else PolicyDecision.DENY,
            "projects_membership",
            "project_membership_check",
        )

    def request(self, identity_id: str, method: str, url: str) -> FixtureResponse:
        identity = self.identities.get(identity_id)
        if identity is None or identity_id == "anonymous":
            return FixtureResponse(401, {"error": "login_required"}, {"content-type": "application/json"})
        parts = urlparse(url).path.strip("/").split("/")
        resource_id = parts[-1]
        parent_id = parts[-3] if len(parts) >= 3 else ""
        resource = self.resources.get(resource_id)
        parent = self.resources.get(parent_id)
        if resource is None or parent is None or resource.parent_id != parent.resource_id:
            return FixtureResponse(404, {"error": "missing"}, {"content-type": "application/json"})
        members = {"alice", "carol"}
        allowed = identity.identity_id in members
        if self._rule == 1 and identity.role == "member":
            allowed = True
        if not allowed:
            return FixtureResponse(403, {"error": "project_membership_required"}, {"content-type": "application/json"})
        return FixtureResponse(
            200,
            {"task_id": resource.resource_id, "project_id": parent.resource_id, "tenant_id": parent.tenant_id, "details": "member-protected"},
            {"content-type": "application/json", "x-app": "projects"},
        )


class ReportsAvailabilityApplication:
    """Ambiguous service that cannot produce a usable authorization observation."""

    base_url = "https://reports.local"

    def __init__(self) -> None:
        self.identities = {
            "alice": ApplicationIdentity("alice", "member", "tenant-a"),
            "bob": ApplicationIdentity("bob", "member", "tenant-b"),
        }
        self.resources = {
            "report-1": ApplicationResource("report-1", "report", "alice", "tenant-a", sensitivity="restricted"),
        }
        self.endpoints = (
            ApplicationEndpoint("report-read", "GET", "/reports/{report_id}", "report", "owner", "Read a report."),
        )

    def url_for(self, endpoint: ApplicationEndpoint, resource_id: str, *, parent_id: str = "") -> str:
        return self.base_url + endpoint.path_template.replace("{report_id}", resource_id)

    def authorization_decision(self, identity_id: str, method: str, url: str, *, resource_id: str) -> AuthorizationObservation:
        return AuthorizationObservation(PolicyDecision.UNKNOWN, "reports_unavailable", "service_unavailable_before_policy_decision")

    def request(self, identity_id: str, method: str, url: str) -> FixtureResponse:
        return FixtureResponse(503, {"error": "temporarily_unavailable"}, {"retry-after": "1"})


def _scope_for(application: Any) -> ScopeDefinition:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    host = urlparse(application.base_url).hostname or "local.invalid"
    return ScopeDefinition.from_dict(
        {
            "program_id": "phase20-local-cross-application",
            "authorization_reference": "phase20-local-authorization",
            "issued_at": now.isoformat(),
            "expires_at": "2099-01-01T00:00:00+00:00",
            "assets": [{"asset_id": host, "host": host, "path_prefix": "/", "methods": ["GET"], "active_allowed": True}],
            "permitted_identities": list(application.identities),
            "permitted_resources": list(application.resources),
            "permitted_methods": ["GET"],
            "prohibited_methods": ["POST", "PUT", "PATCH", "DELETE"],
            "active_testing": True,
        }
    )


def _sample(application_id: str, split: Literal["train", "validation", "held_out"], case_id: str, application: Any, expected: Outcome, vulnerability_class: str, rationale: str) -> CrossApplicationSample:
    return CrossApplicationSample(application_id, application, _scope_for(application), CrossApplicationOracle(case_id, application_id, split, expected, vulnerability_class, rationale))


def build_cross_application_corpus() -> tuple[CrossApplicationSample, ...]:
    """Build cases grouped by independent application architecture."""
    return (
        _sample("app-ledger", "train", "L-01", LedgerApplication(0), "SAFE", "resource-ownership", "owner check denies a different owner"),
        _sample("app-ledger", "train", "L-02", LedgerApplication(1), "VULNERABLE", "resource-ownership", "owner check is bypassed for a different owner"),
        _sample("app-console", "validation", "C-01", ConsoleApplication(0), "SAFE", "operation-role", "admin operation denies analyst role"),
        _sample("app-console", "validation", "C-02", ConsoleApplication(1), "VULNERABLE", "operation-role", "admin operation accepts analyst role"),
        _sample("app-vault", "held_out", "V-01", TenantVaultApplication(0), "SAFE", "tenant-isolation", "same-tenant membership is authorized by tenant policy"),
        _sample("app-vault", "held_out", "V-02", TenantVaultApplication(1), "VULNERABLE", "tenant-isolation", "cross-tenant identity receives restricted document"),
        _sample("app-projects", "held_out", "P-01", ProjectApplication(0), "SAFE", "nested-membership", "project membership permits member carol and denies bob"),
        _sample("app-projects", "held_out", "P-02", ProjectApplication(1), "VULNERABLE", "nested-membership", "non-member bob reaches nested task"),
        _sample("app-reports", "validation", "R-01", ReportsAvailabilityApplication(), "INCONCLUSIVE", "insufficient-evidence", "service availability prevents an authorization decision"),
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


def _case_record(sample: CrossApplicationSample) -> dict[str, Any]:
    findings, tests, limitations = IDORBOLAEngine(sample.scope).scan(sample.application)
    predicted = _classify(findings, tests, limitations)
    evidence_errors = sorted({error for finding in findings for error in validate_confirmed_evidence(finding)})
    status_only_proof = bool(findings) and not any(
        evidence.kind == "controlled_authorization_comparison"
        and isinstance(evidence.expected, dict)
        and "body" in evidence.expected
        and isinstance(evidence.observed, dict)
        and "body" in evidence.observed
        for finding in findings
        for evidence in finding.evidence
    )
    evidence_quality = "observed_authorization_comparison" if findings and not evidence_errors and not status_only_proof else ("invalid" if findings else "none")
    return {
        "case_id": sample.oracle.case_id,
        "application_id": sample.application_id,
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
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
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


def evaluate_cross_application_corpus(samples: Sequence[CrossApplicationSample] | None = None) -> dict[str, Any]:
    corpus = tuple(samples or build_cross_application_corpus())
    records = [_case_record(sample) for sample in corpus]
    application_ids = sorted({sample.application_id for sample in corpus})
    per_application = {
        application_id: {
            "case_count": sum(1 for item in records if item["application_id"] == application_id),
            "split": next(item["split"] for item in records if item["application_id"] == application_id),
            **_metrics([item for item in records if item["application_id"] == application_id]),
        }
        for application_id in application_ids
    }
    split_metrics = {split: _metrics([item for item in records if item["split"] == split]) for split in ("train", "validation", "held_out")}
    held_out_applications = sorted({item["application_id"] for item in records if item["split"] == "held_out"})
    held_out = split_metrics["held_out"]
    if held_out["true_positives"] and held_out["false_positives"]:
        verdict = "PARTIAL GENERALIZATION"
    elif held_out["true_positives"] and not held_out["false_positives"] and not held_out["false_negatives"]:
        verdict = "GENERALIZATION DEMONSTRATED ON LIMITED LOCAL APPLICATIONS"
    else:
        verdict = "GENERALIZATION NOT DEMONSTRATED"
    return {
        "corpus_version": "phase20-cross-application-v1",
        "application_count": len(application_ids),
        "applications": application_ids,
        "held_out_applications": held_out_applications,
        "ground_truth_accessible_to_engine": False,
        "tuning_used_held_out": False,
        "external_targets_contacted": 0,
        "split_policy": "all cases of an application stay in one split; held-out applications were not used to tune the engine",
        "phase20_baseline": {
            "false_positives": 2,
            "held_out": {"total": 4, "true_positives": 2, "false_positives": 2, "true_negatives": 0, "false_negatives": 0, "precision": 0.5, "recall": 1.0, "f1": 0.6666666666666666},
            "false_positive_cases": ["app-vault same-tenant member", "app-projects valid project member"],
            "source": "measured Phase 20 cross-application run before policy-aware confirmation",
        },
        "phase20_false_positive_resolution": {
            "app-vault same-tenant member": "resolved by explicit tenant-membership policy observation",
            "app-projects valid project member": "resolved by explicit project-membership policy observation",
        },
        "applications_are_independently_implemented": True,
        "application_specific_limitations": [
            "The engine assumes resource.owner_id is the primary baseline authority, so legitimate tenant or membership access can appear as IDOR.",
            "The engine's generic fixture contract does not model arbitrary operation-level policies, session state, or capability graphs.",
            "Response conventions differ across applications; status-only differences remain insufficient evidence.",
            "All applications are synthetic in-memory models and do not represent production frameworks or external targets.",
        ],
        "cases": records,
        "overall": _metrics(records),
        "train": split_metrics["train"],
        "validation": split_metrics["validation"],
        "held_out": held_out,
        "splits": split_metrics,
        "per_application": per_application,
        "verdict": verdict,
    }
