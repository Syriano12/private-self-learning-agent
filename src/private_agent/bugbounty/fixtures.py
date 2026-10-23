"""Deterministic in-memory applications used by the Bug Bounty benchmark.

No socket, HTTP client, subprocess, target code, or external asset is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from private_agent.bugbounty.models import AuthorizationObservation, PolicyDecision, ScopeDecision, ScopeDefinition


@dataclass(frozen=True)
class FixtureIdentity:
    identity_id: str
    role: str
    tenant_id: str


@dataclass(frozen=True)
class FixtureResource:
    resource_id: str
    resource_type: str
    owner_id: str
    tenant_id: str
    parent_id: str = ""
    sensitivity: str = "normal"


@dataclass(frozen=True)
class FixtureResponse:
    status: int
    body: dict[str, Any]
    headers: dict[str, str]


class ScopeViolation(RuntimeError):
    """Raised before a request when the explicit scope does not allow it."""


@dataclass(frozen=True)
class FixtureEndpoint:
    endpoint_id: str
    method: str
    path_template: str
    resource_type: str
    access_level: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "endpoint_id": self.endpoint_id,
            "method": self.method,
            "path_template": self.path_template,
            "resource_type": self.resource_type,
            "access_level": self.access_level,
            "description": self.description,
        }


class DeterministicFixtureApp:
    """Small local application model with explicit secure/vulnerable variants."""

    base_url = "https://fixture.local"

    def __init__(self, variant: str = "secure") -> None:
        variants = {"secure", "idor_horizontal", "idor_vertical", "idor_nested", "false_positive", "inconclusive"}
        if variant not in variants:
            raise ValueError("fixture_variant_unsupported")
        self.variant = variant
        self.identities = {
            "alice": FixtureIdentity("alice", "user", "tenant-a"),
            "bob": FixtureIdentity("bob", "user", "tenant-b"),
            "admin": FixtureIdentity("admin", "admin", "tenant-a"),
        }
        self.resources = {
            "order-alice": FixtureResource("order-alice", "order", "alice", "tenant-a", sensitivity="private"),
            "order-bob": FixtureResource("order-bob", "order", "bob", "tenant-b", sensitivity="private"),
            "comment-bob": FixtureResource("comment-bob", "comment", "bob", "tenant-b", parent_id="order-alice", sensitivity="private"),
            "report-admin": FixtureResource("report-admin", "admin_report", "admin", "tenant-a", sensitivity="restricted"),
        }
        self.endpoints = (
            FixtureEndpoint("order-read", "GET", "/api/v1/orders/{order_id}", "order", "owner", "Read an order owned by the identity."),
            FixtureEndpoint("nested-comment-read", "GET", "/api/v1/orders/{order_id}/comments/{comment_id}", "comment", "owner_and_parent", "Read a nested comment."),
            FixtureEndpoint("admin-report-read", "GET", "/api/v1/admin/reports/{report_id}", "admin_report", "admin", "Read a role-protected administrative report."),
        )

    def inventory(self) -> list[dict[str, str]]:
        return [endpoint.to_dict() for endpoint in self.endpoints]

    def url_for(self, endpoint: FixtureEndpoint, resource_id: str, *, parent_id: str = "") -> str:
        path = endpoint.path_template.replace("{order_id}", parent_id or resource_id).replace("{comment_id}", resource_id).replace("{report_id}", resource_id)
        return self.base_url + path

    def _find_endpoint(self, path: str, method: str) -> FixtureEndpoint | None:
        for endpoint in self.endpoints:
            if endpoint.method != method.upper():
                continue
            template_parts = endpoint.path_template.strip("/").split("/")
            path_parts = path.strip("/").split("/")
            if len(template_parts) != len(path_parts):
                continue
            matched = True
            for expected, actual in zip(template_parts, path_parts):
                if expected.startswith("{") and expected.endswith("}"):
                    continue
                if expected != actual:
                    matched = False
                    break
            if matched:
                return endpoint
        return None

    @staticmethod
    def _ids(endpoint: FixtureEndpoint, path: str) -> tuple[str, str]:
        parts = path.strip("/").split("/")
        if endpoint.endpoint_id == "order-read":
            return (parts[-1], "")
        if endpoint.endpoint_id == "nested-comment-read":
            return (parts[-1], parts[-3])
        return (parts[-1], "")

    def _authorized(self, identity: FixtureIdentity, endpoint: FixtureEndpoint, resource: FixtureResource, parent: FixtureResource | None) -> bool:
        if endpoint.access_level == "admin":
            return identity.role == "admin"
        if endpoint.access_level == "owner":
            return identity.identity_id == resource.owner_id or identity.role == "admin"
        if endpoint.access_level == "owner_and_parent":
            return (
                parent is not None
                and parent.owner_id == identity.identity_id
            ) or identity.role == "admin"
        return False

    def authorization_decision(self, identity_id: str, method: str, url: str, *, resource_id: str) -> AuthorizationObservation:
        """Expose the fixture's local policy adapter, not benchmark ground truth."""
        if self.variant == "inconclusive":
            return AuthorizationObservation(PolicyDecision.UNKNOWN, "fixture_policy", "fixture_unavailable_before_policy_observation")
        identity = self.identities.get(identity_id)
        parsed = urlparse(url)
        endpoint = self._find_endpoint(parsed.path, method)
        resource = self.resources.get(resource_id)
        if identity is None or endpoint is None or resource is None:
            return AuthorizationObservation(PolicyDecision.DENY, "fixture_policy", "credential_or_resource_boundary_denied")
        _, parent_id = self._ids(endpoint, parsed.path)
        parent = self.resources.get(parent_id) if parent_id else None
        allowed = self._authorized(identity, endpoint, resource, parent)
        if self.variant == "false_positive" and endpoint.endpoint_id == "order-read" and identity.identity_id != resource.owner_id and identity.role != "admin":
            allowed = True
        return AuthorizationObservation(
            PolicyDecision.ALLOW if allowed else PolicyDecision.DENY,
            "fixture_policy",
            "explicit_deterministic_fixture_authorization_rule",
        )

    def request(self, identity_id: str, method: str, url: str) -> FixtureResponse:
        identity = self.identities.get(identity_id)
        if identity is None:
            return FixtureResponse(401, {"error": "unknown_identity"}, {"content-type": "application/json"})
        parsed = urlparse(url)
        endpoint = self._find_endpoint(parsed.path, method)
        if endpoint is None:
            return FixtureResponse(404, {"error": "not_found"}, {"content-type": "application/json"})
        resource_id, parent_id = self._ids(endpoint, parsed.path)
        resource = self.resources.get(resource_id)
        parent = self.resources.get(parent_id) if parent_id else None
        if resource is None or resource.resource_type != endpoint.resource_type:
            return FixtureResponse(404, {"error": "resource_not_found"}, {"content-type": "application/json"})
        if self.variant == "inconclusive":
            return FixtureResponse(503, {"error": "fixture_unavailable", "request_id": "dynamic"}, {"retry-after": "1"})
        allowed = self._authorized(identity, endpoint, resource, parent)
        if self.variant == "idor_horizontal" and endpoint.endpoint_id == "order-read" and identity.identity_id == "bob" and resource.owner_id == "alice":
            allowed = True
        if self.variant == "idor_vertical" and endpoint.endpoint_id == "admin-report-read" and identity.role != "admin":
            allowed = True
        if self.variant == "idor_nested" and endpoint.endpoint_id == "nested-comment-read" and identity.identity_id == "bob" and resource.owner_id == "bob" and parent is not None and parent.owner_id == "alice":
            allowed = True
        if self.variant == "false_positive" and endpoint.endpoint_id == "order-read" and identity.identity_id != resource.owner_id and identity.role != "admin":
            allowed = True
        if not allowed:
            return FixtureResponse(403, {"error": "forbidden", "request_id": "dynamic"}, {"content-type": "application/json"})
        body = {
            "resource_id": resource.resource_id,
            "resource_type": resource.resource_type,
            "owner_id": resource.owner_id,
            "tenant_id": resource.tenant_id,
            "sensitivity": resource.sensitivity,
            "value": f"controlled-{resource.resource_id}",
            "timestamp": "dynamic",
            "request_id": "dynamic",
        }
        if self.variant == "false_positive" and identity.identity_id != resource.owner_id and identity.role != "admin":
            body = {"resource_id": resource.resource_id, "value": "public-summary", "timestamp": "different", "request_id": "another-dynamic-value"}
        return FixtureResponse(200, body, {"content-type": "application/json", "x-fixture": "deterministic"})


class ScopeEnforcedFixtureClient:
    """Actual active-request boundary for the local fixture.

    The fixture itself is inert data. This client is the only object used by
    the evaluator for requests and checks scope immediately before every call.
    """

    def __init__(self, fixture: DeterministicFixtureApp, scope: ScopeDefinition) -> None:
        self.fixture = fixture
        self.scope = scope
        self.calls = 0

    def request(self, identity_id: str, method: str, url: str, *, resource_id: str) -> FixtureResponse:
        parsed = urlparse(url)
        path_segments = {segment for segment in parsed.path.split("/") if segment}
        if not resource_id or resource_id not in path_segments:
            raise ScopeViolation("resource_url_mismatch")
        decision, reason = self.scope.authorize(
            url,
            method=method,
            mode="active",
            identity=identity_id,
            resource_id=resource_id,
        )
        if decision is not ScopeDecision.ALLOWED:
            raise ScopeViolation(reason)
        self.calls += 1
        return self.fixture.request(identity_id, method, url)


def benchmark_fixture_variants() -> dict[str, DeterministicFixtureApp]:
    return {name: DeterministicFixtureApp(name) for name in ("secure", "idor_horizontal", "idor_vertical", "idor_nested", "false_positive", "inconclusive")}
