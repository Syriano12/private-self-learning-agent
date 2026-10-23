"""Typed, bounded Bug Bounty research records.

These records are advisory evidence only. They never grant permissions, select a
runtime, or authorize an active request.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse


MAX_TEXT = 4096
MAX_EXCERPT = 1800
MAX_JSON_BYTES = 32_000
_SECRET_KEYS = {
    "authorization", "cookie", "set-cookie", "token", "access_token", "refresh_token",
    "password", "secret", "api_key", "apikey", "credential", "credentials",
}
_SECRET_TEXT_PATTERNS = (
    (re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(secret|password|token)\s*[:=]\s*['\"]?[^\s,'\"]+"), r"\1=[REDACTED]"),
    (re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----.*?-----END [A-Z ]+ PRIVATE KEY-----", re.S), "[REDACTED_PRIVATE_KEY]"),
)
_DYNAMIC_KEYS = {"timestamp", "created_at", "updated_at", "request_id", "nonce", "csrf", "csrf_token"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded_text(value: Any, *, limit: int = MAX_TEXT, field_name: str = "text") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name}_must_be_text")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name}_required")
    if len(value) > limit:
        raise ValueError(f"{field_name}_too_large")
    return value


def _strict_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name}_must_be_boolean")
    return value


def _bounded_string_list(value: Any, *, field_name: str, limit: int = 100) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name}_must_be_list")
    if len(value) > limit:
        raise ValueError(f"{field_name}_too_many")
    return tuple(_bounded_text(item, limit=256, field_name=field_name) for item in value)


def _safe_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if depth > 6:
        return "[TRUNCATED]"
    if key.lower().replace("-", "_") in _SECRET_KEYS:
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k)[:128]: _safe_value(v, key=str(k), depth=depth + 1) for k, v in list(value.items())[:100]}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, key=key, depth=depth + 1) for item in list(value)[:100]]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("evidence_non_finite_number")
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            if len(value) > MAX_EXCERPT:
                value = value[:MAX_EXCERPT] + "...[TRUNCATED]"
            for pattern, replacement in _SECRET_TEXT_PATTERNS:
                value = pattern.sub(replacement, value)
        return value
    raise ValueError("evidence_value_not_json_primitive")


def safe_json(value: Any) -> Any:
    result = _safe_value(value)
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("evidence_json_too_large")
    return result


def stable_digest(value: Any) -> str:
    payload = json.dumps(safe_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def public_reference(value: Any) -> str:
    """Expose only a stable opaque digest for authorization references."""
    text = _bounded_text(value, limit=MAX_TEXT, field_name="reference")
    if text.startswith("sha256:") and len(text) == len("sha256:") + 16:
        return text
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _publicize_references(value: Any, *, key: str = "") -> Any:
    normalized = key.lower().replace("-", "_")
    if normalized in {"authorization_reference", "scope_reference"} and isinstance(value, str):
        return public_reference(value)
    if isinstance(value, Mapping):
        return {str(name): _publicize_references(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_publicize_references(item, key=key) for item in value]
    return value


def semantic_normalize(value: Any, *, key: str = "") -> Any:
    """Normalize responses while ignoring documented dynamic fields only."""
    if isinstance(value, Mapping):
        return {
            str(name): semantic_normalize(item, key=str(name))
            for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(name).lower() not in _DYNAMIC_KEYS and str(name).lower().replace("-", "_") not in _SECRET_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [semantic_normalize(item, key=key) for item in value]
    return value


class ScopeDecision(str, Enum):
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    AMBIGUOUS = "AMBIGUOUS"
    EXPIRED = "EXPIRED"


class PolicyDecision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class AuthorizationObservation:
    """A local, explicitly authorized observation of an application's policy."""

    decision: PolicyDecision
    model: str
    basis: str

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PolicyDecision):
            raise ValueError("policy_decision_invalid")
        object.__setattr__(self, "model", _bounded_text(self.model, limit=256, field_name="policy_model"))
        object.__setattr__(self, "basis", _bounded_text(self.basis, limit=512, field_name="policy_basis"))


@dataclass(frozen=True)
class ScopeAsset:
    asset_id: str
    host: str
    path_prefix: str = "/"
    methods: tuple[str, ...] = ("GET",)
    active_allowed: bool = False
    ports: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_id", _bounded_text(self.asset_id, field_name="asset_id"))
        host = _bounded_text(self.host.lower(), field_name="host")
        if "://" in host or "/" in host or " " in host:
            raise ValueError("host_must_be_hostname")
        object.__setattr__(self, "host", host)
        path = self.path_prefix or "/"
        if not path.startswith("/") or len(path) > 512:
            raise ValueError("path_prefix_invalid")
        decoded_path = unquote(path)
        if "\\" in decoded_path or any(segment in {".", ".."} for segment in decoded_path.split("/")):
            raise ValueError("path_prefix_traversal_invalid")
        object.__setattr__(self, "path_prefix", path.rstrip("/") or "/")
        normalized = tuple(sorted({str(item).upper() for item in self.methods}))
        if not normalized or any(not re.fullmatch(r"[A-Z]{3,8}", item) for item in normalized):
            raise ValueError("methods_invalid")
        object.__setattr__(self, "methods", normalized)
        if any(type(port) is not int or port < 1 or port > 65535 for port in self.ports):
            raise ValueError("ports_invalid")
        object.__setattr__(self, "ports", tuple(sorted(set(self.ports))))


@dataclass(frozen=True)
class ScopeDefinition:
    program_id: str
    authorization_reference: str
    issued_at: str
    expires_at: str
    assets: tuple[ScopeAsset, ...]
    excluded_hosts: tuple[str, ...] = ()
    prohibited_methods: tuple[str, ...] = ("DELETE", "PATCH", "PUT", "POST")
    permitted_methods: tuple[str, ...] = ("GET", "HEAD", "OPTIONS")
    permitted_identities: tuple[str, ...] = ()
    permitted_resources: tuple[str, ...] = ()
    active_testing: bool = False
    ambiguous: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, now: datetime | None = None) -> "ScopeDefinition":
        if not isinstance(payload, Mapping):
            raise ValueError("scope_object_required")
        assets_payload = payload.get("assets")
        if not isinstance(assets_payload, list) or not assets_payload:
            raise ValueError("scope_assets_required")
        assets = tuple(
            ScopeAsset(
                asset_id=str(item.get("asset_id", "")),
                host=str(item.get("host", "")),
                path_prefix=str(item.get("path_prefix", "/")),
                methods=_bounded_string_list(item.get("methods", ["GET"]), field_name="methods", limit=16),
                active_allowed=_strict_bool(item.get("active_allowed", False), field_name="active_allowed"),
                ports=tuple(item.get("ports", [])) if isinstance(item.get("ports", []), list) else (_ for _ in ()).throw(ValueError("ports_invalid")),
            )
            for item in assets_payload
            if isinstance(item, Mapping)
        )
        if len(assets) != len(assets_payload):
            raise ValueError("scope_asset_invalid")
        scope = cls(
            program_id=_bounded_text(payload.get("program_id", ""), field_name="program_id"),
            authorization_reference=_bounded_text(payload.get("authorization_reference", ""), field_name="authorization_reference"),
            issued_at=_bounded_text(payload.get("issued_at", ""), field_name="issued_at"),
            expires_at=_bounded_text(payload.get("expires_at", ""), field_name="expires_at"),
            assets=assets,
            excluded_hosts=tuple(item.lower() for item in _bounded_string_list(payload.get("excluded_hosts", []), field_name="excluded_hosts")),
            prohibited_methods=tuple(item.upper() for item in _bounded_string_list(payload.get("prohibited_methods", ["DELETE", "PATCH", "PUT", "POST"]), field_name="prohibited_methods", limit=16)),
            permitted_methods=tuple(item.upper() for item in _bounded_string_list(payload.get("permitted_methods", ["GET", "HEAD", "OPTIONS"]), field_name="permitted_methods", limit=16)),
            permitted_identities=_bounded_string_list(payload.get("permitted_identities", []), field_name="permitted_identities"),
            permitted_resources=_bounded_string_list(payload.get("permitted_resources", []), field_name="permitted_resources"),
            active_testing=_strict_bool(payload.get("active_testing", False), field_name="active_testing"),
            ambiguous=_strict_bool(payload.get("ambiguous", False), field_name="ambiguous"),
        )
        scope.validate(now=now)
        return scope

    def validate(self, *, now: datetime | None = None) -> None:
        if self.ambiguous:
            return
        try:
            issued = datetime.fromisoformat(self.issued_at)
            expires = datetime.fromisoformat(self.expires_at)
        except ValueError as exc:
            raise ValueError("scope_timestamps_invalid") from exc
        now_value = now or datetime.now(timezone.utc)
        if issued.tzinfo is None or expires.tzinfo is None:
            raise ValueError("scope_timestamps_must_be_timezone_aware")
        if expires <= issued:
            raise ValueError("scope_expiry_invalid")
        if expires <= now_value:
            raise ValueError("scope_expired")
        if not self.assets:
            raise ValueError("scope_assets_required")
        if any(method in self.prohibited_methods for method in self.permitted_methods):
            raise ValueError("scope_method_conflict")
        if not set(self.permitted_identities):
            raise ValueError("scope_identities_required")
        if not set(self.permitted_resources):
            raise ValueError("scope_resources_required")

    def authorize(
        self,
        url: str,
        *,
        method: str = "GET",
        mode: str = "passive",
        identity: str = "",
        resource_id: str = "",
        now: datetime | None = None,
    ) -> tuple[ScopeDecision, str]:
        if self.ambiguous:
            return ScopeDecision.AMBIGUOUS, "scope_is_ambiguous"
        try:
            self.validate(now=now)
        except ValueError as exc:
            if str(exc) == "scope_expired":
                return ScopeDecision.EXPIRED, str(exc)
            return ScopeDecision.AMBIGUOUS, str(exc)
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"}:
            return ScopeDecision.DENIED, "scheme_not_permitted"
        if parsed.username is not None or parsed.password is not None:
            return ScopeDecision.DENIED, "userinfo_not_permitted"
        host = (parsed.hostname or "").lower()
        raw_path = parsed.path or "/"
        path = unquote(raw_path)
        twice_decoded_path = unquote(path)
        if "\\" in path or any(segment in {".", ".."} for segment in path.split("/")) or "\\" in twice_decoded_path or any(segment in {".", ".."} for segment in twice_decoded_path.split("/")):
            return ScopeDecision.DENIED, "path_traversal_not_permitted"
        if host in set(self.excluded_hosts):
            return ScopeDecision.DENIED, "host_excluded"
        method = method.upper()
        if method in self.prohibited_methods or method not in self.permitted_methods:
            return ScopeDecision.DENIED, "method_not_permitted"
        try:
            effective_port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        except ValueError:
            return ScopeDecision.DENIED, "port_invalid"
        if effective_port not in {80, 443} and not any(effective_port in asset.ports for asset in self.assets):
            return ScopeDecision.DENIED, "port_not_authorized"
        if mode == "active" and (not self.active_testing or method not in {"GET", "HEAD", "OPTIONS"}):
            return ScopeDecision.DENIED, "active_testing_not_authorized"
        if mode == "active" and not identity:
            return ScopeDecision.DENIED, "identity_required_for_active_testing"
        if mode == "active" and not resource_id:
            return ScopeDecision.DENIED, "resource_required_for_active_testing"
        if identity and identity not in self.permitted_identities:
            return ScopeDecision.DENIED, "identity_not_authorized"
        if resource_id and resource_id not in self.permitted_resources:
            return ScopeDecision.DENIED, "resource_not_authorized"
        for asset in self.assets:
            prefix_matches = asset.path_prefix == "/" or path == asset.path_prefix or path.startswith(asset.path_prefix + "/")
            if asset.host == host and prefix_matches and method in asset.methods:
                if asset.ports and effective_port not in asset.ports:
                    continue
                if not asset.ports and effective_port not in {80, 443}:
                    continue
                if mode == "active" and not asset.active_allowed:
                    return ScopeDecision.DENIED, "asset_active_testing_not_authorized"
                return ScopeDecision.ALLOWED, "explicit_scope_match"
        return ScopeDecision.DENIED, "asset_out_of_scope"

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "authorization_reference": public_reference(self.authorization_reference) if public else self.authorization_reference,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "assets": [{**asset.__dict__, "ports": list(asset.ports)} for asset in self.assets],
            "excluded_hosts": list(self.excluded_hosts),
            "prohibited_methods": list(self.prohibited_methods),
            "permitted_methods": list(self.permitted_methods),
            "permitted_identities": list(self.permitted_identities),
            "permitted_resources": list(self.permitted_resources),
            "active_testing": self.active_testing,
            "ambiguous": self.ambiguous,
        }


@dataclass(frozen=True)
class Evidence:
    kind: str
    source: str
    expected: Any
    observed: Any
    provenance: dict[str, Any]
    request: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _bounded_text(self.kind, field_name="evidence_kind"))
        object.__setattr__(self, "source", _bounded_text(self.source, field_name="evidence_source"))
        object.__setattr__(self, "expected", safe_json(self.expected))
        object.__setattr__(self, "observed", safe_json(self.observed))
        object.__setattr__(self, "provenance", safe_json(self.provenance))
        object.__setattr__(self, "request", safe_json(self.request))
        object.__setattr__(self, "response", safe_json(self.response))

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "expected": self.expected,
            "observed": self.observed,
            "provenance": _publicize_references(self.provenance) if public else self.provenance,
            "request": self.request,
            "response": self.response,
            "evidence_sha256": stable_digest({"kind": self.kind, "source": self.source, "expected": self.expected, "observed": self.observed, "provenance": self.provenance}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Evidence":
        if not isinstance(payload, Mapping):
            raise ValueError("evidence_record_invalid")
        return cls(
            kind=payload.get("kind", ""),
            source=payload.get("source", ""),
            expected=payload.get("expected", {}),
            observed=payload.get("observed", {}),
            provenance=payload.get("provenance", {}),
            request=payload.get("request", {}),
            response=payload.get("response", {}),
        )


class FindingState(str, Enum):
    OBSERVED = "OBSERVED"
    HYPOTHESIS = "HYPOTHESIS"
    VALIDATION_PENDING = "VALIDATION_PENDING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    REPORT_READY = "REPORT_READY"
    HUMAN_REVIEWED = "HUMAN_REVIEWED"


class Severity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


_ALLOWED_TRANSITIONS: dict[FindingState, set[FindingState]] = {
    FindingState.OBSERVED: {FindingState.HYPOTHESIS, FindingState.REJECTED},
    FindingState.HYPOTHESIS: {FindingState.VALIDATION_PENDING, FindingState.REJECTED},
    FindingState.VALIDATION_PENDING: {FindingState.CONFIRMED, FindingState.REJECTED},
    FindingState.CONFIRMED: {FindingState.REPORT_READY, FindingState.REJECTED},
    FindingState.REPORT_READY: {FindingState.HUMAN_REVIEWED},
    FindingState.HUMAN_REVIEWED: set(),
    FindingState.REJECTED: set(),
}


@dataclass
class Finding:
    finding_id: str
    vulnerability_class: str
    title: str
    asset: str
    endpoint: str
    state: FindingState
    severity: Severity
    confidence: float
    scope_reference: str
    preconditions: list[str] = field(default_factory=list)
    identities: list[str] = field(default_factory=list)
    reproduction_steps: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    expected_behavior: str = ""
    observed_behavior: str = ""
    root_cause_hypothesis: str = ""
    verified_root_cause: str = ""
    impact: str = ""
    remediation: str = ""
    regression: str = ""
    limitations: list[str] = field(default_factory=list)
    duplicate_of: str = ""
    disclosure_status: str = "private"
    validation_status: str = "NOT_VALIDATED"
    validation_errors: list[str] = field(default_factory=list)
    validated_evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if self.state in {FindingState.CONFIRMED, FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED}:
            raise ValueError("finding_confirmation_requires_transition")

    def transition(self, state: FindingState) -> None:
        if state == self.state:
            return
        if state not in _ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(f"invalid_finding_transition:{self.state.value}:{state.value}")
        if state in {FindingState.CONFIRMED, FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED}:
            self._require_confirmable(target_state=state)
        self.state = state

    def _require_confirmable(self, *, target_state: FindingState) -> None:
        if not self.scope_reference:
            raise ValueError("finding_scope_reference_required")
        if not self.evidence:
            raise ValueError("finding_evidence_required")
        if not self.reproduction_steps:
            raise ValueError("finding_reproduction_required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("finding_confidence_invalid")
        if not self.expected_behavior or not self.observed_behavior:
            raise ValueError("finding_behavior_comparison_required")
        errors = validate_confirmed_evidence(self, target_state=target_state)
        if errors:
            self.validation_status = "REJECTED"
            self.validation_errors = errors
            self.validated_evidence_sha256 = ""
            raise ValueError("finding_confirmation_evidence_invalid:" + ",".join(errors))
        self.validation_status = "VALIDATED"
        self.validation_errors = []
        self.validated_evidence_sha256 = self._evidence_digest()

    def _evidence_digest(self) -> str:
        canonical = []
        for item in self.evidence:
            rendered = item.to_dict(public=True)
            rendered.pop("evidence_sha256", None)
            canonical.append(rendered)
        return stable_digest(canonical)

    def _invalidate_confirmation(self, errors: list[str]) -> None:
        if self.state in {FindingState.CONFIRMED, FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED}:
            self.state = FindingState.HYPOTHESIS
        self.validation_status = "REJECTED"
        self.validation_errors = sorted(set(errors))
        self.validated_evidence_sha256 = ""

    def _assert_validated_for_reporting(self) -> None:
        if self.state not in {FindingState.CONFIRMED, FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED}:
            return
        errors = validate_confirmed_evidence(self)
        if errors:
            self._invalidate_confirmation(errors)
            raise ValueError("finding_confirmation_evidence_invalid:" + ",".join(errors))
        if self.validation_status != "VALIDATED" or self.validation_errors:
            self._invalidate_confirmation(["finding_validation_metadata_invalid"])
            raise ValueError("finding_validation_metadata_invalid")
        if self.validated_evidence_sha256 != self._evidence_digest():
            self._invalidate_confirmation(["finding_evidence_mutated_after_validation"])
            raise ValueError("finding_evidence_mutated_after_validation")

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        self._assert_validated_for_reporting()
        return {
            "finding_id": self.finding_id,
            "vulnerability_class": self.vulnerability_class,
            "title": self.title,
            "asset": self.asset,
            "endpoint": self.endpoint,
            "state": self.state.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "scope_reference": public_reference(self.scope_reference) if public else self.scope_reference,
            "preconditions": self.preconditions[:20],
            "identities": self.identities[:20],
            "reproduction_steps": self.reproduction_steps[:20],
            "evidence": [item.to_dict(public=public) for item in self.evidence[:20]],
            "expected_behavior": self.expected_behavior,
            "observed_behavior": self.observed_behavior,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "verified_root_cause": self.verified_root_cause,
            "impact": self.impact,
            "remediation": self.remediation,
            "regression": self.regression,
            "limitations": self.limitations[:20],
            "duplicate_of": self.duplicate_of,
            "disclosure_status": self.disclosure_status,
            "validation": {
                "status": self.validation_status,
                "errors": self.validation_errors[:20],
                "evidence_sha256": self.validated_evidence_sha256,
                "validator": "idor_bola_semantic_v1" if self.validation_status == "VALIDATED" else "not_validated",
            },
        }

    @classmethod
    def deterministic_id(cls, vulnerability_class: str, asset: str, endpoint: str, title: str) -> str:
        return "F-" + stable_digest([vulnerability_class, asset, endpoint, title])[:12].upper()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Finding":
        if not isinstance(payload, Mapping):
            raise ValueError("finding_record_invalid")
        original_state = FindingState(str(payload.get("state", FindingState.OBSERVED.value)))
        evidence = [Evidence.from_dict(item) for item in payload.get("evidence", [])]
        finding = cls(
            finding_id=str(payload.get("finding_id", "")),
            vulnerability_class=str(payload.get("vulnerability_class", "")),
            title=str(payload.get("title", "")),
            asset=str(payload.get("asset", "")),
            endpoint=str(payload.get("endpoint", "")),
            state=FindingState.OBSERVED,
            severity=Severity(str(payload.get("severity", Severity.INFO.value))),
            confidence=float(payload.get("confidence", 0.0)),
            scope_reference=str(payload.get("scope_reference", "")),
            preconditions=list(payload.get("preconditions", [])),
            identities=list(payload.get("identities", [])),
            reproduction_steps=list(payload.get("reproduction_steps", [])),
            evidence=evidence,
            expected_behavior=str(payload.get("expected_behavior", "")),
            observed_behavior=str(payload.get("observed_behavior", "")),
            root_cause_hypothesis=str(payload.get("root_cause_hypothesis", "")),
            verified_root_cause=str(payload.get("verified_root_cause", "")),
            impact=str(payload.get("impact", "")),
            remediation=str(payload.get("remediation", "")),
            regression=str(payload.get("regression", "")),
            limitations=list(payload.get("limitations", [])),
            duplicate_of=str(payload.get("duplicate_of", "")),
            disclosure_status=str(payload.get("disclosure_status", "private")),
        )
        validation = payload.get("validation") if isinstance(payload.get("validation"), Mapping) else {}
        finding.validation_status = str(validation.get("status", "NOT_VALIDATED"))
        finding.validation_errors = [str(item) for item in validation.get("errors", [])] if isinstance(validation.get("errors", []), list) else ["validation_errors_malformed"]
        finding.validated_evidence_sha256 = str(validation.get("evidence_sha256", ""))
        if original_state in {FindingState.CONFIRMED, FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED}:
            errors = validate_confirmed_evidence(finding, target_state=original_state)
            if not validation or finding.validation_status != "VALIDATED":
                errors.append("persisted_validation_metadata_missing")
            if finding.validation_errors:
                errors.append("persisted_validation_errors_present")
            if not errors and finding.validated_evidence_sha256 != finding._evidence_digest():
                errors.append("persisted_evidence_digest_mismatch")
            if errors:
                finding.state = original_state
                finding._invalidate_confirmation(errors)
            else:
                finding.state = original_state
        else:
            finding.state = original_state
        return finding


def validate_confirmed_evidence(
    finding: Finding,
    *,
    target_state: FindingState | None = None,
) -> list[str]:
    """Return semantic errors for a finding that claims confirmed IDOR/BOLA evidence."""
    errors: list[str] = []
    effective_state = target_state or finding.state
    if effective_state not in {FindingState.CONFIRMED, FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED}:
        errors.append("finding_not_confirmed")
    if finding.vulnerability_class != "IDOR/BOLA":
        errors.append("unsupported_confirmation_category")
    if not finding.scope_reference:
        errors.append("finding_scope_reference_required")
    if not finding.reproduction_steps:
        errors.append("reproduction_steps_missing")
    if not finding.evidence:
        errors.append("evidence_missing")
    for index, evidence in enumerate(finding.evidence):
        prefix = f"evidence_{index}"
        if evidence.kind != "controlled_authorization_comparison":
            errors.append(f"{prefix}_kind_not_authorization_comparison")
        provenance = evidence.provenance
        for key in ("owner_identity", "test_identity", "resource_id", "endpoint_id", "scope_reference"):
            if not isinstance(provenance.get(key), str) or not provenance[key]:
                errors.append(f"{prefix}_{key}_missing")
        if provenance.get("scope_reference") != finding.scope_reference:
            errors.append(f"{prefix}_scope_reference_binding_mismatch")
        if provenance.get("owner_identity") == provenance.get("test_identity"):
            errors.append(f"{prefix}_identity_is_not_cross_boundary")
        decision = provenance.get("authorization_decision")
        if decision is None:
            errors.append(f"{prefix}_authorization_decision_missing")
        elif decision != PolicyDecision.DENY.value:
            errors.append(f"{prefix}_authorization_decision_not_deny")
        request_identity = evidence.request.get("identity")
        if request_identity != provenance.get("test_identity"):
            errors.append(f"{prefix}_identity_binding_mismatch")
        request_url = evidence.request.get("url")
        resource_id = provenance.get("resource_id")
        parsed_url = urlparse(request_url) if isinstance(request_url, str) else None
        if not isinstance(request_url, str) or not request_url or parsed_url is None or not parsed_url.hostname or not isinstance(resource_id, str) or resource_id not in [unquote(part) for part in parsed_url.path.strip("/").split("/")]:
            errors.append(f"{prefix}_resource_url_binding_missing")
        if parsed_url is not None and parsed_url.hostname and isinstance(finding.asset, str):
            asset_value = finding.asset.rstrip("/")
            asset_url = urlparse(asset_value if "://" in asset_value else "//" + asset_value)
            if not asset_url.hostname:
                errors.append(f"{prefix}_asset_binding_missing")
            elif asset_url.hostname.lower() != parsed_url.hostname.lower():
                errors.append(f"{prefix}_asset_binding_mismatch")
        endpoint_template = finding.endpoint
        if not isinstance(endpoint_template, str) or not endpoint_template:
            errors.append(f"{prefix}_endpoint_binding_missing")
        elif parsed_url is not None and parsed_url.path and endpoint_template.startswith("/"):
            template_parts = [unquote(part) for part in endpoint_template.strip("/").split("/")]
            actual_parts = [unquote(part) for part in parsed_url.path.strip("/").split("/")]
            if len(template_parts) != len(actual_parts) or any(not (part.startswith("{") and part.endswith("}")) and part != actual for part, actual in zip(template_parts, actual_parts)):
                errors.append(f"{prefix}_endpoint_binding_mismatch")
        expected = evidence.expected if isinstance(evidence.expected, dict) else {}
        observed = evidence.observed if isinstance(evidence.observed, dict) else {}
        if expected.get("policy_decision") != PolicyDecision.DENY.value:
            errors.append(f"{prefix}_expected_policy_decision_not_deny")
        if "status" not in expected or "body" not in expected or "status" not in observed or "body" not in observed:
            errors.append(f"{prefix}_comparison_incomplete")
            continue
        if not isinstance(observed.get("status"), int) or observed.get("status", 500) >= 300:
            errors.append(f"{prefix}_observed_result_was_not_allowed")
        expected_body = expected.get("body")
        observed_body = observed.get("body")
        if not isinstance(expected_body, dict) or not isinstance(observed_body, dict):
            errors.append(f"{prefix}_comparison_body_invalid")
            continue
        protected_keys = {"owner_id", "tenant_id", "sensitivity", "classification", "amount", "details", "value"}
        if not protected_keys.intersection(expected_body):
            errors.append(f"{prefix}_baseline_protected_content_missing")
        if not protected_keys.intersection(observed_body):
            errors.append(f"{prefix}_protected_content_not_observed")
        response_status = evidence.response.get("status") if isinstance(evidence.response, dict) else None
        if response_status is not None and response_status != observed.get("status"):
            errors.append(f"{prefix}_response_status_mismatch")
        if not isinstance(evidence.request.get("method"), str) or not evidence.request.get("method"):
            errors.append(f"{prefix}_request_method_missing")
        if not isinstance(provenance.get("policy_model"), str) or not provenance.get("policy_model"):
            errors.append(f"{prefix}_policy_model_missing")
        if not isinstance(provenance.get("policy_basis"), str) or not provenance.get("policy_basis"):
            errors.append(f"{prefix}_policy_basis_missing")
    return sorted(set(errors))


@dataclass(frozen=True)
class TestCaseResult:
    test_id: str
    endpoint: str
    identity: str
    resource_id: str
    status: str
    expected: str
    observed: str
    reason: str
    evidence: Evidence | None = None

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "endpoint": self.endpoint,
            "identity": self.identity,
            "resource_id": self.resource_id,
            "status": self.status,
            "expected": self.expected,
            "observed": self.observed,
            "reason": self.reason,
            "evidence": self.evidence.to_dict(public=public) if self.evidence else None,
        }


@dataclass
class BugBountyRun:
    run_id: str
    scope: ScopeDefinition
    status: str
    mode: str
    inventory: dict[str, Any]
    plan: list[dict[str, Any]]
    findings: list[Finding]
    tests: list[TestCaseResult]
    limitations: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def metrics(self) -> dict[str, Any]:
        confirmed = [item for item in self.findings if item.state in {FindingState.CONFIRMED, FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED}]
        expected_positive = sum(1 for item in self.tests if item.status == "VULNERABLE")
        true_positive = len(confirmed)
        false_positive = sum(1 for item in self.tests if item.status == "FALSE_POSITIVE")
        inconclusive = sum(1 for item in self.tests if item.status == "INCONCLUSIVE")
        suspected = sum(1 for item in self.tests if item.status == "SUSPECTED")
        errors = sum(1 for item in self.tests if item.status == "ERROR")
        return {
            "tests": len(self.tests),
            "confirmed_findings": len(confirmed),
            "true_positives": true_positive,
            "false_positives": false_positive,
            "secure_cases": sum(1 for item in self.tests if item.status == "SECURE"),
            "inconclusive": inconclusive,
            "suspected": suspected,
            "errors": errors,
            "precision": (true_positive / (true_positive + false_positive)) if (true_positive + false_positive) else 1.0,
            "recall": None,
            "recall_note": "Recall is not measurable without an explicit ground-truth corpus for the supplied target.",
        }

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scope": self.scope.to_dict(public=public),
            "status": self.status,
            "mode": self.mode,
            "inventory": safe_json(self.inventory),
            "plan": safe_json(self.plan),
            "findings": [item.to_dict(public=public) for item in self.findings],
            "tests": [item.to_dict(public=public) for item in self.tests],
            "metrics": self.metrics(),
            "limitations": self.limitations[:50],
            "created_at": self.created_at,
        }


def reconcile_bug_bounty_metrics(
    existing: Mapping[str, Any] | None,
    findings: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute claim-sensitive aggregates from the sanitized public records."""
    metrics = dict(existing or {})
    confirmed = sum(
        1
        for item in findings
        if item.get("state") in {FindingState.CONFIRMED.value, FindingState.REPORT_READY.value, FindingState.HUMAN_REVIEWED.value}
        and isinstance(item.get("validation"), Mapping)
        and item["validation"].get("status") == "VALIDATED"
    )
    status_counts = {status: sum(1 for item in tests if item.get("status") == status) for status in ("FALSE_POSITIVE", "SECURE", "INCONCLUSIVE", "SUSPECTED", "ERROR")}
    false_positives = status_counts["FALSE_POSITIVE"]
    metrics.update({
        "tests": len(tests),
        "confirmed_findings": confirmed,
        "true_positives": confirmed,
        "false_positives": false_positives,
        "secure_cases": status_counts["SECURE"],
        "inconclusive": status_counts["INCONCLUSIVE"],
        "suspected": status_counts["SUSPECTED"],
        "errors": status_counts["ERROR"],
        "precision": (confirmed / (confirmed + false_positives)) if confirmed + false_positives else 1.0,
    })
    return metrics
