from __future__ import annotations

import json
import threading
import uuid
import weakref
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Callable

from private_agent.core.experience import sanitize_value
from private_agent.core.recovery import input_fingerprint


_WRITE_ONCE_VALUES: weakref.WeakKeyDictionary[Any, weakref.WeakKeyDictionary[Any, Any]] = weakref.WeakKeyDictionary()
_WRITE_ONCE_UPDATE_TOKEN = object()


def _reference_values(descriptor: Any) -> weakref.WeakKeyDictionary[Any, Any]:
    values = _WRITE_ONCE_VALUES.get(descriptor)
    if values is None:
        values = weakref.WeakKeyDictionary()
        _WRITE_ONCE_VALUES[descriptor] = values
    return values


class _WriteOnceReference:
    """Descriptor-backed reference that cannot be replaced after construction."""

    __slots__ = ("_error", "_name", "__weakref__")

    def __init__(self, error: str = "security_state_immutable") -> None:
        self._error = error
        self._name = "reference"

    def __set_name__(self, owner: type[Any], name: str) -> None:
        self._name = name

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> Any:
        if instance is None:
            # Do not expose a class-level object with a replacement method.
            return None
        try:
            return _reference_values(self)[instance]
        except KeyError as exc:
            raise AttributeError(self._name) from exc

    def __set__(self, instance: Any, value: Any) -> None:
        values = _reference_values(self)
        if instance in values:
            raise AttributeError(self._error)
        values[instance] = value


def _replace_write_once_reference(
    descriptor: _WriteOnceReference,
    instance: Any,
    value: Any,
    *,
    token: object,
) -> None:
    """Internal policy-update hook; the descriptor exposes no replacement method."""
    if token is not _WRITE_ONCE_UPDATE_TOKEN:
        raise AttributeError(descriptor._error)
    values = _reference_values(descriptor)
    if instance not in values:
        raise AttributeError(descriptor._name)
    values[instance] = value


class _MonotonicSealFlag:
    """Allow only the construction transition False -> True."""

    __slots__ = ("_values", "__weakref__")

    def __init__(self) -> None:
        self._values: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> bool:
        if instance is None:
            return None  # type: ignore[return-value]
        try:
            return self._values[instance]
        except KeyError as exc:
            raise AttributeError("_security_state_sealed") from exc

    def __set__(self, instance: Any, value: Any) -> None:
        if instance not in self._values:
            if value is not False:
                raise AttributeError("security_state_immutable")
            self._values[instance] = False
            return
        if self._values[instance] is False and value is True:
            self._values[instance] = True
            return
        raise AttributeError("security_state_immutable")


CAPABILITIES = {
    "READ_WORKSPACE",
    "WRITE_WORKSPACE",
    "NETWORK_ACCESS",
    "EXTERNAL_API",
    "DATABASE_READ",
    "DATABASE_WRITE",
    "SENSITIVE_DATA_ACCESS",
    "SYSTEM_COMMAND",
    "LONG_RUNNING_TASK",
    "SKILL_ACTIVATION",
    "SKILL_UPDATE",
}
RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
POLICY_DECISIONS = {"ALLOW", "DENY", "REQUIRE_APPROVAL", "QUARANTINE"}
APPROVAL_STATUSES = {"PENDING", "APPROVED", "DENIED", "EXPIRED", "CANCELLED"}
AUDIT_EVENTS = {
    "POLICY_EVALUATED",
    "PERMISSION_GRANTED",
    "PERMISSION_DENIED",
    "APPROVAL_REQUESTED",
    "APPROVAL_GRANTED",
    "APPROVAL_DENIED",
    "APPROVAL_CANCELLED",
    "APPROVAL_EXPIRED",
    "APPROVAL_CANCELLED",
    "APPROVAL_INVALIDATED",
    "ACTION_BLOCKED",
    "ACTION_EXECUTED",
    "ACTION_FAILED",
    "SKILL_ACTIVATED",
    "SKILL_QUARANTINED",
    "SKILL_ROLLED_BACK",
    "POLICY_CHANGED",
    "TASK_CREATED",
    "CHECKPOINT_SAVED",
    "TASK_RESUMED",
    "RESUME_BLOCKED",
    "EXECUTION_UNKNOWN",
    "RECOVERY_REQUIRED",
    "APPROVAL_REVALIDATED",
    "STATE_MIGRATION",
    "TASK_COMPLETED",
    "TASK_FAILED",
}

# These events assert an authorization/execution decision.  They must never be
# writable through the generic AuditTrail API because their provenance is
# security-sensitive.
TRUSTED_SECURITY_AUDIT_EVENTS = {
    "ACTION_BLOCKED",
    "ACTION_EXECUTED",
    "ACTION_FAILED",
    "POLICY_EVALUATED",
    "PERMISSION_GRANTED",
    "PERMISSION_DENIED",
    "APPROVAL_REQUESTED",
    "APPROVAL_GRANTED",
    "APPROVAL_DENIED",
    "APPROVAL_CANCELLED",
    "APPROVAL_EXPIRED",
    "APPROVAL_REVALIDATED",
    "APPROVAL_INVALIDATED",
    "POLICY_CHANGED",
}


def _freeze_value(value: Any) -> Any:
    """Recursively convert mutable containers into tagged immutable tuples."""
    if isinstance(value, dict):
        return ("__dict__", tuple(sorted((str(key), _freeze_value(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)):
        return ("__list__", tuple(_freeze_value(item) for item in value))
    if isinstance(value, set):
        return ("__set__", tuple(sorted(_freeze_value(item) for item in value)))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _thaw_value(value: Any) -> Any:
    if isinstance(value, tuple) and value:
        if value[0] == "__dict__":
            return {key: _thaw_value(item) for key, item in value[1]}
        if value[0] == "__list__":
            return [_thaw_value(item) for item in value[1]]
        if value[0] == "__set__":
            return {_thaw_value(item) for item in value[1]}
    return value


@dataclass(frozen=True)
class _SecurityAuditClaim:
    """Immutable, per-AuditTrail claim with controller-owned issuance state.

    The claim contains snapshots rather than live mutable ActionRequest or
    metadata objects.  The owning controller validates the original
    provenance at the write boundary.
    """

    audit: Any
    event_type: str
    action_snapshot: tuple[Any, ...]
    decision: str
    reason: str
    risk_level: str
    policy_version: int
    approval_id: str
    actor: str
    metadata_snapshot: Any
    claim_id: str
    execution_evidence_snapshot: Any = None
    provenance: Any = field(default=None, repr=False, compare=False)
    provenance_snapshot: Any = field(default=None, repr=False, compare=False)
    issuance_nonce: Any = field(default=None, repr=False, compare=False)
    worker_bound: bool = field(default=False, repr=False, compare=False)

    @property
    def action(self) -> "ActionRequest":
        return _action_from_snapshot(self.action_snapshot)

    @property
    def metadata(self) -> dict[str, Any]:
        return _thaw_value(self.metadata_snapshot)


def _action_snapshot(action: "ActionRequest") -> tuple[Any, ...]:
    return (
        action.task_id,
        action.action_id,
        action.tool_or_skill,
        tuple(action.requested_capabilities),
        tuple(action.declared_capabilities or []),
        action.risk_hint,
        _freeze_value(action.required_resources),
        action.network_required,
        action.filesystem_required,
        action.sensitive_data,
        action.input_fingerprint,
        action.policy_version,
        action.reason,
        action.planned_effect,
        action.action_type,
        action.skill_status,
        action.actor,
        action.goal,
    )


def _action_from_snapshot(snapshot: tuple[Any, ...]) -> "ActionRequest":
    return ActionRequest(
        task_id=snapshot[0],
        action_id=snapshot[1],
        tool_or_skill=snapshot[2],
        requested_capabilities=list(snapshot[3]),
        declared_capabilities=list(snapshot[4]),
        risk_hint=snapshot[5],
        required_resources=_thaw_value(snapshot[6]),
        network_required=snapshot[7],
        filesystem_required=snapshot[8],
        sensitive_data=snapshot[9],
        input_fingerprint=snapshot[10],
        policy_version=snapshot[11],
        reason=snapshot[12],
        planned_effect=snapshot[13],
        action_type=snapshot[14],
        skill_status=snapshot[15],
        actor=snapshot[16],
        goal=snapshot[17],
    )


@dataclass
class ActionRequest:
    task_id: str
    action_id: str
    tool_or_skill: str
    requested_capabilities: list[str] = field(default_factory=list)
    declared_capabilities: list[str] | None = None
    risk_hint: str = ""
    required_resources: dict[str, Any] = field(default_factory=dict)
    network_required: bool = False
    filesystem_required: bool = False
    sensitive_data: bool = False
    input_fingerprint: str = ""
    policy_version: int | None = None
    reason: str = ""
    planned_effect: str = ""
    action_type: str = "tool_execution"
    skill_status: str = ""
    actor: str = "system"
    goal: str = ""

    def __post_init__(self) -> None:
        self.requested_capabilities = sorted(set(self.requested_capabilities))
        if self.declared_capabilities is None:
            self.declared_capabilities = list(self.requested_capabilities)
        else:
            self.declared_capabilities = sorted(set(self.declared_capabilities))
        if not self.input_fingerprint:
            self.input_fingerprint = input_fingerprint(self.required_resources)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_value(asdict(self))


@dataclass(slots=True)
class SecurityPolicy:
    policy_version: int = 1
    approval_required_risks: list[str] = field(default_factory=lambda: ["HIGH", "CRITICAL"])
    medium_requires_approval: bool = False
    always_approval_capabilities: list[str] = field(
        default_factory=lambda: ["SENSITIVE_DATA_ACCESS", "SYSTEM_COMMAND", "SKILL_UPDATE"]
    )
    denied_capabilities: list[str] = field(default_factory=list)
    quarantined_skill_statuses: list[str] = field(default_factory=lambda: ["QUARANTINED", "DEGRADED", "ROLLED_BACK", "REJECTED"])
    approval_ttl_seconds: int = 300
    allow_skill_activation_without_approval: bool = True

    def to_dict(self) -> dict[str, Any]:
        return sanitize_value(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SecurityPolicy":
        return cls(
            policy_version=int(payload.get("policy_version", 1)),
            approval_required_risks=list(payload.get("approval_required_risks", ["HIGH", "CRITICAL"])),
            medium_requires_approval=bool(payload.get("medium_requires_approval", False)),
            always_approval_capabilities=list(
                payload.get("always_approval_capabilities", ["SENSITIVE_DATA_ACCESS", "SYSTEM_COMMAND", "SKILL_UPDATE"])
            ),
            denied_capabilities=list(payload.get("denied_capabilities", [])),
            quarantined_skill_statuses=list(
                payload.get("quarantined_skill_statuses", ["QUARANTINED", "DEGRADED", "ROLLED_BACK", "REJECTED"])
            ),
            approval_ttl_seconds=int(payload.get("approval_ttl_seconds", 300)),
            allow_skill_activation_without_approval=bool(payload.get("allow_skill_activation_without_approval", True)),
        )


class _ImmutableSecurityPolicy:
    """Authoritative policy snapshot; public callers receive mutable copies."""

    __slots__ = ("__weakref__",)

    policy_version = _WriteOnceReference("security_policy_immutable")
    approval_required_risks = _WriteOnceReference("security_policy_immutable")
    medium_requires_approval = _WriteOnceReference("security_policy_immutable")
    always_approval_capabilities = _WriteOnceReference("security_policy_immutable")
    denied_capabilities = _WriteOnceReference("security_policy_immutable")
    quarantined_skill_statuses = _WriteOnceReference("security_policy_immutable")
    approval_ttl_seconds = _WriteOnceReference("security_policy_immutable")
    allow_skill_activation_without_approval = _WriteOnceReference("security_policy_immutable")

    def __init__(self, policy: SecurityPolicy) -> None:
        self.policy_version = int(policy.policy_version)
        self.approval_required_risks = tuple(policy.approval_required_risks)
        self.medium_requires_approval = bool(policy.medium_requires_approval)
        self.always_approval_capabilities = tuple(policy.always_approval_capabilities)
        self.denied_capabilities = tuple(policy.denied_capabilities)
        self.quarantined_skill_statuses = tuple(policy.quarantined_skill_statuses)
        self.approval_ttl_seconds = int(policy.approval_ttl_seconds)
        self.allow_skill_activation_without_approval = bool(policy.allow_skill_activation_without_approval)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_value({
            "policy_version": self.policy_version,
            "approval_required_risks": list(self.approval_required_risks),
            "medium_requires_approval": self.medium_requires_approval,
            "always_approval_capabilities": list(self.always_approval_capabilities),
            "denied_capabilities": list(self.denied_capabilities),
            "quarantined_skill_statuses": list(self.quarantined_skill_statuses),
            "approval_ttl_seconds": self.approval_ttl_seconds,
            "allow_skill_activation_without_approval": self.allow_skill_activation_without_approval,
        })


OWNER_EFFECTS = {"ALLOW", "DENY", "REQUIRE_APPROVAL"}
OWNER_SCOPES = {"GLOBAL", "GOAL", "TOOL", "SKILL", "CAPABILITY"}
OWNER_CONDITION_KEYS = {"risk_levels", "action_types", "capabilities", "tool_names", "goal"}


@dataclass
class OwnerPermissionRule:
    rule_id: str
    effect: str
    scope_type: str
    scope_key: str
    conditions: dict[str, Any] = field(default_factory=dict)
    expires_at: str = ""
    policy_version: int = 0
    actor: str = ""
    reason: str = ""
    status: str = "ACTIVE"
    created_at: str = field(default_factory=lambda: _now())

    def __post_init__(self) -> None:
        if not self.rule_id or len(self.rule_id) > 128:
            raise ValueError("invalid_owner_rule_id")
        if self.effect not in OWNER_EFFECTS:
            raise ValueError(f"invalid_owner_effect:{self.effect}")
        if self.scope_type not in OWNER_SCOPES:
            raise ValueError(f"invalid_owner_scope:{self.scope_type}")
        if not self.scope_key or len(self.scope_key) > 256:
            raise ValueError("invalid_owner_scope_key")
        if self.scope_type == "GLOBAL" and self.scope_key != "*":
            raise ValueError("global_scope_key_must_be_star")
        if not isinstance(self.conditions, dict) or set(self.conditions) - OWNER_CONDITION_KEYS:
            raise ValueError("unsupported_owner_condition")
        for key, value in self.conditions.items():
            if key in {"risk_levels", "action_types", "capabilities", "tool_names"}:
                if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
                    raise ValueError(f"invalid_owner_condition:{key}")
            elif key == "goal" and (not isinstance(value, str) or not value or len(value) > 4000):
                raise ValueError("invalid_owner_condition:goal")
        if self.expires_at:
            try:
                self.expires_at = _parse_time(self.expires_at).astimezone(timezone.utc).isoformat()
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid_owner_expiry") from exc
        if self.policy_version < 0:
            raise ValueError("invalid_owner_policy_version")
        if self.status not in {"ACTIVE", "DISABLED", "EXPIRED"}:
            raise ValueError("invalid_owner_rule_status")
        if not self.reason or len(self.reason) > 1000:
            raise ValueError("owner_rule_reason_required")

    def to_dict(self) -> dict[str, Any]:
        return sanitize_value(asdict(self))


@dataclass
class PolicyDecision:
    decision: str
    risk_level: str
    task_id: str
    action_id: str
    tool_or_skill: str
    requested_capabilities: list[str]
    policy_version: int
    approval_required: bool = False
    approval_id: str = ""
    reasons: list[str] = field(default_factory=list)
    evaluated_at: str = field(default_factory=lambda: _now())
    input_fingerprint: str = ""
    declared_capabilities: list[str] = field(default_factory=list)
    goal: str = ""
    action_type: str = ""
    skill_status: str = ""

    def __post_init__(self) -> None:
        if self.decision not in POLICY_DECISIONS:
            raise ValueError(f"invalid_policy_decision:{self.decision}")
        if self.risk_level not in RISK_LEVELS:
            raise ValueError(f"invalid_risk_level:{self.risk_level}")

    @property
    def allowed(self) -> bool:
        return self.decision == "ALLOW"

    def to_dict(self) -> dict[str, Any]:
        return sanitize_value(asdict(self))


@dataclass(frozen=True)
class AuthorizationEvidenceSnapshot:
    """Immutable authorization data used after a boundary decision is issued."""

    decision: str
    risk_level: str
    task_id: str
    action_id: str
    tool_or_skill: str
    requested_capabilities: tuple[str, ...]
    policy_version: int
    approval_id: str
    input_fingerprint: str
    declared_capabilities: tuple[str, ...]
    goal: str
    action_type: str
    skill_status: str

    @classmethod
    def from_decision(cls, decision: PolicyDecision) -> "AuthorizationEvidenceSnapshot":
        return cls(
            decision=decision.decision,
            risk_level=decision.risk_level,
            task_id=decision.task_id,
            action_id=decision.action_id,
            tool_or_skill=decision.tool_or_skill,
            requested_capabilities=tuple(sorted(decision.requested_capabilities)),
            policy_version=decision.policy_version,
            approval_id=decision.approval_id,
            input_fingerprint=decision.input_fingerprint,
            declared_capabilities=tuple(sorted(decision.declared_capabilities)),
            goal=decision.goal,
            action_type=decision.action_type,
            skill_status=decision.skill_status,
        )


@dataclass(frozen=True)
class TrustedOperationBinding:
    """Opaque controller-issued binding for a concrete execution callback."""

    controller: Any = field(repr=False, compare=False)
    action_binding: tuple[Any, ...]
    operation: Any = field(repr=False, compare=False)
    kind: str
    token: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class ExecutionReceipt:
    """One-shot proof that a controller-bound operation actually returned."""

    receipt_id: str
    authorization_id: int
    action_snapshot: tuple[Any, ...]
    authorization: AuthorizationEvidenceSnapshot
    success: bool
    metadata_snapshot: Any
    _state: Any = field(repr=False, compare=False)
    operation_binding: Any = field(default=None, repr=False, compare=False)

    @property
    def action(self) -> "ActionRequest":
        return _action_from_snapshot(self.action_snapshot)

    @property
    def metadata(self) -> dict[str, Any]:
        return _thaw_value(self.metadata_snapshot)


@dataclass
class AuthorizedExecutionResult:
    """Result of authorization, the real operation, and audit recording."""

    decision: PolicyDecision
    result: Any = None
    exception: BaseException | None = None
    receipt: ExecutionReceipt | None = None
    blocked: bool = False

    @property
    def allowed(self) -> bool:
        return self.decision.allowed


def _provenance_snapshot(provenance: Any) -> Any:
    """Capture immutable state for a claim's original provenance object."""
    if isinstance(provenance, AuthorizationEvidenceSnapshot):
        return (
            "AuthorizationEvidenceSnapshot",
            provenance.decision,
            provenance.risk_level,
            provenance.task_id,
            provenance.action_id,
            provenance.tool_or_skill,
            provenance.requested_capabilities,
            provenance.policy_version,
            provenance.approval_id,
            provenance.input_fingerprint,
            provenance.declared_capabilities,
            provenance.goal,
            provenance.action_type,
            provenance.skill_status,
        )
    if isinstance(provenance, PolicyDecision):
        return ("PolicyDecision", _freeze_value(provenance.to_dict()))
    if isinstance(provenance, ApprovalRequest):
        return ("ApprovalRequest", _freeze_value(provenance.to_dict()))
    if isinstance(provenance, (SecurityPolicy, _ImmutableSecurityPolicy)):
        return ("SecurityPolicy", _freeze_value(provenance.to_dict()))
    return ("identity", id(provenance))


class PolicyEngine:
    """Deterministic policy authority; it never consumes free-form LLM approval text."""

    store = _WriteOnceReference("policy_store_immutable")
    _policy = _WriteOnceReference("policy_authority_immutable")
    __slots__ = ("__weakref__",)

    def __init__(self, policy: SecurityPolicy | None = None, *, store: Any | None = None) -> None:
        self.store = store
        if policy is None and store is not None and hasattr(store, "get_latest_security_policy"):
            row = store.get_latest_security_policy()
            if row and row.get("policy_json"):
                try:
                    policy = SecurityPolicy.from_dict(json.loads(row["policy_json"]))
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError("corrupted_security_policy") from exc
        # Keep the authoritative policy private.  The public property returns
        # a defensive snapshot so callers cannot mutate authorization rules in
        # place without the monotonic update path and its audit event.
        self._policy = _ImmutableSecurityPolicy(SecurityPolicy.from_dict((policy or SecurityPolicy()).to_dict()))
        self._persist_policy()

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_policy" and hasattr(self, "_policy"):
            raise AttributeError("policy_authority_immutable")
        object.__setattr__(self, name, value)

    @property
    def policy(self) -> SecurityPolicy:
        return SecurityPolicy.from_dict(self._policy.to_dict())

    def _owner_rules(self) -> list[dict[str, Any]]:
        if self.store is None or not hasattr(self.store, "owner_permission_rules"):
            return []
        return self.store.owner_permission_rules(active_only=True)

    @staticmethod
    def _owner_rule_matches(rule: dict[str, Any], action: ActionRequest, risk: str) -> bool:
        scope = str(rule.get("scope_type", ""))
        key = str(rule.get("scope_key", ""))
        requested = set(action.requested_capabilities)
        if scope == "GLOBAL":
            matched = key == "*"
        elif scope == "GOAL":
            matched = bool(getattr(action, "goal", "")) and key == str(getattr(action, "goal", ""))
        elif scope == "CAPABILITY":
            matched = key in requested
        elif scope in {"TOOL", "SKILL"}:
            matched = key == action.tool_or_skill
        else:
            return False
        if not matched:
            return False
        conditions = rule.get("conditions", {})
        if not isinstance(conditions, dict):
            return False
        checks = {
            "risk_levels": risk,
            "action_types": action.action_type,
            "capabilities": requested,
            "tool_names": action.tool_or_skill,
            "goal": getattr(action, "goal", ""),
        }
        for condition, expected in conditions.items():
            actual = checks.get(condition)
            if condition == "capabilities":
                if not isinstance(expected, list) or not set(expected).issubset(requested):
                    return False
            elif condition == "goal":
                if actual != expected:
                    return False
            elif not isinstance(expected, list) or actual not in expected:
                return False
        return True

    def _owner_decision(self, action: ActionRequest, risk: str) -> tuple[str | None, list[str], int]:
        matches = [rule for rule in self._owner_rules() if self._owner_rule_matches(rule, action, risk)]
        if not matches:
            return None, [], self._current_policy_version()
        specificity = {"GLOBAL": 0, "GOAL": 1, "CAPABILITY": 2, "TOOL": 3, "SKILL": 3}
        # Decision type is authoritative before scope specificity: any matching
        # DENY wins, then any REQUIRE_APPROVAL, then ALLOW. Specificity and
        # policy version break ties within the selected decision type.
        for effect in ("DENY", "REQUIRE_APPROVAL", "ALLOW"):
            candidates = [rule for rule in matches if rule.get("effect") == effect]
            if candidates:
                chosen = max(
                    candidates,
                    key=lambda rule: (
                        specificity.get(str(rule.get("scope_type")), -1),
                        int(rule.get("policy_version", 0)),
                    ),
                )
                break
        version = max(self._current_policy_version(), *(int(rule.get("policy_version", 0)) for rule in matches))
        return str(chosen.get("effect")), [f"owner_rule:{chosen.get('rule_id')}", str(chosen.get("reason", ""))], version

    def _current_policy_version(self) -> int:
        if self.store is not None and hasattr(self.store, "latest_owner_policy_version"):
            return max(self._policy.policy_version, int(self.store.latest_owner_policy_version()))
        return self._policy.policy_version

    def evaluate(self, action: ActionRequest) -> PolicyDecision:
        requested = set(action.requested_capabilities)
        declared = set(action.declared_capabilities)
        reasons: list[str] = []
        unknown = sorted((requested | declared) - CAPABILITIES)
        if unknown:
            return self._decision(action, "DENY", "CRITICAL", [f"unknown_capability:{item}" for item in unknown])
        if not requested.issubset(declared):
            missing = sorted(requested - declared)
            return self._decision(action, "DENY", "CRITICAL", [f"undeclared_capability:{item}" for item in missing])
        denied = sorted(requested.intersection(self.policy.denied_capabilities))
        if denied:
            return self._decision(action, "DENY", "CRITICAL", [f"policy_denied_capability:{item}" for item in denied])
        if action.action_type in {"modify_policy", "grant_permission", "bypass_approval"}:
            return self._decision(action, "DENY", "CRITICAL", [f"forbidden_action_type:{action.action_type}"])
        if action.skill_status and action.skill_status in self.policy.quarantined_skill_statuses:
            return self._decision(action, "QUARANTINE", "CRITICAL", [f"skill_status:{action.skill_status}"])
        if action.network_required and "NETWORK_ACCESS" not in requested and "EXTERNAL_API" not in requested:
            return self._decision(action, "DENY", "HIGH", ["network_requirement_undeclared"])
        if action.filesystem_required and not requested.intersection({"READ_WORKSPACE", "WRITE_WORKSPACE"}):
            return self._decision(action, "DENY", "HIGH", ["filesystem_requirement_undeclared"])
        if action.sensitive_data and "SENSITIVE_DATA_ACCESS" not in requested:
            return self._decision(action, "DENY", "CRITICAL", ["sensitive_data_capability_undeclared"])

        risk = self._risk_for(action, requested)
        if (
            action.action_type in {"tool_execution", "skill_invocation", "skill_activation"}
            and not action.goal
            and any(str(rule.get("scope_type", "")) == "GOAL" for rule in self._owner_rules())
        ):
            return self._decision(action, "DENY", risk, ["goal_context_required_for_owner_policy"])
        owner_effect, owner_reasons, effective_version = self._owner_decision(action, risk)
        if owner_effect == "DENY":
            decision = self._decision(action, "DENY", risk, owner_reasons)
            decision.policy_version = effective_version
            return decision
        if owner_effect == "REQUIRE_APPROVAL":
            decision = self._decision(action, "REQUIRE_APPROVAL", risk, owner_reasons, approval_required=True)
            decision.policy_version = effective_version
            return decision
        owner_allow = owner_effect == "ALLOW"
        requires_approval = risk in set(self.policy.approval_required_risks)
        if risk == "MEDIUM" and self.policy.medium_requires_approval:
            requires_approval = True
        if requested.intersection(self.policy.always_approval_capabilities):
            requires_approval = True
        if action.action_type == "skill_activation" and self.policy.allow_skill_activation_without_approval:
            requires_approval = False
        if owner_allow:
            requires_approval = False
        if requires_approval:
            reasons.append(f"approval_required_for_risk:{risk}")
            return self._decision(action, "REQUIRE_APPROVAL", risk, reasons, approval_required=True)
        return self._decision(action, "ALLOW", risk, reasons)

    def update_policy(self, policy: SecurityPolicy) -> SecurityPolicy:
        if policy.policy_version <= self._policy.policy_version:
            raise ValueError("policy_version_must_increase")
        descriptor = type(self).__dict__["_policy"]
        _replace_write_once_reference(
            descriptor,
            self,
            _ImmutableSecurityPolicy(SecurityPolicy.from_dict(policy.to_dict())),
            token=_WRITE_ONCE_UPDATE_TOKEN,
        )
        self._persist_policy()
        return self.policy

    def _risk_for(self, action: ActionRequest, capabilities: set[str]) -> str:
        if capabilities.intersection({"SENSITIVE_DATA_ACCESS", "SYSTEM_COMMAND"}) or action.sensitive_data:
            derived = "CRITICAL"
        elif capabilities.intersection({"NETWORK_ACCESS", "EXTERNAL_API", "SKILL_UPDATE"}):
            derived = "HIGH"
        elif capabilities.intersection({"WRITE_WORKSPACE", "DATABASE_WRITE", "LONG_RUNNING_TASK", "SKILL_ACTIVATION"}):
            derived = "MEDIUM"
        else:
            derived = "LOW"
        hint = action.risk_hint.upper()
        if hint not in RISK_LEVELS:
            return derived
        return max((derived, hint), key=lambda value: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(value))

    def _decision(
        self,
        action: ActionRequest,
        decision: str,
        risk: str,
        reasons: list[str],
        *,
        approval_required: bool = False,
    ) -> PolicyDecision:
        return PolicyDecision(
            decision=decision,
            risk_level=risk,
            task_id=action.task_id,
            action_id=action.action_id,
            tool_or_skill=action.tool_or_skill,
            requested_capabilities=action.requested_capabilities,
            policy_version=self._current_policy_version(),
            approval_required=approval_required,
            reasons=reasons,
            input_fingerprint=action.input_fingerprint,
            declared_capabilities=list(action.declared_capabilities),
            goal=action.goal,
            action_type=action.action_type,
            skill_status=action.skill_status,
        )

    def _persist_policy(self) -> None:
        if self.store is not None and hasattr(self.store, "save_security_policy"):
            self.store.save_security_policy(self._policy.to_dict())


@dataclass
class ApprovalRequest:
    approval_id: str
    task_id: str
    action_id: str
    tool_or_skill: str
    requested_capabilities: list[str]
    risk_level: str
    reason: str
    planned_effect: str
    input_fingerprint: str
    policy_version: int
    requested_at: str
    expires_at: str
    status: str = "PENDING"
    decided_at: str = ""
    actor: str = "system"
    goal: str = ""
    legacy: bool = False

    def __post_init__(self) -> None:
        if self.status not in APPROVAL_STATUSES:
            raise ValueError(f"invalid_approval_status:{self.status}")

    def to_dict(self) -> dict[str, Any]:
        return sanitize_value(asdict(self))


_APPROVAL_STATE: weakref.WeakKeyDictionary[Any, dict[str, ApprovalRequest]] = weakref.WeakKeyDictionary()
_APPROVAL_STATE_LOCK = threading.RLock()


def _approval_records(gate: Any) -> dict[str, ApprovalRequest]:
    with _APPROVAL_STATE_LOCK:
        records = _APPROVAL_STATE.get(gate)
        if records is None:
            records = {}
            _APPROVAL_STATE[gate] = records
        return records


class ApprovalGate:
    """Explicit human approval state machine with exact action binding and expiration."""

    __slots__ = ("store", "event_logger", "policy_version_provider", "__weakref__")

    SENSITIVE = {"send_message", "publish", "payment", "delete_data", "account_change", "external_irreversible"}

    def __init__(
        self,
        *,
        store: Any | None = None,
        event_logger: Callable[[str, ApprovalRequest], None] | None = None,
        policy_version_provider: Callable[[], int] | None = None,
    ) -> None:
        self.store = store
        self.event_logger = event_logger
        self.policy_version_provider = policy_version_provider

    @property
    def _requests(self) -> MappingProxyType:
        """Expose only detached read-only views for legacy introspection."""
        with _APPROVAL_STATE_LOCK:
            return MappingProxyType(
                {approval_id: self._public_view(request) for approval_id, request in _approval_records(self).items()}
            )

    def _publish(self, request: ApprovalRequest) -> None:
        with _APPROVAL_STATE_LOCK:
            _approval_records(self)[request.approval_id] = ApprovalRequest(**request.to_dict())

    def set_policy_version_provider(self, provider: Callable[[], int]) -> None:
        self.policy_version_provider = provider

    @staticmethod
    def _public_view(request: ApprovalRequest) -> ApprovalRequest:
        """Return a detached view so callers cannot mutate the authoritative request."""
        return ApprovalRequest(**request.to_dict())

    def request(self, action: ActionRequest, decision: PolicyDecision, *, ttl_seconds: int) -> ApprovalRequest:
        now = datetime.now(timezone.utc)
        request = ApprovalRequest(
            approval_id=str(uuid.uuid4()),
            task_id=action.task_id,
            action_id=action.action_id,
            tool_or_skill=action.tool_or_skill,
            requested_capabilities=list(action.requested_capabilities),
            risk_level=decision.risk_level,
            reason=action.reason or "; ".join(decision.reasons) or "Policy requires explicit human approval",
            planned_effect=action.planned_effect,
            input_fingerprint=action.input_fingerprint,
            policy_version=decision.policy_version,
            requested_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=max(1, ttl_seconds))).isoformat(),
            actor=action.actor,
            goal=action.goal,
        )
        self._persist(request, enforce_lease=True)
        self._publish(request)
        self._emit("APPROVAL_REQUESTED", self._public_view(request))
        return self._public_view(request)

    def approve(self, approval_id: str, *, actor: str = "human") -> ApprovalRequest:
        request = self._get(approval_id)
        self._expire_if_needed(request)
        if request.status != "PENDING":
            raise ValueError(f"approval_not_pending:{request.status}")
        request.status = "APPROVED"
        request.decided_at = _now()
        request.actor = actor
        self._persist(request)
        self._publish(request)
        self._emit("APPROVAL_GRANTED", self._public_view(request))
        return self._public_view(request)

    def deny(self, approval_id: str, *, actor: str = "human") -> ApprovalRequest:
        request = self._get(approval_id)
        self._expire_if_needed(request)
        if request.status != "PENDING":
            raise ValueError(f"approval_not_pending:{request.status}")
        request.status = "DENIED"
        request.decided_at = _now()
        request.actor = actor
        self._persist(request)
        self._publish(request)
        self._emit("APPROVAL_DENIED", self._public_view(request))
        return self._public_view(request)

    def cancel(self, approval_id: str, *, actor: str = "system") -> ApprovalRequest:
        request = self._get(approval_id)
        self._expire_if_needed(request)
        if request.status != "PENDING":
            raise ValueError(f"approval_not_pending:{request.status}")
        request.status = "CANCELLED"
        request.decided_at = _now()
        request.actor = actor
        self._persist(request)
        self._publish(request)
        self._emit("APPROVAL_CANCELLED", self._public_view(request))
        return self._public_view(request)

    def get(self, approval_id: str) -> ApprovalRequest | None:
        try:
            request = self._get(approval_id)
        except KeyError:
            return None
        self._expire_if_needed(request)
        # Public callers receive a detached view.  Security internals use
        # _get() so a caller cannot turn PENDING into APPROVED in place.
        return self._public_view(request)

    def valid_for(
        self,
        approval_id: str,
        action: ActionRequest,
        policy_version: int,
        risk_level: str | None = None,
    ) -> bool:
        request = self.get(approval_id)
        return bool(request and request.status == "APPROVED" and self.bound_for(approval_id, action, policy_version, risk_level))

    def bound_for(
        self,
        approval_id: str,
        action: ActionRequest,
        policy_version: int,
        risk_level: str | None = None,
    ) -> bool:
        request = self.get(approval_id)
        if request is None:
            return False
        return self._bound_request(request, action, policy_version, risk_level)

    def _bound_request(
        self,
        request: ApprovalRequest,
        action: ActionRequest,
        policy_version: int,
        risk_level: str | None = None,
    ) -> bool:
        """Check a controller-held request without refreshing its identity."""
        self._expire_if_needed(request)
        if request.legacy or not risk_level:
            return False
        if self.policy_version_provider is None:
            return False
        try:
            if int(policy_version) != int(self.policy_version_provider()):
                return False
        except (TypeError, ValueError, RuntimeError):
            return False
        binding_matches = (
            request.task_id == action.task_id
            and request.action_id == action.action_id
            and request.tool_or_skill == action.tool_or_skill
            and request.requested_capabilities == action.requested_capabilities
            and request.goal == action.goal
        )
        binding_matches = binding_matches and request.risk_level == risk_level
        return binding_matches and request.input_fingerprint == action.input_fingerprint and request.policy_version == policy_version

    def check(self, action: str, *, approved: bool = False) -> ApprovalRequest | None:
        """Legacy request constructor; callers must use SecurityController.authorize()."""
        if action not in self.SENSITIVE:
            return None
        request = ApprovalRequest(
            approval_id=str(uuid.uuid4()),
            task_id="legacy",
            action_id=action,
            tool_or_skill=action,
            requested_capabilities=["SENSITIVE_DATA_ACCESS"],
            risk_level="CRITICAL",
            reason="Sensitive external action",
            planned_effect="Action completes only after explicit approval",
            input_fingerprint=input_fingerprint({"action": action}),
            policy_version=1,
            requested_at=_now(),
            expires_at=(datetime.now(timezone.utc) + timedelta(seconds=300)).isoformat(),
            status="PENDING",
            legacy=True,
        )
        self._persist(request)
        self._publish(request)
        return self._public_view(request)

    def allow(
        self,
        request: ApprovalRequest,
        *,
        action: ActionRequest | None = None,
        policy_version: int | None = None,
        risk_level: str | None = None,
    ) -> bool:
        """Legacy helper; fail closed unless exact binding and current version are supplied."""
        if action is None or policy_version is None or not risk_level:
            return False
        try:
            current = self._get(request.approval_id)
        except KeyError:
            return False
        self._expire_if_needed(current)
        if current.status != "APPROVED":
            return False
        return self.bound_for(request.approval_id, action, policy_version, risk_level)

    def _get(self, approval_id: str) -> ApprovalRequest:
        if self.store is not None and hasattr(self.store, "get_approval_request"):
            payload = self.store.get_approval_request(approval_id)
            if not payload:
                raise KeyError(f"approval_not_found:{approval_id}")
            required_binding_fields = {
                "approval_id", "task_id", "action_id", "tool_or_skill",
                "requested_capabilities", "risk_level", "input_fingerprint",
                "policy_version", "requested_at", "expires_at", "goal",
            }
            if not required_binding_fields.issubset(payload):
                raise KeyError(f"incomplete_approval_binding:{approval_id}")
            request = ApprovalRequest(**payload)
            self._publish(request)
            return self._public_view(request)
        with _APPROVAL_STATE_LOCK:
            request = _approval_records(self).get(approval_id)
            if request is not None:
                return self._public_view(request)
        raise KeyError(f"approval_not_found:{approval_id}")

    def _is_current(self, request: ApprovalRequest) -> bool:
        try:
            current = self._get(request.approval_id)
        except (KeyError, TypeError, ValueError):
            return False
        return current.to_dict() == request.to_dict()

    def _expire_if_needed(self, request: ApprovalRequest) -> None:
        if request.status not in {"PENDING", "APPROVED"}:
            return
        try:
            expired = _parse_time(request.expires_at) <= datetime.now(timezone.utc)
        except (TypeError, ValueError):
            # A persisted malformed expiry is not an approval.  Mark it
            # unusable and keep the failure inside the normal fail-closed path.
            expired = True
        if expired:
            request.status = "EXPIRED"
            request.decided_at = _now()
            self._persist(request)
            self._publish(request)
            self._emit("APPROVAL_EXPIRED", self._public_view(request))

    def _persist(self, request: ApprovalRequest, *, enforce_lease: bool = False) -> None:
        if self.store is not None and hasattr(self.store, "save_approval_request"):
            try:
                from private_agent.storage import Store

                if isinstance(self.store, Store):
                    if request.status == "PENDING":
                        self.store.save_approval_request(request.to_dict(), enforce_lease=enforce_lease)
                    else:
                        self.store._save_approval_transition(
                            request.to_dict(), gate=self, enforce_lease=enforce_lease
                        )
                    return
            except (ImportError, TypeError):
                pass
            self.store.save_approval_request(request.to_dict(), enforce_lease=enforce_lease)

    def _emit(self, event_type: str, request: ApprovalRequest) -> None:
        if self.event_logger is not None:
            self.event_logger(event_type, request)


class AuditTrail:
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_security_owner" and getattr(self, "_owner_bound", False):
            raise AttributeError("security_owner_immutable")
        if name == "_owner_bound" and getattr(self, "_owner_bound", False) and value is not True:
            raise AttributeError("security_owner_binding_immutable")
        if getattr(self, "_security_state_sealed", False) and name in {
            "_claim_register",
            "_claim_lookup",
            "_claim_consume",
            "_claim_issuer",
        }:
            raise AttributeError("security_state_immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        store: Any | None = None,
        observability: Any | None = None,
        _security_owner: Any | None = None,
    ) -> None:
        object.__setattr__(self, "_owner_bound", False)
        self.store = store
        self.observability = observability
        self._security_owner = _security_owner
        self._owner_bound = _security_owner is not None
        self._events: list[dict[str, Any]] = []
        self._claim_issuer = object()
        self._claim_lock = threading.RLock()
        self._active_claims: dict[str, _SecurityAuditClaim] = {}
        self._writing_claim: _SecurityAuditClaim | None = None
        claim_records: dict[str, _SecurityAuditClaim] = {}
        claim_records_lock = threading.RLock()

        def issue_security_claim(*args: Any, **kwargs: Any) -> _SecurityAuditClaim:
            # Registration is performed in this closure immediately after the
            # internal builder returns.  There is deliberately no public
            # registration callable that accepts arbitrary claim objects.
            claim = self._build_security_claim(*args, **kwargs)
            with claim_records_lock:
                claim_records[claim.claim_id] = claim
            return claim

        def write_security_claim(claim: _SecurityAuditClaim) -> dict[str, Any]:
            with self._claim_lock:
                if (
                    not isinstance(claim, _SecurityAuditClaim)
                    or claim.audit is not self
                    or claim.event_type not in TRUSTED_SECURITY_AUDIT_EVENTS
                    or claim.issuance_nonce is None
                ):
                    raise ValueError("security_audit_claim_untrusted")
                with claim_records_lock:
                    if claim_records.get(claim.claim_id) is not claim:
                        raise ValueError("security_audit_claim_untrusted")
                self._validate_issued_claim(claim)
                try:
                    self._writing_claim = claim
                    return self._record(
                        claim.event_type,
                        claim.action,
                        decision=claim.decision,
                        reason=claim.reason,
                        risk_level=claim.risk_level,
                        policy_version=claim.policy_version,
                        approval_id=claim.approval_id,
                        actor=claim.actor,
                        metadata=claim.metadata,
                        _claim=claim,
                        _requires_lease=claim.worker_bound,
                    )
                finally:
                    # Claims are one-shot even when persistence fails.
                    self._writing_claim = None
                    self._active_claims.pop(claim.claim_id, None)
                    with claim_records_lock:
                        claim_records.pop(claim.claim_id, None)
                    release = getattr(self._security_owner, "_release_observed_block_provenance", None)
                    if callable(release):
                        release(claim.provenance)

        # These closures retain the issuance records; neither registration nor
        # the registry itself is exposed as an object attribute.
        self._issue_security_claim = issue_security_claim
        self._write_security_claim = write_security_claim
        self._security_state_sealed = True

    def _bind_security_owner(self, owner: Any) -> None:
        if owner is None:
            raise ValueError("security_owner_required")
        if self._owner_bound:
            if self._security_owner is owner:
                return
            raise ValueError("security_owner_immutable")
        self._security_owner = owner
        self._owner_bound = True

    def _build_security_claim(
        self,
        issuer: object,
        event_type: str,
        action: ActionRequest,
        *,
        decision: str,
        reason: str,
        risk_level: str,
        policy_version: int,
        approval_id: str = "",
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
        execution_evidence: Any | None = None,
        provenance: Any | None = None,
    ) -> _SecurityAuditClaim:
        if issuer is not self._claim_issuer:
            raise ValueError("security_audit_issuer_untrusted")
        if event_type not in TRUSTED_SECURITY_AUDIT_EVENTS:
            raise ValueError(f"unsupported_security_audit_claim:{event_type}")
        action_binding = (
            action.task_id,
            action.action_id,
            action.tool_or_skill,
            tuple(sorted(action.requested_capabilities)),
            tuple(sorted(action.declared_capabilities or [])),
            action.input_fingerprint,
            action.goal,
            action.action_type,
            action.skill_status,
        )
        if event_type in {"ACTION_EXECUTED", "ACTION_FAILED"}:
            if not isinstance(execution_evidence, ExecutionReceipt):
                raise ValueError("execution_receipt_required")
            if execution_evidence.authorization.decision != "ALLOW":
                raise ValueError("execution_receipt_invalid")
            snapshot_binding = (
                execution_evidence.authorization.task_id,
                execution_evidence.authorization.action_id,
                execution_evidence.authorization.tool_or_skill,
                execution_evidence.authorization.requested_capabilities,
                execution_evidence.authorization.declared_capabilities,
                execution_evidence.authorization.input_fingerprint,
                execution_evidence.authorization.goal,
                execution_evidence.authorization.action_type,
                execution_evidence.authorization.skill_status,
            )
            if snapshot_binding != (
                action.task_id,
                action.action_id,
                action.tool_or_skill,
                tuple(sorted(action.requested_capabilities)),
                tuple(sorted(action.declared_capabilities or [])),
                action.input_fingerprint,
                action.goal,
                action.action_type,
                action.skill_status,
            ):
                raise ValueError("execution_evidence_mismatch")
            if execution_evidence.authorization.policy_version != policy_version:
                raise ValueError("execution_receipt_policy_mismatch")
            if approval_id != execution_evidence.authorization.approval_id:
                raise ValueError("execution_receipt_approval_mismatch")
            if event_type == "ACTION_EXECUTED" and not execution_evidence.success:
                raise ValueError("execution_receipt_result_mismatch")
            if event_type == "ACTION_FAILED" and execution_evidence.success:
                raise ValueError("execution_receipt_result_mismatch")
            if provenance is not execution_evidence:
                raise ValueError("execution_receipt_provenance_mismatch")
        if provenance is None:
            raise ValueError("security_audit_provenance_required")
        owner = self._security_owner
        if isinstance(provenance, PolicyDecision):
            refresh = getattr(owner, "_refresh_issued_policy_decision", None)
            if not callable(refresh) or not refresh(provenance):
                raise ValueError("security_audit_provenance_untrusted")
        if owner is None or not owner._accept_audit_provenance(
            provenance,
            action=action,
            event_type=event_type,
            decision=decision,
            risk_level=risk_level,
            policy_version=policy_version,
            approval_id=approval_id,
        ):
            raise ValueError("security_audit_provenance_untrusted")
        if event_type not in {"ACTION_EXECUTED", "ACTION_FAILED"}:
            if isinstance(provenance, AuthorizationEvidenceSnapshot):
                if event_type != "ACTION_BLOCKED" or provenance.decision != "ALLOW":
                    raise ValueError("authorization_provenance_event_mismatch")
                snapshot_binding = (
                    provenance.task_id,
                    provenance.action_id,
                    provenance.tool_or_skill,
                    provenance.requested_capabilities,
                    provenance.declared_capabilities,
                    provenance.input_fingerprint,
                    provenance.goal,
                    provenance.action_type,
                    provenance.skill_status,
                )
                if (
                    snapshot_binding != action_binding
                    or provenance.policy_version != policy_version
                    or provenance.risk_level != risk_level
                    or provenance.approval_id != approval_id
                ):
                    raise ValueError("authorization_provenance_mismatch")
            elif isinstance(provenance, PolicyDecision):
                decision_binding = (
                    provenance.task_id,
                    provenance.action_id,
                    provenance.tool_or_skill,
                    tuple(sorted(provenance.requested_capabilities)),
                    tuple(sorted(provenance.declared_capabilities)),
                    provenance.input_fingerprint,
                    provenance.goal,
                    provenance.action_type,
                    provenance.skill_status,
                )
                if decision_binding != action_binding or provenance.policy_version != policy_version or provenance.risk_level != risk_level:
                    raise ValueError("policy_decision_provenance_mismatch")
                expected_decisions = {
                    "POLICY_EVALUATED": {provenance.decision},
                    "PERMISSION_GRANTED": {"ALLOW"},
                    "PERMISSION_DENIED": {"DENY"},
                    "ACTION_BLOCKED": {"DENY", "QUARANTINE", "REQUIRE_APPROVAL"},
                }
                if event_type not in expected_decisions or decision not in expected_decisions[event_type]:
                    raise ValueError("policy_decision_event_mismatch")
            elif isinstance(provenance, ApprovalRequest):
                approval_binding = (
                    provenance.task_id,
                    provenance.action_id,
                    provenance.tool_or_skill,
                    tuple(sorted(provenance.requested_capabilities)),
                    tuple(sorted(provenance.requested_capabilities)),
                    provenance.input_fingerprint,
                    provenance.goal,
                    action.action_type,
                    action.skill_status,
                )
                if event_type == "APPROVAL_INVALIDATED":
                    invalidation_binding_matches = (
                        provenance.approval_id == approval_id
                        and provenance.policy_version <= policy_version
                    )
                    if not invalidation_binding_matches:
                        raise ValueError("approval_provenance_mismatch")
                elif (
                    event_type not in {
                        "APPROVAL_REQUESTED",
                        "APPROVAL_GRANTED",
                        "APPROVAL_DENIED",
                        "APPROVAL_CANCELLED",
                        "APPROVAL_EXPIRED",
                        "APPROVAL_REVALIDATED",
                        "PERMISSION_GRANTED",
                    }
                    or approval_binding != action_binding
                    or provenance.approval_id != approval_id
                    or provenance.policy_version != policy_version
                    or provenance.risk_level != risk_level
                    or (event_type == "PERMISSION_GRANTED" and decision != "ALLOW")
                ):
                    raise ValueError("approval_provenance_mismatch")
            elif event_type == "ACTION_BLOCKED" and owner._is_observed_block_provenance(provenance):
                if metadata is None or metadata.get("provenance_kind") not in {None, "observed_only", "trusted_authorization"}:
                    raise ValueError("observed_block_provenance_mismatch")
            elif event_type == "POLICY_CHANGED" and isinstance(provenance, (SecurityPolicy, _ImmutableSecurityPolicy)):
                if action.action_type != "policy_update" or provenance.policy_version != policy_version:
                    raise ValueError("policy_change_provenance_mismatch")
            else:
                raise ValueError("security_audit_provenance_untrusted")
        worker_bound = False
        lease_context = getattr(self.store, "has_active_lease_context", None)
        if callable(lease_context) and action.task_id:
            try:
                worker_bound = bool(lease_context(action.task_id))
            except Exception:
                worker_bound = False
        claim = _SecurityAuditClaim(
            audit=self,
            event_type=event_type,
            action_snapshot=_action_snapshot(action),
            decision=decision,
            reason=reason,
            risk_level=risk_level,
            policy_version=policy_version,
            approval_id=approval_id,
            actor=actor,
            metadata_snapshot=_freeze_value(metadata or {}),
            claim_id=str(uuid.uuid4()),
            execution_evidence_snapshot=execution_evidence,
            provenance=provenance,
            provenance_snapshot=_provenance_snapshot(provenance),
            issuance_nonce=object(),
            worker_bound=worker_bound,
        )
        with self._claim_lock:
            self._active_claims[claim.claim_id] = claim
        return claim

    def _validate_issued_claim(self, claim: _SecurityAuditClaim) -> None:
        """Revalidate provenance and the controller-created issuance nonce."""
        provenance = claim.provenance
        owner = self._security_owner
        if not isinstance(owner, SecurityController):
            raise ValueError("security_audit_owner_untrusted")
        if claim.issuance_nonce is None or _provenance_snapshot(provenance) != claim.provenance_snapshot:
            raise ValueError("security_audit_provenance_mutated")
        if not owner._accept_audit_provenance(
            provenance,
            action=claim.action,
            event_type=claim.event_type,
            decision=claim.decision,
            risk_level=claim.risk_level,
            policy_version=claim.policy_version,
            approval_id=claim.approval_id,
        ):
            raise ValueError("security_audit_provenance_untrusted")

        if claim.event_type in {"ACTION_EXECUTED", "ACTION_FAILED"}:
            evidence = claim.execution_evidence_snapshot
            if not isinstance(evidence, ExecutionReceipt) or provenance is not evidence:
                raise ValueError("execution_receipt_required")
            if not owner._validate_execution_receipt(
                evidence,
                action=claim.action,
                event_type=claim.event_type,
                decision=claim.decision,
                risk_level=claim.risk_level,
                policy_version=claim.policy_version,
                approval_id=claim.approval_id,
            ):
                raise ValueError("execution_receipt_untrusted")
            authorization = evidence.authorization
            action = claim.action
            if (
                authorization.task_id != action.task_id
                or authorization.action_id != action.action_id
                or authorization.tool_or_skill != action.tool_or_skill
                or authorization.requested_capabilities != tuple(sorted(action.requested_capabilities))
                or authorization.declared_capabilities != tuple(sorted(action.declared_capabilities or []))
                or authorization.input_fingerprint != action.input_fingerprint
                or authorization.goal != action.goal
                or authorization.action_type != action.action_type
                or authorization.skill_status != action.skill_status
                or authorization.policy_version != claim.policy_version
                or authorization.risk_level != claim.risk_level
                or authorization.approval_id != claim.approval_id
            ):
                raise ValueError("execution_receipt_mismatch")
            expected_decision = "ALLOW" if claim.event_type == "ACTION_EXECUTED" else "FAILED"
            if claim.decision != expected_decision:
                raise ValueError("execution_event_decision_mismatch")
            return

        if isinstance(provenance, PolicyDecision):
            expected_decisions = {
                "POLICY_EVALUATED": {provenance.decision},
                "PERMISSION_GRANTED": {"ALLOW"},
                "PERMISSION_DENIED": {"DENY"},
                "ACTION_BLOCKED": {"DENY", "QUARANTINE", "REQUIRE_APPROVAL"},
            }
            if claim.event_type not in expected_decisions or claim.decision not in expected_decisions[claim.event_type]:
                raise ValueError("policy_decision_event_mismatch")
            return

        if isinstance(provenance, AuthorizationEvidenceSnapshot):
            if claim.event_type != "ACTION_BLOCKED" or provenance.decision != "ALLOW" or claim.decision != "BLOCKED":
                raise ValueError("authorization_provenance_event_mismatch")
            return

        if isinstance(provenance, ApprovalRequest):
            expected_statuses = {
                "APPROVAL_REQUESTED": "PENDING",
                "APPROVAL_GRANTED": "APPROVED",
                "APPROVAL_DENIED": "DENIED",
                "APPROVAL_CANCELLED": "CANCELLED",
                "APPROVAL_EXPIRED": "EXPIRED",
                "APPROVAL_REVALIDATED": "APPROVED",
                "PERMISSION_GRANTED": "APPROVED",
            }
            if claim.event_type in expected_statuses and provenance.status != expected_statuses[claim.event_type]:
                raise ValueError("approval_event_status_mismatch")
            if claim.event_type == "APPROVAL_INVALIDATED" and claim.decision != provenance.status:
                raise ValueError("approval_invalidation_status_mismatch")
            return

        if isinstance(provenance, (SecurityPolicy, _ImmutableSecurityPolicy)):
            if (
                claim.event_type != "POLICY_CHANGED"
                or claim.decision != "ALLOW"
                or claim.action.action_type != "policy_update"
                or self._security_owner.policy_engine._policy is not provenance
                or provenance.policy_version != claim.policy_version
            ):
                raise ValueError("policy_change_provenance_mismatch")
            return

        if claim.event_type == "ACTION_BLOCKED" and owner._is_observed_block_provenance(provenance):
            if claim.metadata.get("provenance_kind") not in {None, "observed_only", "trusted_authorization"}:
                raise ValueError("observed_block_provenance_mismatch")
            return
        raise ValueError("security_audit_provenance_untrusted")

    def record(
        self,
        event_type: str,
        action: ActionRequest,
        *,
        decision: str,
        reason: str = "",
        risk_level: str = "LOW",
        policy_version: int = 1,
        approval_id: str = "",
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if event_type == "ACTION_BLOCKED":
            raise ValueError("action_blocked_requires_security_controller")
        if event_type in TRUSTED_SECURITY_AUDIT_EVENTS:
            raise ValueError("security_event_requires_trusted_writer")
        return self._record(
            event_type,
            action,
            decision=decision,
            reason=reason,
            risk_level=risk_level,
            policy_version=policy_version,
            approval_id=approval_id,
            actor=actor,
            metadata=metadata,
        )

    def _record(
        self,
        event_type: str,
        action: ActionRequest,
        *,
        decision: str,
        reason: str = "",
        risk_level: str = "LOW",
        policy_version: int = 1,
        approval_id: str = "",
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
        _claim: _SecurityAuditClaim | None = None,
        _requires_lease: bool = False,
    ) -> dict[str, Any]:
        if event_type in TRUSTED_SECURITY_AUDIT_EVENTS:
            if (
                _claim is None
                or self._writing_claim is not _claim
                or _claim.audit is not self
                or _claim.event_type != event_type
                or _claim.issuance_nonce is None
            ):
                raise ValueError(
                    "action_blocked_untrusted_writer"
                    if event_type == "ACTION_BLOCKED"
                    else "security_event_requires_trusted_writer"
                )
            self._validate_issued_claim(_claim)
            # The claim, not caller-supplied arguments, is the source of
            # security-sensitive provenance fields.
            action = _claim.action
            decision = _claim.decision
            reason = _claim.reason
            risk_level = _claim.risk_level
            policy_version = _claim.policy_version
            approval_id = _claim.approval_id
            actor = _claim.actor
            metadata = _claim.metadata
        if event_type not in AUDIT_EVENTS:
            raise ValueError(f"invalid_audit_event:{event_type}")
        event = sanitize_value(
            {
                "event_id": str(uuid.uuid4()),
                "timestamp": _now(),
                "event_type": event_type,
                "task_id": action.task_id,
                "action_id": action.action_id,
                "actor": actor,
                "tool_or_skill": action.tool_or_skill,
                "capabilities": action.requested_capabilities,
                "risk_level": risk_level,
                "policy_version": policy_version,
                "approval_id": approval_id,
                "decision": decision,
                "reason": reason,
                "metadata": metadata or {},
            }
        )
        if self.store is not None and hasattr(self.store, "save_audit_event"):
            self.store.save_audit_event(
                event,
                requires_lease=_requires_lease or event_type in {"ACTION_EXECUTED", "ACTION_FAILED", "ACTION_BLOCKED"},
            )
        self._events.append(event)
        observer = self.observability or getattr(self.store, "observability", None)
        if observer is not None:
            mapped_type = {
                "PERMISSION_GRANTED": "SECURITY_ALLOW",
                "PERMISSION_DENIED": "SECURITY_DENY",
                "POLICY_EVALUATED": {
                    "ALLOW": "SECURITY_ALLOW",
                    "DENY": "SECURITY_DENY",
                    "REQUIRE_APPROVAL": "SECURITY_REQUIRE_APPROVAL",
                    "QUARANTINE": "SECURITY_QUARANTINE",
                }.get(decision, "SECURITY_DENY"),
                "APPROVAL_GRANTED": "APPROVAL_APPROVED",
                "APPROVAL_CANCELLED": "APPROVAL_CANCELLED",
                "SKILL_QUARANTINED": "SKILL_BLOCKED",
                "ACTION_EXECUTED": "ACTION_COMPLETED" if decision == "ALLOW" else "ACTION_FAILED",
                "ACTION_BLOCKED": "ACTION_BLOCKED",
            }.get(event_type, event_type)
            try:
                observer.emit(
                    mapped_type,
                    task_id=event["task_id"],
                    action_id=event["action_id"],
                    component="security",
                    severity="ERROR" if mapped_type in {"SECURITY_DENY", "SECURITY_REQUIRE_APPROVAL", "SECURITY_QUARANTINE", "ACTION_BLOCKED", "ACTION_FAILED"} else "INFO",
                    metadata={
                        "security_event": event_type,
                        "decision": decision,
                        "reason": reason,
                        "risk_level": risk_level,
                        "policy_version": policy_version,
                        "approval_id": approval_id,
                        **(
                            {
                                "authorized_policy_version": event["metadata"].get("authorized_policy_version"),
                                "observed_current_policy_version": event["metadata"].get("observed_current_policy_version"),
                                "policy_version_source": event["metadata"].get("policy_version_source"),
                                "provenance_kind": event["metadata"].get("provenance_kind"),
                            }
                            if event_type in {"ACTION_BLOCKED", "ACTION_EXECUTED", "ACTION_FAILED"}
                            else {}
                        ),
                    },
                    event_id=f"security:{event['event_id']}:{mapped_type}",
                )
            except Exception:
                # The monitor itself decides degraded vs strict behavior; an
                # observer must never turn an authorization decision into an
                # authorization bypass.
                pass
        return event

    def _write_security_claim(self, claim: _SecurityAuditClaim) -> dict[str, Any]:
        with self._claim_lock:
            if (
                not isinstance(claim, _SecurityAuditClaim)
                or claim.audit is not self
                or claim.event_type not in TRUSTED_SECURITY_AUDIT_EVENTS
                or claim.issuance_nonce is None
                or self._claim_lookup(claim.claim_id) is not claim
            ):
                raise ValueError("security_audit_claim_untrusted")
            self._validate_issued_claim(claim)
            try:
                self._writing_claim = claim
                return self._record(
                    claim.event_type,
                    claim.action,
                    decision=claim.decision,
                    reason=claim.reason,
                    risk_level=claim.risk_level,
                    policy_version=claim.policy_version,
                    approval_id=claim.approval_id,
                    actor=claim.actor,
                    metadata=claim.metadata,
                    _claim=claim,
                    _requires_lease=claim.worker_bound,
                )
            finally:
                # Security audit claims are one-shot even if persistence fails.
                self._writing_claim = None
                self._active_claims.pop(claim.claim_id, None)
                self._claim_consume(claim.claim_id)
                release = getattr(self._security_owner, "_release_observed_block_provenance", None)
                if callable(release):
                    release(claim.provenance)

    def _record_blocked(self, claim: _SecurityAuditClaim | None = None, **kwargs: Any) -> dict[str, Any]:
        if kwargs or claim is None:
            raise ValueError("security_audit_claim_required")
        if not isinstance(claim, _SecurityAuditClaim) or claim.event_type != "ACTION_BLOCKED":
            raise ValueError("security_audit_claim_untrusted")
        return self._write_security_claim(claim)

    def list(self, *, task_id: str | None = None) -> list[dict[str, Any]]:
        if self.store is not None and hasattr(self.store, "all_audit_events"):
            return self.store.all_audit_events(task_id=task_id)
        return [event for event in self._events if task_id is None or event["task_id"] == task_id]


class SecurityController:
    """Policy and approval control plane used both before planning execution and at the boundary."""

    store = _WriteOnceReference()
    audit = _WriteOnceReference()
    policy_engine = _WriteOnceReference()
    approval_gate = _WriteOnceReference()
    _operation_binding_token = _WriteOnceReference()
    _receipt_lookup = _WriteOnceReference()
    _receipt_factory = _WriteOnceReference()
    _security_state_sealed = _MonotonicSealFlag()

    __slots__ = (
        "_authorization_evidence",
        "_authorization_evidence_lock",
        "_execution_inflight",
        "_observed_block_provenance",
        "_issued_policy_decisions",
        "__weakref__",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_security_state_sealed", False) and name in {
            "audit",
            "policy_engine",
            "approval_gate",
            "_receipt_lookup",
            "_receipt_factory",
            "_operation_binding_token",
            "_security_state_sealed",
        }:
            raise AttributeError("security_state_immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        store: Any | None = None,
        policy_engine: PolicyEngine | None = None,
        approval_gate: ApprovalGate | None = None,
        audit_trail: AuditTrail | None = None,
        observability: Any | None = None,
    ) -> None:
        self.store = store
        # Execution evidence is intentionally kept private and is issued only
        # by authorize(..., boundary=True).  The object identity check prevents
        # callers from manufacturing an equivalent PolicyDecision dataclass.
        self._authorization_evidence: dict[
            int,
            tuple[PolicyDecision, AuthorizationEvidenceSnapshot, tuple[Any, ...], tuple[Any, ...]],
        ] = {}
        # Evidence consumption is a one-shot operation.  The lock covers the
        # final validation and removal so concurrent callers cannot consume the
        # same evidence twice.  It does not provide exactly-once guarantees for
        # external side effects performed before this method is reached.
        self._authorization_evidence_lock = threading.RLock()
        self._execution_inflight: set[int] = set()
        self._security_state_sealed = False
        self._observed_block_provenance: dict[int, object] = {}
        self._issued_policy_decisions: dict[int, tuple[PolicyDecision, Any]] = {}
        self._operation_binding_token = object()
        receipt_records: dict[str, ExecutionReceipt] = {}
        receipt_records_lock = threading.RLock()

        def lookup_receipt(receipt_id: str) -> ExecutionReceipt | None:
            with receipt_records_lock:
                return receipt_records.get(receipt_id)

        original_new_receipt = self._new_execution_receipt

        def issue_receipt(*args: Any, **kwargs: Any) -> ExecutionReceipt:
            receipt = original_new_receipt(*args, **kwargs)
            with receipt_records_lock:
                receipt_records[receipt.receipt_id] = receipt
            return receipt

        self._receipt_lookup = lookup_receipt
        self._receipt_factory = issue_receipt
        if audit_trail is None:
            self.audit = AuditTrail(
                store=store,
                observability=observability,
                _security_owner=self,
            )
        else:
            self.audit = audit_trail
            if getattr(self.audit, "_security_owner", None) is None:
                binder = getattr(self.audit, "_bind_security_owner", None)
                if not callable(binder):
                    raise TypeError("unsupported_audit_trail:owner_binding_required")
                binder(self)
        if not callable(getattr(self.audit, "_write_security_claim", None)) or not callable(
            getattr(self.audit, "_issue_security_claim", None)
        ) or not hasattr(self.audit, "_claim_issuer") or getattr(self.audit, "_security_owner", None) is not self:
            raise TypeError("unsupported_audit_trail:trusted_security_writer_required")
        if observability is not None:
            self.audit.observability = observability
        self.policy_engine = policy_engine or PolicyEngine(store=store)
        self.approval_gate = approval_gate or ApprovalGate(
            store=store,
            event_logger=self._approval_event,
            policy_version_provider=self.policy_engine._current_policy_version,
        )
        if hasattr(self.approval_gate, "set_policy_version_provider"):
            self.approval_gate.set_policy_version_provider(self.policy_engine._current_policy_version)
        bind_gate = getattr(store, "_bind_approval_gate", None)
        if callable(bind_gate):
            bind_gate(self.approval_gate, owner=self)
        self._security_state_sealed = True

    def _accept_audit_provenance(
        self,
        provenance: Any,
        *,
        action: ActionRequest,
        event_type: str,
        decision: str,
        risk_level: str,
        policy_version: int,
        approval_id: str,
    ) -> bool:
        """Validate that claim provenance came from this controller's live state."""
        if isinstance(provenance, ExecutionReceipt):
            return self._validate_execution_receipt(
                provenance,
                action=action,
                event_type=event_type,
                decision=decision,
                risk_level=risk_level,
                policy_version=policy_version,
                approval_id=approval_id,
            )
        if isinstance(provenance, AuthorizationEvidenceSnapshot):
            active = any(
                entry[1] is provenance
                for entry in self._authorization_evidence.values()
            )
            return active

        if isinstance(provenance, PolicyDecision):
            issued = self._issued_policy_decisions.get(id(provenance))
            if issued is None or issued[0] is not provenance or issued[1] != _provenance_snapshot(provenance):
                return False
            evaluated = self.policy_engine.evaluate(action)
            binding = self._authorization_binding(action)
            evaluated_binding = self._authorization_binding(action)
            provenance_binding = (
                provenance.task_id,
                provenance.action_id,
                provenance.tool_or_skill,
                tuple(sorted(provenance.requested_capabilities)),
                tuple(sorted(provenance.declared_capabilities)),
                provenance.input_fingerprint,
                provenance.goal,
                provenance.action_type,
                provenance.skill_status,
            )
            if provenance_binding != binding or evaluated_binding != binding:
                return False
            same_policy_decision = (
                provenance.decision == evaluated.decision
                and provenance.risk_level == evaluated.risk_level
                and provenance.policy_version == evaluated.policy_version
            )
            approved_policy_decision = (
                provenance.decision == "ALLOW"
                and evaluated.decision == "REQUIRE_APPROVAL"
                and bool(provenance.approval_id)
                and provenance.approval_id == approval_id
                and self.approval_gate.valid_for(
                    provenance.approval_id,
                    action,
                    evaluated.policy_version,
                    evaluated.risk_level,
                )
            )
            return same_policy_decision or approved_policy_decision

        if isinstance(provenance, ApprovalRequest):
            if not self.approval_gate._is_current(provenance):
                return False
            if provenance.legacy:
                return event_type in {
                    "APPROVAL_REQUESTED",
                    "APPROVAL_GRANTED",
                    "APPROVAL_DENIED",
                    "APPROVAL_CANCELLED",
                    "APPROVAL_EXPIRED",
                }
            if event_type == "APPROVAL_INVALIDATED":
                return provenance.approval_id == approval_id and provenance.policy_version <= policy_version
            return self.approval_gate._bound_request(
                provenance,
                action,
                policy_version,
                risk_level,
            )

        if isinstance(provenance, (SecurityPolicy, _ImmutableSecurityPolicy)):
            return (
                event_type == "POLICY_CHANGED"
                and action.task_id == "policy"
                and action.tool_or_skill == "SecurityPolicy"
                and self.policy_engine._policy is provenance
                and provenance.policy_version == policy_version
            )

        return self._is_observed_block_provenance(provenance)

    def _validate_execution_receipt(
        self,
        receipt: ExecutionReceipt,
        *,
        action: ActionRequest,
        event_type: str,
        decision: str,
        risk_level: str,
        policy_version: int,
        approval_id: str,
    ) -> bool:
        if not isinstance(receipt, ExecutionReceipt) or not callable(receipt._state):
            return False
        lookup = getattr(self, "_receipt_lookup", None)
        if not callable(lookup) or lookup(receipt.receipt_id) is not receipt:
            return False
        if not self._validate_operation_binding(receipt.operation_binding, action):
            return False
        if not receipt._state(receipt, "spent"):
            return False
        authorization = receipt.authorization
        if authorization.decision != "ALLOW":
            return False
        if (
            authorization.task_id != action.task_id
            or authorization.action_id != action.action_id
            or authorization.tool_or_skill != action.tool_or_skill
            or authorization.requested_capabilities != tuple(sorted(action.requested_capabilities))
            or authorization.declared_capabilities != tuple(sorted(action.declared_capabilities or []))
            or authorization.input_fingerprint != action.input_fingerprint
            or authorization.goal != action.goal
            or authorization.action_type != action.action_type
            or authorization.skill_status != action.skill_status
            or authorization.policy_version != policy_version
            or authorization.risk_level != risk_level
            or authorization.approval_id != approval_id
        ):
            return False
        if event_type == "ACTION_EXECUTED":
            return decision == "ALLOW" and receipt.success
        if event_type == "ACTION_FAILED":
            return decision == "FAILED" and not receipt.success
        return False

    def _is_observed_block_provenance(self, provenance: Any) -> bool:
        return self._observed_block_provenance.get(id(provenance)) is provenance

    def _release_observed_block_provenance(self, provenance: Any) -> None:
        self._observed_block_provenance.pop(id(provenance), None)

    def _refresh_issued_policy_decision(self, decision: PolicyDecision) -> bool:
        """Refresh a decision after controlled approval binding mutations."""
        with self._authorization_evidence_lock:
            issued = self._issued_policy_decisions.get(id(decision))
            if issued is None or issued[0] is not decision:
                return False
            self._issued_policy_decisions[id(decision)] = (decision, _provenance_snapshot(decision))
            return True

    def set_observability(self, observability: Any | None) -> None:
        self.audit.observability = observability

    def preflight(self, action: ActionRequest, *, approval_id: str = "") -> PolicyDecision:
        decision = self.policy_engine.evaluate(action)
        self._persist_decision(decision)
        self.audit._write_security_claim(
            self.audit._issue_security_claim(
                self.audit._claim_issuer,
                "POLICY_EVALUATED",
                action,
                decision=decision.decision,
                reason=action.reason or "; ".join(decision.reasons),
                risk_level=decision.risk_level,
                policy_version=decision.policy_version,
                provenance=decision,
            )
        )
        if decision.decision in {"DENY", "QUARANTINE"}:
            if decision.decision == "DENY":
                self.audit._write_security_claim(
                    self.audit._issue_security_claim(
                        self.audit._claim_issuer,
                        "PERMISSION_DENIED",
                        action,
                        decision=decision.decision,
                        reason="; ".join(decision.reasons),
                        risk_level=decision.risk_level,
                        policy_version=decision.policy_version,
                        provenance=decision,
                    )
                )
            else:
                self.audit.record(
                    "SKILL_QUARANTINED",
                    action,
                    decision=decision.decision,
                    reason="; ".join(decision.reasons),
                    risk_level=decision.risk_level,
                    policy_version=decision.policy_version,
                )
            return decision
        if decision.decision == "REQUIRE_APPROVAL":
            if approval_id:
                request = self.approval_gate._get(approval_id)
                if request and self.approval_gate._bound_request(
                    request,
                    action,
                    decision.policy_version,
                    decision.risk_level,
                ):
                    if request.status == "APPROVED":
                        decision.decision = "ALLOW"
                        decision.approval_id = approval_id
                        decision.reasons.append("exact_approval_valid")
                        self.audit._write_security_claim(
                            self.audit._issue_security_claim(
                                self.audit._claim_issuer,
                                "APPROVAL_REVALIDATED",
                                action,
                                decision="ALLOW",
                                reason="Persisted approval matched the current action binding",
                                risk_level=decision.risk_level,
                                policy_version=decision.policy_version,
                                approval_id=approval_id,
                                actor=request.actor,
                                provenance=request,
                            )
                        )
                        self.audit._write_security_claim(
                            self.audit._issue_security_claim(
                                self.audit._claim_issuer,
                                "PERMISSION_GRANTED",
                                action,
                                decision="ALLOW",
                                reason="Exact human approval matched the action binding",
                                risk_level=decision.risk_level,
                                policy_version=decision.policy_version,
                                approval_id=approval_id,
                                actor=request.actor,
                                provenance=request,
                            )
                        )
                        return decision
                    if request.status == "PENDING":
                        decision.approval_id = approval_id
                        decision.reasons.append("persisted_approval_still_pending")
                        return decision
                if request is not None:
                    self.audit._write_security_claim(
                        self.audit._issue_security_claim(
                            self.audit._claim_issuer,
                            "APPROVAL_INVALIDATED",
                            action,
                            decision=request.status,
                            reason="Persisted approval no longer matches current policy or action binding",
                            risk_level=decision.risk_level,
                            policy_version=decision.policy_version,
                            approval_id=approval_id,
                            provenance=request,
                        )
                    )
            request = self.approval_gate.request(action, decision, ttl_seconds=self.policy_engine.policy.approval_ttl_seconds)
            decision.approval_id = request.approval_id
            decision.reasons.append("waiting_for_explicit_human_approval")
        else:
            self.audit._write_security_claim(
                self.audit._issue_security_claim(
                    self.audit._claim_issuer,
                    "PERMISSION_GRANTED",
                    action,
                    decision="ALLOW",
                    reason=action.reason or "Deterministic policy allowed the action",
                    risk_level=decision.risk_level,
                    policy_version=decision.policy_version,
                    provenance=decision,
                )
            )
        return decision

    def authorize(self, action: ActionRequest, *, approval_id: str = "", boundary: bool = False) -> PolicyDecision:
        decision = self.policy_engine.evaluate(action)
        self._persist_decision(decision)
        self.audit._write_security_claim(
            self.audit._issue_security_claim(
                self.audit._claim_issuer,
                "POLICY_EVALUATED",
                action,
                decision=decision.decision,
                reason=action.reason or "; ".join(decision.reasons),
                risk_level=decision.risk_level,
                policy_version=decision.policy_version,
                approval_id=approval_id,
                metadata={"boundary": boundary},
                provenance=decision,
            )
        )
        if decision.decision in {"DENY", "QUARANTINE"}:
            self._record_blocked_event(
                self.audit._issue_security_claim(
                    self.audit._claim_issuer,
                    "ACTION_BLOCKED",
                    action,
                    decision=decision.decision,
                    reason="; ".join(decision.reasons),
                    risk_level=decision.risk_level,
                    policy_version=decision.policy_version,
                    approval_id=approval_id,
                    metadata=self._blocked_provenance_metadata(
                        {"boundary": boundary},
                        authorized_policy_version=None,
                        observed_current_policy_version=decision.policy_version,
                        policy_version_source="observed_current",
                        provenance_kind="validated_policy_decision",
                    ),
                    provenance=decision,
                ),
            )
            return decision
        if decision.decision == "REQUIRE_APPROVAL":
            if not approval_id or not self.approval_gate.valid_for(
                approval_id,
                action,
                decision.policy_version,
                decision.risk_level,
            ):
                if not approval_id:
                    request = self.approval_gate.request(action, decision, ttl_seconds=self.policy_engine.policy.approval_ttl_seconds)
                    decision.approval_id = request.approval_id
                else:
                    request = self.approval_gate._get(approval_id)
                    if request is not None:
                        self.audit._write_security_claim(
                            self.audit._issue_security_claim(
                                self.audit._claim_issuer,
                                "APPROVAL_INVALIDATED",
                                action,
                                decision=request.status,
                                reason="Execution-boundary approval binding or policy validation failed",
                                risk_level=decision.risk_level,
                                policy_version=decision.policy_version,
                                approval_id=approval_id,
                                provenance=request,
                            )
                        )
                decision.reasons.append("approval_invalid_or_not_bound")
                decision.reasons.append("execution_requires_valid_approval")
                decision.decision = "REQUIRE_APPROVAL"
                self._record_blocked_event(
                    self.audit._issue_security_claim(
                        self.audit._claim_issuer,
                        "ACTION_BLOCKED",
                        action,
                        decision=decision.decision,
                        reason="; ".join(decision.reasons),
                        risk_level=decision.risk_level,
                        policy_version=decision.policy_version,
                        approval_id=decision.approval_id or approval_id,
                        metadata=self._blocked_provenance_metadata(
                            {"boundary": boundary},
                            authorized_policy_version=None,
                            observed_current_policy_version=decision.policy_version,
                            policy_version_source="observed_current",
                            provenance_kind="validated_policy_decision",
                        ),
                        provenance=decision,
                    ),
                )
                return decision
            decision.decision = "ALLOW"
            decision.approval_id = approval_id
            decision.reasons.append("exact_approval_valid")
        self.audit._write_security_claim(
            self.audit._issue_security_claim(
                self.audit._claim_issuer,
                "PERMISSION_GRANTED",
                action,
                decision="ALLOW",
                reason="Execution-boundary policy check passed",
                risk_level=decision.risk_level,
                policy_version=decision.policy_version,
                approval_id=approval_id,
                metadata={"boundary": boundary},
                provenance=decision,
            )
        )
        if boundary:
            with self._authorization_evidence_lock:
                snapshot = AuthorizationEvidenceSnapshot.from_decision(decision)
                self._authorization_evidence[id(decision)] = (
                    decision,
                    snapshot,
                    self._authorization_binding(action),
                    self._authorization_signature(snapshot),
                )
        return decision

    @staticmethod
    def _blocked_provenance_metadata(
        metadata: dict[str, Any] | None,
        *,
        authorized_policy_version: int | None,
        observed_current_policy_version: int,
        policy_version_source: str,
        provenance_kind: str = "observed_only",
    ) -> dict[str, Any]:
        """Return blocked-event metadata with controller-owned provenance facts."""
        audit_metadata = dict(metadata or {})
        # These fields are trusted audit facts, not caller annotations.
        audit_metadata["authorized_policy_version"] = authorized_policy_version
        audit_metadata["observed_current_policy_version"] = observed_current_policy_version
        audit_metadata["policy_version_source"] = policy_version_source
        audit_metadata["provenance_kind"] = provenance_kind
        return audit_metadata

    def _record_blocked_event(
        self,
        claim: _SecurityAuditClaim | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if kwargs or claim is None:
            raise ValueError("security_audit_claim_required")
        if not isinstance(claim, _SecurityAuditClaim) or claim.event_type != "ACTION_BLOCKED":
            raise ValueError("security_audit_claim_untrusted")
        return self.audit._write_security_claim(claim)

    @staticmethod
    def _authorization_binding(action: ActionRequest) -> tuple[Any, ...]:
        return (
            action.task_id,
            action.action_id,
            action.tool_or_skill,
            tuple(sorted(action.requested_capabilities)),
            tuple(sorted(action.declared_capabilities or [])),
            action.input_fingerprint,
            action.goal,
            action.action_type,
            action.skill_status,
        )

    @staticmethod
    def _authorization_signature(decision: PolicyDecision | AuthorizationEvidenceSnapshot) -> tuple[Any, ...]:
        return (
            decision.decision,
            decision.risk_level,
            decision.task_id,
            decision.action_id,
            decision.tool_or_skill,
            tuple(sorted(decision.requested_capabilities)),
            decision.policy_version,
            decision.approval_id,
            decision.input_fingerprint,
            tuple(sorted(decision.declared_capabilities)),
            decision.goal,
            decision.action_type,
            decision.skill_status,
        )

    def _pending_boundary(self, action: ActionRequest, authorization: PolicyDecision) -> AuthorizationEvidenceSnapshot:
        """Return live boundary evidence or fail before an operation starts."""
        with self._authorization_evidence_lock:
            evidence = self._authorization_evidence.get(id(authorization))
            if evidence is None or evidence[0] is not authorization:
                raise ValueError("execution_authorization_untrusted")
            _, snapshot, issued_binding, issued_signature = evidence
            if issued_signature != self._authorization_signature(authorization):
                raise ValueError("execution_authorization_mutated")
            if snapshot.decision != "ALLOW" or authorization.decision != "ALLOW":
                raise ValueError("execution_authorization_required")
            if issued_binding != self._authorization_binding(action):
                raise ValueError("execution_authorization_mismatch")
            current = self.policy_engine.evaluate(action)
            if current.policy_version != snapshot.policy_version:
                raise ValueError("execution_authorization_stale")
            if current.decision in {"DENY", "QUARANTINE"}:
                raise ValueError("execution_authorization_currently_blocked")
            if current.decision == "REQUIRE_APPROVAL" and not (
                snapshot.approval_id
                and self.approval_gate.valid_for(
                    snapshot.approval_id,
                    action,
                    current.policy_version,
                    current.risk_level,
                )
            ):
                raise ValueError("execution_authorization_approval_invalid")
            if current.risk_level != snapshot.risk_level:
                raise ValueError("execution_authorization_mismatch")
            return snapshot

    def _begin_execution(self, action: ActionRequest, authorization: PolicyDecision) -> None:
        with self._authorization_evidence_lock:
            authorization_id = id(authorization)
            if authorization_id in self._execution_inflight:
                raise ValueError("execution_authorization_in_flight")
            self._pending_boundary(action, authorization)
            self._execution_inflight.add(authorization_id)

    def _new_execution_receipt(
        self,
        action: ActionRequest,
        authorization: PolicyDecision,
        *,
        success: bool,
        metadata: dict[str, Any] | None = None,
        operation_binding: TrustedOperationBinding | None = None,
    ) -> ExecutionReceipt:
        with self._authorization_evidence_lock:
            snapshot = self._pending_boundary(action, authorization)
            if id(authorization) not in self._execution_inflight:
                raise ValueError("execution_authorization_not_in_flight")
            holder: dict[str, ExecutionReceipt] = {}
            state = {"value": "active"}

            def receipt_state(candidate: ExecutionReceipt, expected: str, *, consume: bool = False) -> bool:
                if candidate is not holder.get("receipt") or state["value"] != expected:
                    return False
                if consume:
                    state["value"] = "spent"
                return True

            receipt = ExecutionReceipt(
                receipt_id=str(uuid.uuid4()),
                authorization_id=id(authorization),
                action_snapshot=_action_snapshot(action),
                authorization=snapshot,
                success=success,
                metadata_snapshot=_freeze_value(metadata or {}),
                operation_binding=operation_binding,
                _state=receipt_state,
            )
            holder["receipt"] = receipt
            return receipt

    def _bind_operation(
        self,
        action: ActionRequest,
        operation: Callable[[], Any],
        *,
        kind: str,
    ) -> TrustedOperationBinding:
        return TrustedOperationBinding(
            controller=self,
            action_binding=self._authorization_binding(action),
            operation=operation,
            kind=kind,
            token=self._operation_binding_token,
        )

    def _validate_operation_binding(
        self,
        binding: TrustedOperationBinding | None,
        action: ActionRequest,
        operation: Callable[[], Any] | None = None,
    ) -> bool:
        if not isinstance(binding, TrustedOperationBinding):
            return False
        if binding.controller is not self or binding.token is not self._operation_binding_token:
            return False
        if binding.action_binding != self._authorization_binding(action):
            return False
        return operation is None or binding.operation is operation

    def execute_authorized(
        self,
        action: ActionRequest,
        operation: Callable[[], Any],
        *,
        approval_id: str = "",
        authorization: PolicyDecision | None = None,
        result_validator: Callable[[Any], tuple[bool, dict[str, Any]]] | None = None,
        operation_binding: TrustedOperationBinding | None = None,
        pre_execution_check: Callable[[], None] | None = None,
    ) -> AuthorizedExecutionResult:
        """Run one operation inside the trusted authorization boundary.

        The receipt is created only after ``operation`` returns or raises.  The
        public compatibility method ``record_execution`` accepts that receipt,
        not a caller-supplied action/ALLOW pair.
        """
        decision = authorization or self.authorize(action, approval_id=approval_id, boundary=True)
        if not decision.allowed:
            return AuthorizedExecutionResult(decision=decision)
        if not self._validate_operation_binding(operation_binding, action, operation):
            raise ValueError("trusted_operation_binding_required")
        self._begin_execution(action, decision)
        if pre_execution_check is not None:
            try:
                pre_execution_check()
            except Exception as exc:
                # No external operation started.  Drop the in-flight evidence
                # without producing ACTION_FAILED/ACTION_EXECUTED so a stale
                # worker cannot later replay this authorization.
                with self._authorization_evidence_lock:
                    self._authorization_evidence.pop(id(decision), None)
                    self._execution_inflight.discard(id(decision))
                return AuthorizedExecutionResult(decision=decision, exception=exc, blocked=True)
        try:
            result = operation()
        except Exception as exc:
            receipt = self._receipt_factory(
                action,
                decision,
                success=False,
                metadata={"exception_type": type(exc).__name__},
                operation_binding=operation_binding,
            )
            self.record_execution(receipt)
            return AuthorizedExecutionResult(decision=decision, exception=exc, receipt=receipt)

        success = True
        metadata: dict[str, Any] = {}
        if result_validator is not None:
            try:
                validation = result_validator(result)
                if not isinstance(validation, tuple) or len(validation) != 2:
                    raise ValueError("invalid_execution_result_validator")
                success = bool(validation[0])
                metadata = dict(validation[1] or {})
            except Exception as exc:
                success = False
                metadata = {"result_validation_exception": type(exc).__name__}
        receipt = self._receipt_factory(
            action,
            decision,
            success=success,
            metadata=metadata,
            operation_binding=operation_binding,
        )
        self.record_execution(receipt)
        return AuthorizedExecutionResult(decision=decision, result=result, receipt=receipt)

    def record_execution(self, receipt: ExecutionReceipt, **kwargs: Any) -> dict[str, Any]:
        """Persist an execution result only when given a trusted one-shot receipt."""
        if kwargs or not isinstance(receipt, ExecutionReceipt):
            raise ValueError("execution_receipt_required")
        if not callable(receipt._state):
            raise ValueError("execution_receipt_untrusted")
        lookup = getattr(self, "_receipt_lookup", None)
        if not callable(lookup) or lookup(receipt.receipt_id) is not receipt:
            raise ValueError("execution_receipt_untrusted")
        action = receipt.action
        with self._authorization_evidence_lock:
            if not receipt._state(receipt, "active", consume=True):
                raise ValueError("execution_receipt_untrusted")
            evidence = self._authorization_evidence.get(receipt.authorization_id)
            if evidence is None or evidence[1] is not receipt.authorization:
                raise ValueError("execution_receipt_untrusted")
            snapshot = evidence[1]
            if self._authorization_binding(action) != evidence[2]:
                raise ValueError("execution_receipt_mismatch")
            if self._authorization_signature(snapshot) != evidence[3]:
                raise ValueError("execution_receipt_mutated")
            current = self.policy_engine.evaluate(action)
            if (
                snapshot.policy_version != current.policy_version
                or snapshot.risk_level != current.risk_level
                or current.decision in {"DENY", "QUARANTINE"}
            ) and receipt.success:
                self._authorization_evidence.pop(receipt.authorization_id, None)
                self._execution_inflight.discard(receipt.authorization_id)
                raise ValueError("execution_authorization_stale")
            if receipt.success and current.decision == "REQUIRE_APPROVAL" and not (
                snapshot.approval_id
                and self.approval_gate.valid_for(
                    snapshot.approval_id,
                    action,
                    current.policy_version,
                    current.risk_level,
                )
            ):
                self._authorization_evidence.pop(receipt.authorization_id, None)
                self._execution_inflight.discard(receipt.authorization_id)
                raise ValueError("execution_authorization_approval_invalid")
            self._authorization_evidence.pop(receipt.authorization_id, None)
            self._execution_inflight.discard(receipt.authorization_id)

        authorized_version = snapshot.policy_version
        audit_metadata = receipt.metadata
        audit_metadata["authorized_policy_version"] = authorized_version
        audit_metadata["observed_current_policy_version"] = current.policy_version
        audit_metadata["policy_version_source"] = "trusted_execution_receipt"
        event_type = "ACTION_EXECUTED" if receipt.success else "ACTION_FAILED"
        return self.audit._write_security_claim(
            self.audit._issue_security_claim(
                self.audit._claim_issuer,
                event_type,
                action,
                decision="ALLOW" if receipt.success else "FAILED",
                reason="Operation returned successfully" if receipt.success else "Operation returned a failure after authorization",
                risk_level=current.risk_level,
                policy_version=authorized_version,
                approval_id=snapshot.approval_id,
                metadata=audit_metadata,
                execution_evidence=receipt,
                provenance=receipt,
            )
        )

    def record_blocked(
        self,
        action: ActionRequest,
        *,
        reason: str,
        authorization: PolicyDecision | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        current = self.policy_engine.evaluate(action)
        trusted_authorized_version: int | None = None
        trusted_allow_evidence = False
        if authorization is not None:
            with self._authorization_evidence_lock:
                evidence = self._authorization_evidence.get(id(authorization))
                if evidence is not None and evidence[0] is authorization:
                    _, snapshot, issued_binding, issued_signature = evidence
                    if issued_signature != self._authorization_signature(authorization):
                        raise ValueError("blocked_audit_authorization_mutated")
                    if snapshot.decision != "ALLOW":
                        raise ValueError("blocked_audit_authorization_invalid")
                    if issued_binding != self._authorization_binding(action):
                        raise ValueError("blocked_audit_authorization_mismatch")
                    trusted_authorized_version = snapshot.policy_version
                    trusted_allow_evidence = True
                elif authorization.decision == "ALLOW":
                    # An ALLOW object without boundary evidence is forged or
                    # stale. Never copy its mutable policy_version into audit.
                    raise ValueError("blocked_audit_authorization_untrusted")
                elif authorization.decision not in {"DENY", "QUARANTINE", "REQUIRE_APPROVAL"}:
                    raise ValueError("blocked_audit_authorization_invalid")
        # The legacy audit_events schema requires an integer policy_version.
        # For observed-only blocks, retain the field as the current observed
        # version and make the absence of trusted authorization explicit in
        # metadata; it is never labeled as an authorized version.
        event_policy_version = trusted_authorized_version if trusted_allow_evidence else current.policy_version
        audit_metadata = self._blocked_provenance_metadata(
            metadata,
            authorized_policy_version=trusted_authorized_version,
            observed_current_policy_version=current.policy_version,
            policy_version_source="trusted_authorization" if trusted_allow_evidence else "observed_current",
            provenance_kind="trusted_authorization" if trusted_allow_evidence else "observed_only",
        )
        observed_provenance = None
        if trusted_allow_evidence:
            claim_provenance = snapshot
        else:
            observed_provenance = object()
            self._observed_block_provenance[id(observed_provenance)] = observed_provenance
            claim_provenance = observed_provenance
        issued_claim = None
        try:
            issued_claim = self.audit._issue_security_claim(
                self.audit._claim_issuer,
                "ACTION_BLOCKED",
                action,
                decision="BLOCKED",
                reason=reason,
                risk_level=current.risk_level,
                policy_version=event_policy_version,
                metadata=audit_metadata,
                provenance=claim_provenance,
            )
            self._record_blocked_event(
                issued_claim
            )
        finally:
            if observed_provenance is not None and issued_claim is None:
                self._observed_block_provenance.pop(id(observed_provenance), None)

    def preflight_plan(
        self,
        task_id: str,
        plan: Any,
        tools: Any,
        *,
        approval_ids: dict[str, str] | None = None,
    ) -> list[PolicyDecision]:
        decisions: list[PolicyDecision] = []
        approval_ids = approval_ids or {}
        for step in plan.steps:
            tool = tools.get(step.tool)
            action = self.action_for_tool(task_id, step.id, tool, step.input, goal=str(plan.goal))
            approval_id = approval_ids.get(step.id) or approval_ids.get(step.tool, "")
            decisions.append(self.preflight(action, approval_id=approval_id))
        return decisions

    def action_for_tool(
        self,
        task_id: str,
        action_id: str,
        tool: Any,
        inputs: dict[str, Any],
        *,
        goal: str = "",
    ) -> ActionRequest:
        capabilities = list(getattr(tool, "required_capabilities", []) or [])
        permission = str(getattr(tool, "permission_level", ""))
        if permission == "private_write" and "WRITE_WORKSPACE" not in capabilities:
            capabilities.append("WRITE_WORKSPACE")
        declared_value = getattr(tool, "declared_capabilities", None)
        declared = list(capabilities) if declared_value is None else list(declared_value)
        return ActionRequest(
            task_id=task_id,
            action_id=action_id,
            tool_or_skill=str(getattr(tool, "name", tool.__class__.__name__)),
            requested_capabilities=capabilities,
            declared_capabilities=declared,
            risk_hint=str(getattr(tool, "risk_level", "")),
            required_resources={},
            network_required=bool(getattr(tool, "network_required", False)),
            filesystem_required=bool(getattr(tool, "filesystem_required", False)),
            reason=f"Execute registered tool {getattr(tool, 'name', tool.__class__.__name__)}",
            planned_effect=str(getattr(tool, "planned_effect", "Registered tool execution")),
            input_fingerprint=input_fingerprint(inputs),
            action_type="tool_execution",
            goal=goal,
        )

    def action_for_skill(
        self,
        task_id: str,
        action_id: str,
        candidate: Any,
        inputs: dict[str, Any],
        *,
        goal: str = "",
    ) -> ActionRequest:
        metadata = candidate.metadata if isinstance(getattr(candidate, "metadata", {}), dict) else {}
        raw = list(getattr(candidate, "declared_capabilities", []) or [])
        if not raw:
            raw = list(metadata.get("required_capabilities", []) or [])
        mapped = {
            "network": "NETWORK_ACCESS",
            "filesystem": "WRITE_WORKSPACE",
            "environment": "SENSITIVE_DATA_ACCESS",
        }
        capabilities = [mapped.get(item, item) for item in raw]
        return ActionRequest(
            task_id=task_id,
            action_id=action_id,
            tool_or_skill=str(candidate.name),
            requested_capabilities=capabilities,
            declared_capabilities=capabilities,
            risk_hint=str(metadata.get("risk_level", "LOW")),
            required_resources=dict(getattr(candidate, "required_resources", {}) or {}),
            network_required="NETWORK_ACCESS" in capabilities or "EXTERNAL_API" in capabilities,
            filesystem_required=any(item in capabilities for item in {"READ_WORKSPACE", "WRITE_WORKSPACE"}),
            sensitive_data="SENSITIVE_DATA_ACCESS" in capabilities,
            input_fingerprint=input_fingerprint(inputs),
            reason=f"Invoke approved skill {candidate.name}",
            planned_effect=str(metadata.get("planned_effect", "Skill invocation")),
            action_type="skill_invocation",
            skill_status=str(getattr(candidate, "status", "")),
            goal=goal,
        )

    def authorize_skill_activation(
        self,
        task_id: str,
        candidate: Any,
        *,
        inputs: dict[str, Any] | None = None,
        goal: str = "",
        approval_id: str = "",
    ) -> PolicyDecision:
        action = self.action_for_skill(
            task_id,
            f"activate:{candidate.name}:{candidate.version}",
            candidate,
            inputs or {},
            goal=goal,
        )
        action.action_type = "skill_activation"
        return self.authorize(action, approval_id=approval_id, boundary=True)

    def update_policy(self, policy: SecurityPolicy, *, actor: str = "system") -> SecurityPolicy:
        previous = self.policy_engine.policy
        updated = self.policy_engine.update_policy(policy)
        action = ActionRequest(
            task_id="policy",
            action_id=f"policy:{updated.policy_version}",
            tool_or_skill="SecurityPolicy",
            requested_capabilities=[],
            declared_capabilities=[],
            reason="Security policy version changed",
            planned_effect="Replace deterministic policy configuration",
            action_type="policy_update",
            actor=actor,
        )
        self.audit._write_security_claim(
            self.audit._issue_security_claim(
                self.audit._claim_issuer,
                "POLICY_CHANGED",
                action,
                decision="ALLOW",
                reason=f"policy_version:{previous.policy_version}->{updated.policy_version}",
                risk_level="LOW",
                policy_version=updated.policy_version,
                actor=actor,
                provenance=self.policy_engine._policy,
            )
        )
        return updated

    def _approval_event(self, event_type: str, request: ApprovalRequest) -> None:
        action = ActionRequest(
            task_id=request.task_id,
            action_id=request.action_id,
            tool_or_skill=request.tool_or_skill,
            requested_capabilities=request.requested_capabilities,
            declared_capabilities=request.requested_capabilities,
            risk_hint=request.risk_level,
            input_fingerprint=request.input_fingerprint,
            actor=request.actor,
            goal=request.goal,
        )
        self.audit._write_security_claim(
            self.audit._issue_security_claim(
                self.audit._claim_issuer,
                event_type,
                action,
                decision=request.status,
                reason=request.reason,
                risk_level=request.risk_level,
                policy_version=request.policy_version,
                approval_id=request.approval_id,
                actor=request.actor,
                provenance=request,
            )
        )

    def _persist_decision(self, decision: PolicyDecision) -> None:
        with self._authorization_evidence_lock:
            self._issued_policy_decisions[id(decision)] = (decision, _provenance_snapshot(decision))
        if self.store is not None and hasattr(self.store, "save_security_decision"):
            self.store.save_security_decision(decision.to_dict())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
