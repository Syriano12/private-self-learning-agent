from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from private_agent.core.experience import sanitize_value
from private_agent.core.recovery import input_fingerprint


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
    "APPROVAL_EXPIRED",
    "APPROVAL_CANCELLED",
    "ACTION_BLOCKED",
    "ACTION_EXECUTED",
    "SKILL_ACTIVATED",
    "SKILL_QUARANTINED",
    "SKILL_ROLLED_BACK",
    "POLICY_CHANGED",
}


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


@dataclass
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


class PolicyEngine:
    """Deterministic policy authority; it never consumes free-form LLM approval text."""

    def __init__(self, policy: SecurityPolicy | None = None, *, store: Any | None = None) -> None:
        self.policy = policy or SecurityPolicy()
        self.store = store
        self._persist_policy()

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
        requires_approval = risk in set(self.policy.approval_required_risks)
        if risk == "MEDIUM" and self.policy.medium_requires_approval:
            requires_approval = True
        if requested.intersection(self.policy.always_approval_capabilities):
            requires_approval = True
        if action.action_type == "skill_activation" and self.policy.allow_skill_activation_without_approval:
            requires_approval = False
        if requires_approval:
            reasons.append(f"approval_required_for_risk:{risk}")
            return self._decision(action, "REQUIRE_APPROVAL", risk, reasons, approval_required=True)
        return self._decision(action, "ALLOW", risk, reasons)

    def update_policy(self, policy: SecurityPolicy) -> SecurityPolicy:
        if policy.policy_version <= self.policy.policy_version:
            raise ValueError("policy_version_must_increase")
        self.policy = policy
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
            policy_version=self.policy.policy_version,
            approval_required=approval_required,
            reasons=reasons,
        )

    def _persist_policy(self) -> None:
        if self.store is not None and hasattr(self.store, "save_security_policy"):
            self.store.save_security_policy(self.policy.to_dict())


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

    def __post_init__(self) -> None:
        if self.status not in APPROVAL_STATUSES:
            raise ValueError(f"invalid_approval_status:{self.status}")

    def to_dict(self) -> dict[str, Any]:
        return sanitize_value(asdict(self))


class ApprovalGate:
    """Explicit human approval state machine with exact action binding and expiration."""

    SENSITIVE = {"send_message", "publish", "payment", "delete_data", "account_change", "external_irreversible"}

    def __init__(self, *, store: Any | None = None, event_logger: Callable[[str, ApprovalRequest], None] | None = None) -> None:
        self.store = store
        self.event_logger = event_logger
        self._requests: dict[str, ApprovalRequest] = {}

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
        )
        self._requests[request.approval_id] = request
        self._persist(request)
        self._emit("APPROVAL_REQUESTED", request)
        return request

    def approve(self, approval_id: str, *, actor: str = "human") -> ApprovalRequest:
        request = self._get(approval_id)
        self._expire_if_needed(request)
        if request.status != "PENDING":
            raise ValueError(f"approval_not_pending:{request.status}")
        request.status = "APPROVED"
        request.decided_at = _now()
        request.actor = actor
        self._persist(request)
        self._emit("APPROVAL_GRANTED", request)
        return request

    def deny(self, approval_id: str, *, actor: str = "human") -> ApprovalRequest:
        request = self._get(approval_id)
        self._expire_if_needed(request)
        if request.status != "PENDING":
            raise ValueError(f"approval_not_pending:{request.status}")
        request.status = "DENIED"
        request.decided_at = _now()
        request.actor = actor
        self._persist(request)
        self._emit("APPROVAL_DENIED", request)
        return request

    def cancel(self, approval_id: str, *, actor: str = "system") -> ApprovalRequest:
        request = self._get(approval_id)
        self._expire_if_needed(request)
        if request.status != "PENDING":
            raise ValueError(f"approval_not_pending:{request.status}")
        request.status = "CANCELLED"
        request.decided_at = _now()
        request.actor = actor
        self._persist(request)
        self._emit("APPROVAL_CANCELLED", request)
        return request

    def get(self, approval_id: str) -> ApprovalRequest | None:
        try:
            request = self._get(approval_id)
        except KeyError:
            return None
        self._expire_if_needed(request)
        return request

    def valid_for(
        self,
        approval_id: str,
        action: ActionRequest,
        policy_version: int,
        risk_level: str | None = None,
    ) -> bool:
        request = self.get(approval_id)
        if request is None or request.status != "APPROVED":
            return False
        binding_matches = (
            request.task_id == action.task_id
            and request.action_id == action.action_id
            and request.tool_or_skill == action.tool_or_skill
            and request.requested_capabilities == action.requested_capabilities
        )
        if risk_level is not None:
            binding_matches = binding_matches and request.risk_level == risk_level
        return binding_matches and request.input_fingerprint == action.input_fingerprint and request.policy_version == policy_version

    def check(self, action: str, *, approved: bool = False) -> ApprovalRequest | None:
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
        )
        self._requests[request.approval_id] = request
        self._persist(request)
        return request

    def allow(self, request: ApprovalRequest) -> bool:
        self._expire_if_needed(request)
        return request.status == "APPROVED"

    def _get(self, approval_id: str) -> ApprovalRequest:
        if approval_id in self._requests:
            return self._requests[approval_id]
        if self.store is not None and hasattr(self.store, "get_approval_request"):
            payload = self.store.get_approval_request(approval_id)
            if payload:
                request = ApprovalRequest(**payload)
                self._requests[approval_id] = request
                return request
        raise KeyError(f"approval_not_found:{approval_id}")

    def _expire_if_needed(self, request: ApprovalRequest) -> None:
        if request.status in {"PENDING", "APPROVED"} and _parse_time(request.expires_at) <= datetime.now(timezone.utc):
            request.status = "EXPIRED"
            request.decided_at = _now()
            self._persist(request)
            self._emit("APPROVAL_EXPIRED", request)

    def _persist(self, request: ApprovalRequest) -> None:
        if self.store is not None and hasattr(self.store, "save_approval_request"):
            self.store.save_approval_request(request.to_dict())

    def _emit(self, event_type: str, request: ApprovalRequest) -> None:
        if self.event_logger is not None:
            self.event_logger(event_type, request)


class AuditTrail:
    def __init__(self, *, store: Any | None = None) -> None:
        self.store = store
        self._events: list[dict[str, Any]] = []

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
        self._events.append(event)
        if self.store is not None and hasattr(self.store, "save_audit_event"):
            self.store.save_audit_event(event)
        return event

    def list(self, *, task_id: str | None = None) -> list[dict[str, Any]]:
        if self.store is not None and hasattr(self.store, "all_audit_events"):
            return self.store.all_audit_events(task_id=task_id)
        return [event for event in self._events if task_id is None or event["task_id"] == task_id]


class SecurityController:
    """Policy and approval control plane used both before planning execution and at the boundary."""

    def __init__(
        self,
        *,
        store: Any | None = None,
        policy_engine: PolicyEngine | None = None,
        approval_gate: ApprovalGate | None = None,
        audit_trail: AuditTrail | None = None,
    ) -> None:
        self.store = store
        self.audit = audit_trail or AuditTrail(store=store)
        self.policy_engine = policy_engine or PolicyEngine(store=store)
        self.approval_gate = approval_gate or ApprovalGate(store=store, event_logger=self._approval_event)

    def preflight(self, action: ActionRequest, *, approval_id: str = "") -> PolicyDecision:
        decision = self.policy_engine.evaluate(action)
        self._persist_decision(decision)
        self.audit.record(
            "POLICY_EVALUATED",
            action,
            decision=decision.decision,
            reason=action.reason or "; ".join(decision.reasons),
            risk_level=decision.risk_level,
            policy_version=decision.policy_version,
        )
        if decision.decision in {"DENY", "QUARANTINE"}:
            self.audit.record(
                "PERMISSION_DENIED" if decision.decision == "DENY" else "SKILL_QUARANTINED",
                action,
                decision=decision.decision,
                reason="; ".join(decision.reasons),
                risk_level=decision.risk_level,
                policy_version=decision.policy_version,
            )
            return decision
        if decision.decision == "REQUIRE_APPROVAL":
            if approval_id and self.approval_gate.valid_for(
                approval_id,
                action,
                decision.policy_version,
                decision.risk_level,
            ):
                approved = self.approval_gate.get(approval_id)
                decision.decision = "ALLOW"
                decision.approval_id = approval_id
                decision.reasons.append("exact_approval_valid")
                self.audit.record(
                    "PERMISSION_GRANTED",
                    action,
                    decision="ALLOW",
                    reason="Exact human approval matched the action binding",
                    risk_level=decision.risk_level,
                    policy_version=decision.policy_version,
                    approval_id=approval_id,
                    actor=approved.actor if approved else "human",
                )
                return decision
            request = self.approval_gate.request(action, decision, ttl_seconds=self.policy_engine.policy.approval_ttl_seconds)
            decision.approval_id = request.approval_id
            decision.reasons.append("waiting_for_explicit_human_approval")
        else:
            self.audit.record(
                "PERMISSION_GRANTED",
                action,
                decision="ALLOW",
                reason=action.reason or "Deterministic policy allowed the action",
                risk_level=decision.risk_level,
                policy_version=decision.policy_version,
            )
        return decision

    def authorize(self, action: ActionRequest, *, approval_id: str = "", boundary: bool = False) -> PolicyDecision:
        decision = self.policy_engine.evaluate(action)
        self._persist_decision(decision)
        self.audit.record(
            "POLICY_EVALUATED",
            action,
            decision=decision.decision,
            reason=action.reason or "; ".join(decision.reasons),
            risk_level=decision.risk_level,
            policy_version=decision.policy_version,
            approval_id=approval_id,
            metadata={"boundary": boundary},
        )
        if decision.decision in {"DENY", "QUARANTINE"}:
            self.audit.record(
                "ACTION_BLOCKED",
                action,
                decision=decision.decision,
                reason="; ".join(decision.reasons),
                risk_level=decision.risk_level,
                policy_version=decision.policy_version,
                approval_id=approval_id,
                metadata={"boundary": boundary},
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
                    decision.reasons.append("approval_invalid_or_not_bound")
                decision.reasons.append("execution_requires_valid_approval")
                decision.decision = "REQUIRE_APPROVAL"
                self.audit.record(
                    "ACTION_BLOCKED",
                    action,
                    decision=decision.decision,
                    reason="; ".join(decision.reasons),
                    risk_level=decision.risk_level,
                    policy_version=decision.policy_version,
                    approval_id=decision.approval_id or approval_id,
                    metadata={"boundary": boundary},
                )
                return decision
            decision.decision = "ALLOW"
            decision.approval_id = approval_id
            decision.reasons.append("exact_approval_valid")
        self.audit.record(
            "PERMISSION_GRANTED",
            action,
            decision="ALLOW",
            reason="Execution-boundary policy check passed",
            risk_level=decision.risk_level,
            policy_version=decision.policy_version,
            approval_id=approval_id,
            metadata={"boundary": boundary},
        )
        return decision

    def record_execution(self, action: ActionRequest, *, success: bool, metadata: dict[str, Any] | None = None) -> None:
        self.audit.record(
            "ACTION_EXECUTED",
            action,
            decision="ALLOW" if success else "FAILED",
            reason="Action reached the executor" if success else "Action returned a failure after authorization",
            risk_level=self.policy_engine.evaluate(action).risk_level,
            policy_version=self.policy_engine.policy.policy_version,
            metadata=metadata,
        )

    def record_blocked(self, action: ActionRequest, *, reason: str, metadata: dict[str, Any] | None = None) -> None:
        self.audit.record(
            "ACTION_BLOCKED",
            action,
            decision="BLOCKED",
            reason=reason,
            risk_level=self.policy_engine.evaluate(action).risk_level,
            policy_version=self.policy_engine.policy.policy_version,
            metadata=metadata,
        )

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
            action = self.action_for_tool(task_id, step.id, tool, step.input)
            approval_id = approval_ids.get(step.id) or approval_ids.get(step.tool, "")
            decisions.append(self.preflight(action, approval_id=approval_id))
        return decisions

    def action_for_tool(self, task_id: str, action_id: str, tool: Any, inputs: dict[str, Any]) -> ActionRequest:
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
        )

    def action_for_skill(self, task_id: str, action_id: str, candidate: Any, inputs: dict[str, Any]) -> ActionRequest:
        metadata = candidate.metadata if isinstance(getattr(candidate, "metadata", {}), dict) else {}
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
        )

    def authorize_skill_activation(self, task_id: str, candidate: Any, *, approval_id: str = "") -> PolicyDecision:
        action = self.action_for_skill(task_id, f"activate:{candidate.name}:{candidate.version}", candidate, {})
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
        self.audit.record(
            "POLICY_CHANGED",
            action,
            decision="ALLOW",
            reason=f"policy_version:{previous.policy_version}->{updated.policy_version}",
            policy_version=updated.policy_version,
            actor=actor,
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
        )
        self.audit.record(
            event_type,
            action,
            decision=request.status,
            reason=request.reason,
            risk_level=request.risk_level,
            policy_version=request.policy_version,
            approval_id=request.approval_id,
            actor=request.actor,
        )

    def _persist_decision(self, decision: PolicyDecision) -> None:
        if self.store is not None and hasattr(self.store, "save_security_decision"):
            self.store.save_security_decision(decision.to_dict())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
