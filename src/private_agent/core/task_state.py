from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from private_agent.core.planning_provenance import (
    PLANNING_MODES,
    UNKNOWN,
    planning_mode_from_plan,
    plan_version_from_state,
)

from private_agent.core.experience import sanitize_value


STATE_VERSION = 2
MAX_TASK_GOAL_LENGTH = 4000
_STOP_REASON_REFERENCE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_CLARIFICATION_ID = re.compile(r"^[0-9a-f-]{36}$")
MAX_CLARIFICATION_REQUESTS = 16
MAX_CLARIFICATION_ITEMS = 16
MAX_CLARIFICATION_ITEM_LENGTH = 200
_LEGACY_STATE_FIELDS = frozenset(
    {
        "task_id", "goal", "status", "current_phase", "current_step", "plan_version",
        "plan_fingerprint", "plan", "completed_actions", "pending_actions", "failed_actions",
        "observations", "verification_results", "recovery_state", "action_records", "approval_ids",
        "security_decisions", "context", "result", "state_version", "state_fingerprint",
        "created_at", "updated_at", "last_checkpoint_at", "last_successful_checkpoint",
    }
)
TASK_STATUSES = {
    "CREATED",
    "QUEUED",
    "PLANNING",
    "READY",
    "EXECUTING",
    "EXECUTION_UNKNOWN",
    "OBSERVING",
    "VERIFYING",
    "WAITING_APPROVAL",
    "WAITING_CLARIFICATION",
    "BLOCKED",
    "FAILED",
    "COMPLETED",
}
TERMINAL_STATUSES = {"BLOCKED", "FAILED", "COMPLETED"}

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "CREATED": {"CREATED", "PLANNING", "READY", "WAITING_APPROVAL", "WAITING_CLARIFICATION", "BLOCKED", "FAILED", "EXECUTION_UNKNOWN"},
    "QUEUED": {"QUEUED", "PLANNING", "READY", "WAITING_APPROVAL", "WAITING_CLARIFICATION", "BLOCKED", "FAILED", "EXECUTION_UNKNOWN"},
    "PLANNING": {"PLANNING", "READY", "WAITING_APPROVAL", "WAITING_CLARIFICATION", "BLOCKED", "FAILED", "EXECUTION_UNKNOWN"},
    "READY": {"READY", "EXECUTING", "WAITING_APPROVAL", "COMPLETED", "BLOCKED", "FAILED", "EXECUTION_UNKNOWN"},
    "EXECUTING": {"EXECUTING", "OBSERVING", "EXECUTION_UNKNOWN", "BLOCKED", "FAILED"},
    "EXECUTION_UNKNOWN": {"EXECUTION_UNKNOWN", "OBSERVING", "VERIFYING", "READY", "BLOCKED", "FAILED"},
    "OBSERVING": {"OBSERVING", "VERIFYING", "EXECUTION_UNKNOWN", "READY", "BLOCKED", "FAILED"},
    "VERIFYING": {"VERIFYING", "READY", "COMPLETED", "BLOCKED", "FAILED"},
    "WAITING_APPROVAL": {"WAITING_APPROVAL", "READY", "BLOCKED", "FAILED"},
    "WAITING_CLARIFICATION": {"WAITING_CLARIFICATION", "QUEUED", "READY", "BLOCKED", "FAILED"},
    "BLOCKED": {"BLOCKED"},
    "FAILED": {"FAILED"},
    "COMPLETED": {"COMPLETED"},
}


class StateIntegrityError(ValueError):
    """Raised when durable task state cannot be trusted safely."""


class InvalidStateTransition(StateIntegrityError):
    """Raised when a state machine transition is not explicitly allowed."""


class TaskStopReasonCode(str, Enum):
    """Closed, non-free-text causes set only at explicit stop transitions."""

    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    EXECUTION_UNKNOWN = "EXECUTION_UNKNOWN"
    TASK_FAILED = "TASK_FAILED"
    WORKER_FAILED = "WORKER_FAILED"


@dataclass(frozen=True)
class TaskStopReason:
    """Safe public projection of a stop cause; never stores explanatory text."""

    code: TaskStopReasonCode
    step_id: str | None = None
    action_id: str | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "code", TaskStopReasonCode(self.code))
        except (TypeError, ValueError) as exc:
            raise StateIntegrityError("task_stop_reason_code_invalid") from exc
        for name in ("step_id", "action_id"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or _STOP_REASON_REFERENCE.fullmatch(value) is None
            ):
                raise StateIntegrityError(f"task_stop_reason_{name}_invalid")

    @classmethod
    def create(
        cls,
        code: TaskStopReasonCode,
        *,
        step_id: Any = None,
        action_id: Any = None,
    ) -> "TaskStopReason":
        def safe_reference(value: Any) -> str | None:
            if isinstance(value, str) and _STOP_REASON_REFERENCE.fullmatch(value):
                return value
            return None

        return cls(
            code=code,
            step_id=safe_reference(step_id),
            action_id=safe_reference(action_id),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskStopReason":
        if not isinstance(payload, dict) or "code" not in payload:
            raise StateIntegrityError("task_stop_reason_must_be_object")
        if set(payload) - {"code", "step_id", "action_id"}:
            raise StateIntegrityError("task_stop_reason_fields_invalid")
        try:
            code = TaskStopReasonCode(payload["code"])
        except (TypeError, ValueError) as exc:
            raise StateIntegrityError("task_stop_reason_code_invalid") from exc
        return cls(code=code, step_id=payload.get("step_id"), action_id=payload.get("action_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "step_id": self.step_id,
            "action_id": self.action_id,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_fingerprint(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]


def plan_fingerprint(plan: dict[str, Any]) -> str:
    return canonical_fingerprint(sanitize_value(plan))


def _contains_redacted(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_redacted(key) or _contains_redacted(item) for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_redacted(item) for item in value)
    return isinstance(value, str) and "[REDACTED]" in value


def action_fingerprint(
    *,
    plan_version: int,
    step_id: str,
    tool: str,
    inputs: dict[str, Any],
    specialist_assignment: dict[str, Any] | None = None,
) -> str:
    payload = {
        "plan_version": int(plan_version),
        "step_id": step_id,
        "tool": tool,
        "input": inputs,
    }
    if specialist_assignment:
        payload["specialist_assignment"] = specialist_assignment
    return canonical_fingerprint(payload)


def plan_actions(plan: dict[str, Any], plan_version: int) -> list[str]:
    actions: list[str] = []
    for raw in plan.get("steps", []):
        if not isinstance(raw, dict):
            raise StateIntegrityError("plan_step_must_be_object")
        step_id = raw.get("id")
        tool = raw.get("tool")
        inputs = raw.get("input")
        if not isinstance(step_id, str) or not step_id or not isinstance(tool, str) or not isinstance(inputs, dict):
            raise StateIntegrityError("plan_step_missing_identity")
        assignment = raw.get("specialist_assignment", {})
        if not isinstance(assignment, dict):
            raise StateIntegrityError("plan_step_specialist_assignment_invalid")
        actions.append(
            action_fingerprint(
                plan_version=plan_version,
                step_id=step_id,
                tool=tool,
                inputs=inputs,
                specialist_assignment=assignment,
            )
        )
    return actions


@dataclass
class TaskState:
    task_id: str
    goal: str
    status: str = "CREATED"
    current_phase: str = "TASK_CREATED"
    current_step: str = ""
    plan_version: int = 0
    plan_fingerprint: str = ""
    plan: dict[str, Any] = field(default_factory=dict)
    completed_actions: list[str] = field(default_factory=list)
    pending_actions: list[str] = field(default_factory=list)
    failed_actions: list[str] = field(default_factory=list)
    observations: dict[str, dict[str, Any]] = field(default_factory=dict)
    verification_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    recovery_state: dict[str, Any] = field(default_factory=dict)
    action_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    approval_ids: dict[str, str] = field(default_factory=dict)
    security_decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    stop_reason: TaskStopReason | None = None
    state_version: int = STATE_VERSION
    state_fingerprint: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_checkpoint_at: str = ""
    last_successful_checkpoint: str = ""

    @classmethod
    def create(cls, task_id: str, goal: str) -> "TaskState":
        state = cls(task_id=task_id, goal=goal)
        state.touch()
        return state

    @classmethod
    def create_queued(cls, task_id: str, goal: str, *, context: dict[str, Any] | None = None) -> "TaskState":
        state = cls(task_id=task_id, goal=goal, status="QUEUED", current_phase="QUEUED")
        if context is not None:
            if not isinstance(context, dict):
                raise StateIntegrityError("task_context_must_be_object")
            state.context = dict(context)
        state.checkpoint("TASK_QUEUED", phase="QUEUED", successful=True)
        return state

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskState":
        if not isinstance(payload, dict):
            raise StateIntegrityError("state_must_be_object")
        try:
            serialized_version = int(payload["state_version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StateIntegrityError("state_version_invalid") from exc
        if serialized_version not in {1, STATE_VERSION}:
            raise StateIntegrityError(f"unsupported_state_version:{serialized_version}")
        required = _LEGACY_STATE_FIELDS | ({"stop_reason"} if serialized_version == STATE_VERSION else set())
        missing = sorted(required - set(payload))
        if missing:
            raise StateIntegrityError(f"missing_state_fields:{','.join(missing)}")
        migrating_legacy = serialized_version == 1
        if migrating_legacy:
            # v1 fingerprints covered only the original fields. Ignore any
            # unversioned extension (including a forged stop_reason), verify
            # the original payload, then migrate with an unavailable reason.
            legacy_payload = {name: payload[name] for name in _LEGACY_STATE_FIELDS}
            legacy_fingerprint = payload["state_fingerprint"]
            legacy_payload["state_fingerprint"] = ""
            if (
                not isinstance(legacy_fingerprint, str)
                or not legacy_fingerprint
                or canonical_fingerprint(sanitize_value(legacy_payload)) != legacy_fingerprint
            ):
                raise StateIntegrityError("state_fingerprint_mismatch")
        state = cls(
            task_id=str(payload["task_id"]),
            goal=str(payload["goal"]),
            status=str(payload["status"]),
            current_phase=str(payload["current_phase"]),
            current_step=str(payload["current_step"]),
            plan_version=int(payload["plan_version"]),
            plan_fingerprint=str(payload["plan_fingerprint"]),
            plan=dict(payload["plan"]),
            completed_actions=list(payload["completed_actions"]),
            pending_actions=list(payload["pending_actions"]),
            failed_actions=list(payload["failed_actions"]),
            observations=dict(payload["observations"]),
            verification_results=dict(payload["verification_results"]),
            recovery_state=dict(payload["recovery_state"]),
            action_records=dict(payload["action_records"]),
            approval_ids=dict(payload["approval_ids"]),
            security_decisions=dict(payload["security_decisions"]),
            context=dict(payload["context"]),
            result=dict(payload["result"]),
            stop_reason=(
                None
                if migrating_legacy or payload["stop_reason"] is None
                else TaskStopReason.from_dict(payload["stop_reason"])
            ),
            state_version=STATE_VERSION,
            state_fingerprint="" if migrating_legacy else str(payload["state_fingerprint"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            last_checkpoint_at=str(payload["last_checkpoint_at"]),
            last_successful_checkpoint=str(payload["last_successful_checkpoint"]),
        )
        if migrating_legacy:
            state.refresh_fingerprint()
        state.validate()
        return state

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "goal": self.goal,
            "status": self.status,
            "current_phase": self.current_phase,
            "current_step": self.current_step,
            "plan_version": self.plan_version,
            "plan_fingerprint": self.plan_fingerprint,
            "plan": self.plan,
            "completed_actions": self.completed_actions,
            "pending_actions": self.pending_actions,
            "failed_actions": self.failed_actions,
            "observations": self.observations,
            "verification_results": self.verification_results,
            "recovery_state": self.recovery_state,
            "action_records": self.action_records,
            "approval_ids": self.approval_ids,
            "security_decisions": self.security_decisions,
            "context": self.context,
            "result": self.result,
            "stop_reason": self.stop_reason.to_dict() if self.stop_reason is not None else None,
            "state_version": self.state_version,
            "state_fingerprint": self.state_fingerprint,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_checkpoint_at": self.last_checkpoint_at,
            "last_successful_checkpoint": self.last_successful_checkpoint,
        }
        return sanitize_value(payload)

    def _fingerprint_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["state_fingerprint"] = ""
        return payload

    def refresh_fingerprint(self) -> str:
        self.state_fingerprint = canonical_fingerprint(self._fingerprint_payload())
        return self.state_fingerprint

    def touch(self) -> None:
        self.updated_at = utc_now()
        self.refresh_fingerprint()

    def set_stop_reason(
        self,
        code: TaskStopReasonCode | None,
        *,
        step_id: Any = None,
        action_id: Any = None,
    ) -> None:
        self.stop_reason = (
            None
            if code is None
            else TaskStopReason.create(code, step_id=step_id, action_id=action_id)
        )
        self.touch()

    def transition(self, status: str, *, phase: str | None = None, step_id: str | None = None) -> None:
        if status not in TASK_STATUSES:
            raise InvalidStateTransition(f"unknown_task_status:{status}")
        if status not in _ALLOWED_TRANSITIONS.get(self.status, set()):
            raise InvalidStateTransition(f"invalid_transition:{self.status}->{status}")
        if self.status in {"WAITING_APPROVAL", "WAITING_CLARIFICATION", "EXECUTION_UNKNOWN"} and status != self.status:
            self.stop_reason = None
        self.status = status
        if phase is not None:
            self.current_phase = phase
        if step_id is not None:
            self.current_step = step_id
        self.touch()

    def checkpoint(self, checkpoint_type: str, *, status: str | None = None, phase: str | None = None, step_id: str | None = None, successful: bool = False) -> None:
        if status is not None:
            self.transition(status, phase=phase, step_id=step_id)
        elif phase is not None or step_id is not None:
            if phase is not None:
                self.current_phase = phase
            if step_id is not None:
                self.current_step = step_id
            self.touch()
        self.last_checkpoint_at = utc_now()
        if successful:
            self.last_successful_checkpoint = checkpoint_type
        self.touch()

    def set_plan(self, plan: dict[str, Any], *, version: int | None = None) -> None:
        if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list) or not plan["steps"]:
            raise StateIntegrityError("plan_required_for_persistence")
        # Keep the in-memory plan for the current, already-authorized run, but
        # persist only its sanitized projection.  A later reload detects the
        # redacted marker and refuses to execute it as if it were the original
        # input; callers must create a fresh authorized plan instead.
        try:
            json.dumps(sanitize_value(plan), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise StateIntegrityError("plan_not_json_safe") from exc
        # JSON round-tripping detaches nested planner objects without invoking
        # arbitrary __deepcopy__ hooks from an untrusted provider.
        detached_plan = json.loads(json.dumps(plan, ensure_ascii=False, allow_nan=False))
        planning_mode = detached_plan.get("planning_mode", UNKNOWN)
        if not isinstance(planning_mode, str) or planning_mode not in PLANNING_MODES:
            raise StateIntegrityError("plan_planning_mode_invalid")
        detached_plan["planning_mode"] = planning_mode
        self.plan = detached_plan
        self.plan_version = int(version if version is not None else self.plan_version + 1)
        self.plan_fingerprint = plan_fingerprint(self.plan)
        actions = plan_actions(self.plan, self.plan_version)
        self.pending_actions = [key for key in actions if key not in self.completed_actions]
        self.current_step = ""
        self.touch()

    def action_key_for_step(self, step: Any) -> str:
        return action_fingerprint(
            plan_version=self.plan_version,
            step_id=str(step.id),
            tool=str(step.tool),
            inputs=dict(step.input),
            specialist_assignment=getattr(step, "specialist_assignment", None),
        )

    def action_record(self, key: str) -> dict[str, Any]:
        return self.action_records.setdefault(
            key,
            {
                "action_key": key,
                "status": "NOT_STARTED",
                "execution_status": "",
                "step_id": self.current_step,
                "tool_name": "",
                "input_fingerprint": "",
                "output": None,
                "observation": None,
                "verification": None,
                "approval_id": "",
                "planning_mode": planning_mode_from_plan(self.plan),
                "plan_version": plan_version_from_state({"plan_version": self.plan_version if self.plan else 0}),
                "updated_at": utc_now(),
            },
        )

    def mark_action(self, key: str, *, status: str, step_id: str, tool_name: str, input_fingerprint: str, **fields: Any) -> None:
        record = self.action_records.setdefault(key, {"action_key": key})
        fields.setdefault("planning_mode", planning_mode_from_plan(self.plan))
        fields.setdefault(
            "plan_version",
            plan_version_from_state({"plan_version": self.plan_version if self.plan else 0}),
        )
        record.update(
            {
                "action_key": key,
                "status": status,
                "step_id": step_id,
                "tool_name": tool_name,
                "input_fingerprint": input_fingerprint,
                "updated_at": utc_now(),
                **fields,
            }
        )
        if status == "VERIFIED" and key not in self.completed_actions:
            self.completed_actions.append(key)
        if status == "FAILED" and key not in self.failed_actions:
            self.failed_actions.append(key)
        self.pending_actions = [item for item in self.pending_actions if item not in self.completed_actions]
        self.touch()

    def validate(self) -> None:
        if self.state_version != STATE_VERSION:
            raise StateIntegrityError(f"unsupported_state_version:{self.state_version}")
        if self.status not in TASK_STATUSES:
            raise StateIntegrityError(f"unknown_task_status:{self.status}")
        if self.stop_reason is not None:
            if not isinstance(self.stop_reason, TaskStopReason):
                raise StateIntegrityError("task_stop_reason_invalid")
            allowed_statuses = {
                TaskStopReasonCode.APPROVAL_REQUIRED: {"WAITING_APPROVAL"},
                TaskStopReasonCode.CLARIFICATION_REQUIRED: {"WAITING_CLARIFICATION"},
                TaskStopReasonCode.POLICY_BLOCKED: {"BLOCKED"},
                TaskStopReasonCode.EXECUTION_UNKNOWN: {"EXECUTION_UNKNOWN", "BLOCKED"},
                TaskStopReasonCode.TASK_FAILED: {"FAILED"},
                TaskStopReasonCode.WORKER_FAILED: {"FAILED"},
            }
            if self.status not in allowed_statuses[self.stop_reason.code]:
                raise StateIntegrityError("task_stop_reason_status_mismatch")
        clarifications = self.context.get("clarifications", [])
        if not isinstance(clarifications, list) or len(clarifications) > MAX_CLARIFICATION_REQUESTS:
            raise StateIntegrityError("task_clarifications_invalid")
        request_ids: set[str] = set()
        pending_clarifications = 0
        for item in clarifications:
            required_fields = {
                "request_id", "question", "missing_information", "status",
                "answer_message_id", "created_at", "answered_at",
            }
            if not isinstance(item, dict) or set(item) != required_fields:
                raise StateIntegrityError("task_clarification_fields_invalid")
            request_id = item.get("request_id")
            question = item.get("question")
            missing = item.get("missing_information")
            status = item.get("status")
            answer_id = item.get("answer_message_id")
            if (
                not isinstance(request_id, str) or _CLARIFICATION_ID.fullmatch(request_id) is None
                or request_id in request_ids
                or not isinstance(question, str) or not question.strip() or len(question) > 1000
                or not isinstance(missing, list) or not 1 <= len(missing) <= MAX_CLARIFICATION_ITEMS
                or any(not isinstance(value, str) or not value.strip() or len(value) > MAX_CLARIFICATION_ITEM_LENGTH for value in missing)
                or status not in {"PENDING", "ANSWERED"}
                or not isinstance(item.get("created_at"), str) or not item["created_at"]
                or not isinstance(item.get("answered_at"), str)
            ):
                raise StateIntegrityError("task_clarification_invalid")
            request_ids.add(request_id)
            if status == "PENDING":
                pending_clarifications += 1
                if answer_id != "" or item["answered_at"] != "":
                    raise StateIntegrityError("task_clarification_pending_answer_invalid")
            elif (
                not isinstance(answer_id, str) or _STOP_REASON_REFERENCE.fullmatch(answer_id) is None
                or not item["answered_at"]
            ):
                raise StateIntegrityError("task_clarification_answer_invalid")
        if pending_clarifications > 1 or (pending_clarifications and self.status != "WAITING_CLARIFICATION"):
            raise StateIntegrityError("task_clarification_status_mismatch")
        if (self.status == "WAITING_CLARIFICATION") != (pending_clarifications == 1):
            raise StateIntegrityError("task_clarification_request_missing")
        if not self.task_id or not self.goal:
            raise StateIntegrityError("task_identity_required")
        if not isinstance(self.plan, dict):
            raise StateIntegrityError("plan_must_be_object")
        if _contains_redacted(self.plan):
            raise StateIntegrityError("redacted_execution_input_requires_resolution")
        valid_actions: set[str] = set()
        if self.plan:
            planning_mode = self.plan.get("planning_mode", UNKNOWN)
            if not isinstance(planning_mode, str) or planning_mode not in PLANNING_MODES:
                raise StateIntegrityError("plan_planning_mode_invalid")
            if self.plan_version < 1 or not self.plan_fingerprint:
                raise StateIntegrityError("plan_version_and_fingerprint_required")
            if plan_fingerprint(self.plan) != self.plan_fingerprint:
                raise StateIntegrityError("plan_fingerprint_mismatch")
            valid_actions = set(plan_actions(self.plan, self.plan_version))
            known = set(self.completed_actions) | set(self.pending_actions) | set(self.failed_actions)
            if not known.issubset(valid_actions | set(self.action_records)):
                raise StateIntegrityError("state_action_not_in_plan")
            if set(self.completed_actions) & set(self.pending_actions):
                raise StateIntegrityError("completed_action_pending")
            for key in self.completed_actions:
                if self.action_records.get(key, {}).get("status") != "VERIFIED":
                    raise StateIntegrityError("completed_action_without_verification")
        if self.status == "COMPLETED":
            verification = self.result.get("verification", {})
            if verification.get("verified") is not True:
                details = verification.get("details", {}) if isinstance(verification, dict) else {}
                step_statuses = details.get("step_statuses", []) if isinstance(details, dict) else []
                limited_static_completion = (
                    self.result.get("final_status") == "COMPLETED"
                    and verification.get("status") == "INSUFFICIENT"
                    and verification.get("verification_scope") == "STATIC_ONLY"
                    and verification.get("security_impact_verified") is False
                    and details.get("code") == "security_impact_not_verified"
                    and details.get("verification_scope") == "STATIC_ONLY"
                    and details.get("security_impact_verified") is False
                    and isinstance(step_statuses, list)
                    and len(step_statuses) == len(valid_actions)
                    and details.get("step_count") == len(valid_actions)
                    and all(item == "VERIFIED" for item in step_statuses)
                    and bool(valid_actions)
                    and len(self.completed_actions) == len(valid_actions)
                    and set(self.completed_actions) == valid_actions
                    and all(self.action_records.get(key, {}).get("status") == "VERIFIED" for key in valid_actions)
                )
                if not limited_static_completion:
                    raise StateIntegrityError("completed_without_persisted_verification")
        expected = self.state_fingerprint
        if not expected:
            raise StateIntegrityError("state_fingerprint_required")
        self.state_fingerprint = ""
        actual = canonical_fingerprint(self._fingerprint_payload())
        self.state_fingerprint = expected
        if expected != actual:
            raise StateIntegrityError("state_fingerprint_mismatch")

    def safe_pending_steps(self) -> list[dict[str, Any]]:
        if not self.plan:
            return []
        if _contains_redacted(self.plan):
            raise StateIntegrityError("redacted_execution_input_requires_resolution")
        completed = set(self.completed_actions)
        result = []
        for raw in self.plan.get("steps", []):
            key = action_fingerprint(
                plan_version=self.plan_version,
                step_id=raw["id"],
                tool=raw["tool"],
                inputs=raw["input"],
                specialist_assignment=raw.get("specialist_assignment"),
            )
            if key not in completed:
                result.append(raw)
        return result
