from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from private_agent.core.experience import sanitize_value


STATE_VERSION = 1
TASK_STATUSES = {
    "CREATED",
    "PLANNING",
    "READY",
    "EXECUTING",
    "EXECUTION_UNKNOWN",
    "OBSERVING",
    "VERIFYING",
    "WAITING_APPROVAL",
    "BLOCKED",
    "FAILED",
    "COMPLETED",
}
TERMINAL_STATUSES = {"BLOCKED", "FAILED", "COMPLETED"}

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "CREATED": {"CREATED", "PLANNING", "READY", "WAITING_APPROVAL", "BLOCKED", "FAILED"},
    "PLANNING": {"PLANNING", "READY", "WAITING_APPROVAL", "BLOCKED", "FAILED"},
    "READY": {"READY", "EXECUTING", "WAITING_APPROVAL", "COMPLETED", "BLOCKED", "FAILED", "EXECUTION_UNKNOWN"},
    "EXECUTING": {"EXECUTING", "OBSERVING", "EXECUTION_UNKNOWN", "BLOCKED", "FAILED"},
    "EXECUTION_UNKNOWN": {"EXECUTION_UNKNOWN", "OBSERVING", "VERIFYING", "READY", "BLOCKED", "FAILED"},
    "OBSERVING": {"OBSERVING", "VERIFYING", "EXECUTION_UNKNOWN", "READY", "BLOCKED", "FAILED"},
    "VERIFYING": {"VERIFYING", "READY", "COMPLETED", "BLOCKED", "FAILED"},
    "WAITING_APPROVAL": {"WAITING_APPROVAL", "READY", "BLOCKED", "FAILED"},
    "BLOCKED": {"BLOCKED"},
    "FAILED": {"FAILED"},
    "COMPLETED": {"COMPLETED"},
}


class StateIntegrityError(ValueError):
    """Raised when durable task state cannot be trusted safely."""


class InvalidStateTransition(StateIntegrityError):
    """Raised when a state machine transition is not explicitly allowed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_fingerprint(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]


def plan_fingerprint(plan: dict[str, Any]) -> str:
    return canonical_fingerprint(plan)


def action_fingerprint(*, plan_version: int, step_id: str, tool: str, inputs: dict[str, Any]) -> str:
    return canonical_fingerprint(
        {
            "plan_version": int(plan_version),
            "step_id": step_id,
            "tool": tool,
            "input": inputs,
        }
    )


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
        actions.append(action_fingerprint(plan_version=plan_version, step_id=step_id, tool=tool, inputs=inputs))
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
    def from_dict(cls, payload: dict[str, Any]) -> "TaskState":
        if not isinstance(payload, dict):
            raise StateIntegrityError("state_must_be_object")
        required = {
            "task_id", "goal", "status", "current_phase", "current_step", "plan_version",
            "plan_fingerprint", "plan", "completed_actions", "pending_actions", "failed_actions",
            "observations", "verification_results", "recovery_state", "action_records", "approval_ids",
            "security_decisions", "context", "result", "state_version", "state_fingerprint",
            "created_at", "updated_at", "last_checkpoint_at", "last_successful_checkpoint",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise StateIntegrityError(f"missing_state_fields:{','.join(missing)}")
        if int(payload["state_version"]) != STATE_VERSION:
            raise StateIntegrityError(f"unsupported_state_version:{payload['state_version']}")
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
            state_version=int(payload["state_version"]),
            state_fingerprint=str(payload["state_fingerprint"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            last_checkpoint_at=str(payload["last_checkpoint_at"]),
            last_successful_checkpoint=str(payload["last_successful_checkpoint"]),
        )
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

    def transition(self, status: str, *, phase: str | None = None, step_id: str | None = None) -> None:
        if status not in TASK_STATUSES:
            raise InvalidStateTransition(f"unknown_task_status:{status}")
        if status not in _ALLOWED_TRANSITIONS.get(self.status, set()):
            raise InvalidStateTransition(f"invalid_transition:{self.status}->{status}")
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
        self.plan = sanitize_value(plan)
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
                "updated_at": utc_now(),
            },
        )

    def mark_action(self, key: str, *, status: str, step_id: str, tool_name: str, input_fingerprint: str, **fields: Any) -> None:
        record = self.action_records.setdefault(key, {"action_key": key})
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
        if not self.task_id or not self.goal:
            raise StateIntegrityError("task_identity_required")
        if not isinstance(self.plan, dict):
            raise StateIntegrityError("plan_must_be_object")
        if self.plan:
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
        completed = set(self.completed_actions)
        result = []
        for raw in self.plan.get("steps", []):
            key = action_fingerprint(
                plan_version=self.plan_version,
                step_id=raw["id"],
                tool=raw["tool"],
                inputs=raw["input"],
            )
            if key not in completed:
                result.append(raw)
        return result
