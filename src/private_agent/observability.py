from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from private_agent.core.experience import sanitize_value


EVENT_SCHEMA_VERSION = 1
SEVERITIES = {"DEBUG", "INFO", "NOTICE", "WARNING", "ERROR", "CRITICAL"}

# The vocabulary is deliberately closed: callers cannot turn arbitrary log strings
# into timeline events that later code would mistake for lifecycle facts.
EVENT_TYPES = frozenset(
    {
        "TASK_CREATED",
        "TASK_RESUMED",
        "TASK_COMPLETED",
        "TASK_FAILED",
        "TASK_BLOCKED",
        "PLAN_CREATED",
        "PLAN_UPDATED",
        "PLAN_REJECTED",
        "ACTION_PROPOSED",
        "ACTION_STARTED",
        "ACTION_COMPLETED",
        "ACTION_FAILED",
        "ACTION_BLOCKED",
        "EXECUTION_UNKNOWN",
        "OBSERVATION_STARTED",
        "OBSERVATION_COMPLETED",
        "VERIFICATION_STARTED",
        "VERIFICATION_PASSED",
        "VERIFICATION_FAILED",
        "RECOVERY_STARTED",
        "RECOVERY_COMPLETED",
        "RECOVERY_REQUIRED",
        "REPLAN_STARTED",
        "REPLAN_COMPLETED",
        "APPROVAL_REQUESTED",
        "APPROVAL_APPROVED",
        "APPROVAL_DENIED",
        "APPROVAL_EXPIRED",
        "APPROVAL_INVALIDATED",
        "SECURITY_ALLOW",
        "SECURITY_DENY",
        "SECURITY_REQUIRE_APPROVAL",
        "SECURITY_QUARANTINE",
        "SKILL_CANDIDATE",
        "SKILL_CONTRACT_VALIDATED",
        "SKILL_AST_ANALYZED",
        "SKILL_MUTATION_TESTED",
        "SKILL_SANDBOXED",
        "SKILL_VERIFIED",
        "SKILL_APPROVAL_REQUESTED",
        "SKILL_REGISTRY_UPDATED",
        "SKILL_ACTIVATED",
        "SKILL_BLOCKED",
        "SKILL_ROLLED_BACK",
        "CHECKPOINT_SAVED",
        "RESUME_BLOCKED",
        "MEMORY_RETRIEVED",
        "MEMORY_STORED",
        "REFLECTION_STARTED",
        "REFLECTION_COMPLETED",
        "LEARNING_STARTED",
        "LEARNING_COMPLETED",
        "LEARNING_REJECTED",
        "TOOL_STARTED",
        "TOOL_COMPLETED",
        "TOOL_FAILED",
        "OBSERVABILITY_DEGRADED",
    }
)

_FAILURE_EVENTS = frozenset(
    {
        "TASK_FAILED",
        "TASK_BLOCKED",
        "PLAN_REJECTED",
        "ACTION_FAILED",
        "ACTION_BLOCKED",
        "EXECUTION_UNKNOWN",
        "VERIFICATION_FAILED",
        "SECURITY_DENY",
        "SECURITY_REQUIRE_APPROVAL",
        "SECURITY_QUARANTINE",
        "SKILL_BLOCKED",
        "TOOL_FAILED",
        "RESUME_BLOCKED",
        "LEARNING_REJECTED",
        "OBSERVABILITY_DEGRADED",
    }
)

_SENSITIVE_METADATA_KEYS = {
    "input",
    "inputs",
    "output",
    "outputs",
    "payload",
    "raw_payload",
    "raw_input",
    "raw_output",
    "headers",
    "authorization",
    "credentials",
}


class ObservabilityError(RuntimeError):
    """Base error for the observability layer."""


class ObservabilityValidationError(ObservabilityError, ValueError):
    pass


class ObservabilityStorageError(ObservabilityError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_metadata(value: Any, *, key: str = "") -> Any:
    """Redact secrets and summarize raw/large values deterministically."""
    lowered = key.lower()
    if lowered in _SENSITIVE_METADATA_KEYS:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return {
            "redacted": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "type": type(value).__name__,
        }
    if isinstance(value, dict):
        return {str(name): _safe_metadata(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item, key=key) for item in value[:100]]
    sanitized = sanitize_value(value, key=key)
    if isinstance(sanitized, str) and len(sanitized) > 512:
        raw = sanitized.encode("utf-8")
        return {
            "truncated": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "preview": sanitized[:160],
        }
    return sanitized


def safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ObservabilityValidationError("metadata_must_be_object")
    value = _safe_metadata(metadata)
    return value if isinstance(value, dict) else {}


@dataclass
class StructuredEvent:
    event_type: str
    component: str
    severity: str = "INFO"
    task_id: str = ""
    action_id: str = ""
    step_id: str = ""
    correlation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=_now)
    schema_version: int = EVENT_SCHEMA_VERSION
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ObservabilityValidationError(f"invalid_event_type:{self.event_type}")
        if not self.component or not isinstance(self.component, str):
            raise ObservabilityValidationError("component_required")
        if self.severity not in SEVERITIES:
            raise ObservabilityValidationError(f"invalid_event_severity:{self.severity}")
        if int(self.schema_version) != EVENT_SCHEMA_VERSION:
            raise ObservabilityValidationError(f"unsupported_event_schema:{self.schema_version}")
        if not self.event_id or not self.timestamp:
            raise ObservabilityValidationError("event_identity_required")
        if not self.correlation_id:
            self.correlation_id = f"task:{self.task_id}" if self.task_id else "system"
        self.metadata = safe_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "action_id": self.action_id,
            "step_id": self.step_id,
            "correlation_id": self.correlation_id,
            "component": self.component,
            "severity": self.severity,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "metadata": safe_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructuredEvent":
        if not isinstance(payload, dict):
            raise ObservabilityValidationError("event_must_be_object")
        required = {"event_id", "event_type", "timestamp", "correlation_id", "component", "severity", "schema_version", "metadata"}
        missing = sorted(required - set(payload))
        if missing:
            raise ObservabilityValidationError(f"event_fields_missing:{','.join(missing)}")
        return cls(
            event_id=str(payload["event_id"]),
            event_type=str(payload["event_type"]),
            timestamp=str(payload["timestamp"]),
            task_id=str(payload.get("task_id", "")),
            action_id=str(payload.get("action_id", "")),
            step_id=str(payload.get("step_id", "")),
            correlation_id=str(payload["correlation_id"]),
            component=str(payload["component"]),
            severity=str(payload["severity"]),
            schema_version=int(payload["schema_version"]),
            metadata=dict(payload["metadata"]),
            sequence=int(payload.get("sequence", 0)),
        )


@dataclass
class _TaskTimer:
    started: float
    segment: int = 0


class EventStore:
    """Small observability-specific facade over the existing SQLite Store."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def save(self, event: StructuredEvent | dict[str, Any]) -> int:
        payload = event.to_dict() if isinstance(event, StructuredEvent) else event
        return int(self.store.save_observability_event(payload))

    def timeline(self, task_id: str) -> list[dict[str, Any]]:
        return self.store.observability_events_for_task(task_id)

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.recent_observability_events(limit)

    def all(self) -> list[dict[str, Any]]:
        return self.store.all_observability_events()


class ObservabilityMonitor:
    """Structured telemetry facade; it observes and never authorizes or executes."""

    def __init__(self, store: Any, *, component: str = "agent-runtime", fail_mode: str = "degraded") -> None:
        if fail_mode not in {"degraded", "raise"}:
            raise ValueError("invalid_observability_fail_mode")
        self.store = store
        self.event_store = EventStore(store)
        self.component = component
        self.fail_mode = fail_mode
        self._task_timers: dict[str, _TaskTimer] = {}
        self.degraded_writes = 0
        self.last_storage_error = ""
        # Existing stores/registries can discover the same monitor without a
        # second database or a second execution path.
        setattr(store, "observability", self)

    def emit(
        self,
        event_type: str,
        *,
        task_id: str = "",
        action_id: str = "",
        step_id: str = "",
        correlation_id: str = "",
        component: str | None = None,
        severity: str = "INFO",
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> StructuredEvent | None:
        metadata = dict(metadata or {})
        if event_type in {"TASK_COMPLETED", "TASK_FAILED", "TASK_BLOCKED"} and task_id:
            timer = self._task_timers.get(task_id)
            if timer is not None:
                metadata.setdefault("duration_ms", round((time.monotonic() - timer.started) * 1000, 3))
                metadata.setdefault("segment", timer.segment)
        event = StructuredEvent(
            event_id=event_id or str(uuid.uuid4()),
            event_type=event_type,
            timestamp=_now(),
            task_id=task_id,
            action_id=action_id,
            step_id=step_id,
            correlation_id=correlation_id,
            component=component or self.component,
            severity=severity,
            metadata=metadata,
        )
        try:
            sequence = self.event_store.save(event)
            event.sequence = int(sequence)
            if event.event_type == "TASK_CREATED" and task_id:
                self._task_timers.setdefault(task_id, _TaskTimer(time.monotonic()))
            elif event.event_type == "TASK_RESUMED" and task_id:
                self._task_timers[task_id] = _TaskTimer(time.monotonic(), segment=1)
            return event
        except Exception as exc:  # observability must not change authorization/execution by default
            if isinstance(exc, sqlite3.IntegrityError):
                raise
            self.degraded_writes += 1
            self.last_storage_error = type(exc).__name__
            if self.fail_mode == "raise":
                raise ObservabilityStorageError("observability_write_failed") from exc
            return event

    def emit_task_end(self, task_id: str, event_type: str, *, metadata: dict[str, Any] | None = None) -> StructuredEvent | None:
        payload = dict(metadata or {})
        timer = self._task_timers.get(task_id)
        if timer is not None:
            payload.setdefault("duration_ms", round((time.monotonic() - timer.started) * 1000, 3))
            payload.setdefault("segment", timer.segment)
        event = self.emit(event_type, task_id=task_id, metadata=payload, severity="ERROR" if event_type in _FAILURE_EVENTS else "INFO")
        if event_type in {"TASK_COMPLETED", "TASK_FAILED", "TASK_BLOCKED"}:
            self._task_timers.pop(task_id, None)
        return event

    def get_task_timeline(self, task_id: str) -> list[dict[str, Any]]:
        return self.event_store.timeline(task_id)

    def get_recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.event_store.recent(limit)

    def get_task_metrics(self, task_id: str) -> dict[str, Any]:
        return self._metrics(self.get_task_timeline(task_id), task_id=task_id)

    def get_system_metrics(self) -> dict[str, Any]:
        return self._metrics(self.event_store.all(), task_id=None)

    def get_failures(self, task_id: str | None = None) -> list[dict[str, Any]]:
        events = self.get_task_timeline(task_id) if task_id else self.event_store.all()
        return [event for event in events if event["event_type"] in _FAILURE_EVENTS]

    def get_security_events(self, task_id: str | None = None) -> list[dict[str, Any]]:
        events = self.get_task_timeline(task_id) if task_id else self.event_store.all()
        return [event for event in events if event["event_type"].startswith("SECURITY_") or event["event_type"].startswith("APPROVAL_")]

    def get_tool_events(self, task_id: str | None = None) -> list[dict[str, Any]]:
        events = self.get_task_timeline(task_id) if task_id else self.event_store.all()
        return [event for event in events if event["event_type"].startswith("TOOL_")]

    @staticmethod
    def _metrics(events: Iterable[dict[str, Any]], *, task_id: str | None) -> dict[str, Any]:
        events = list(events)
        task_ids = {event.get("task_id") for event in events if event["event_type"] == "TASK_CREATED" and event.get("task_id")}
        action_ids = {
            (event.get("task_id", ""), event.get("action_id"))
            for event in events
            if event["event_type"] in {"ACTION_PROPOSED", "ACTION_STARTED", "ACTION_COMPLETED", "ACTION_FAILED", "ACTION_BLOCKED"}
            and event.get("action_id")
        }
        counts = {
            "task_count": len(task_ids),
            "completed_tasks": sum(event["event_type"] == "TASK_COMPLETED" for event in events),
            "failed_tasks": sum(event["event_type"] == "TASK_FAILED" for event in events),
            "blocked_tasks": sum(event["event_type"] == "TASK_BLOCKED" for event in events),
            "action_count": len(action_ids),
            "successful_actions": sum(event["event_type"] == "ACTION_COMPLETED" for event in events),
            "failed_actions": sum(event["event_type"] == "ACTION_FAILED" for event in events),
            "blocked_actions": sum(event["event_type"] == "ACTION_BLOCKED" for event in events),
            "verification_success": sum(event["event_type"] == "VERIFICATION_PASSED" for event in events),
            "verification_failure": sum(event["event_type"] == "VERIFICATION_FAILED" for event in events),
            "recovery_count": sum(event["event_type"] == "RECOVERY_STARTED" for event in events),
            "replan_count": sum(event["event_type"] == "REPLAN_STARTED" for event in events),
            "approval_requests": sum(event["event_type"] in {"APPROVAL_REQUESTED", "SKILL_APPROVAL_REQUESTED"} for event in events),
            "approval_approvals": sum(event["event_type"] == "APPROVAL_APPROVED" for event in events),
            "approval_denials": sum(event["event_type"] == "APPROVAL_DENIED" for event in events),
            "approval_expirations": sum(event["event_type"] == "APPROVAL_EXPIRED" for event in events),
            "tool_failures": sum(event["event_type"] == "TOOL_FAILED" for event in events),
            "execution_unknown": sum(event["event_type"] == "EXECUTION_UNKNOWN" for event in events),
        }
        durations = {
            "task_duration_ms": 0.0,
            "action_duration_ms": 0.0,
            "tool_duration_ms": 0.0,
            "planning_duration_ms": 0.0,
            "observation_duration_ms": 0.0,
            "verification_duration_ms": 0.0,
            "recovery_duration_ms": 0.0,
        }
        for event in events:
            metadata = event.get("metadata") or {}
            duration = metadata.get("duration_ms")
            if not isinstance(duration, (int, float)):
                continue
            kind = event["event_type"]
            if kind.startswith("TASK_"):
                durations["task_duration_ms"] += float(duration)
            if kind.startswith("ACTION_"):
                durations["action_duration_ms"] += float(duration)
            if kind.startswith("TOOL_"):
                durations["tool_duration_ms"] += float(duration)
            if kind == "PLAN_CREATED" or kind == "PLAN_UPDATED":
                durations["planning_duration_ms"] += float(duration)
            if kind == "OBSERVATION_COMPLETED":
                durations["observation_duration_ms"] += float(duration)
            if kind in {"VERIFICATION_PASSED", "VERIFICATION_FAILED"}:
                durations["verification_duration_ms"] += float(duration)
            if kind == "RECOVERY_COMPLETED":
                durations["recovery_duration_ms"] += float(duration)
        result: dict[str, Any] = {**counts, **{key: round(value, 3) for key, value in durations.items()}}
        if task_id is not None:
            result["task_id"] = task_id
        result["observability_event_count"] = len(events)
        return result


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "EVENT_TYPES",
    "EventStore",
    "ObservabilityError",
    "ObservabilityMonitor",
    "ObservabilityStorageError",
    "ObservabilityValidationError",
    "StructuredEvent",
    "safe_metadata",
]
