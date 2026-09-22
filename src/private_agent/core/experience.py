from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = ("api_key", "apikey", "password", "secret", "credential", "authorization", "cookie", "token", "access_token", "refresh_token")
_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[^\s,;]+"),
)


def sanitize_value(value: Any, *, key: str = "") -> Any:
    lowered = key.lower()
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        return _REDACTED
    if isinstance(value, dict):
        return {str(name): sanitize_value(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [sanitize_value(item, key=key) for item in value]
    if isinstance(value, str):
        sanitized = value
        for pattern in _SECRET_PATTERNS:
            sanitized = pattern.sub(_REDACTED, sanitized)
        return sanitized
    return value


@dataclass
class ExperienceAction:
    action_type: str
    step_id: str
    tool_name: str
    input_fingerprint: str
    execution_status: str = ""
    observation: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] | None = None
    recovery: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return sanitize_value(asdict(self))


@dataclass
class ExperienceRecord:
    task_id: str
    goal: str
    context: dict[str, Any]
    initial_plan: dict[str, Any]
    actions: list[ExperienceAction] = field(default_factory=list)
    final_outcome: str = ""
    identity: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_action(self, action: ExperienceAction) -> None:
        self.actions.append(action)

    def build_identity(self) -> dict[str, Any]:
        tools = []
        failure_types = []
        recovery_strategies = []
        for action in self.actions:
            if action.tool_name and action.tool_name not in tools:
                tools.append(action.tool_name)
            if action.diagnosis and action.diagnosis.get("failure_type") not in failure_types:
                failure_types.append(action.diagnosis["failure_type"])
            if action.recovery and action.recovery.get("strategy") not in recovery_strategies:
                recovery_strategies.append(action.recovery["strategy"])
        task_type = self.context.get("task_type", "general") if isinstance(self.context, dict) else "general"
        capabilities = self.context.get("required_capabilities", []) if isinstance(self.context, dict) else []
        constraints = self.context.get("constraints", {}) if isinstance(self.context, dict) else {}
        self.identity = {
            "task_type": task_type,
            "goal": self.goal,
            "required_capabilities": list(capabilities) if isinstance(capabilities, list) else [],
            "tools": tools,
            "constraints": constraints if isinstance(constraints, dict) else {},
            "failure_types": failure_types,
            "recovery_strategies": recovery_strategies,
            "outcome": self.final_outcome,
            "verification_status": "VERIFIED" if self.final_outcome == "COMPLETED" else self.final_outcome,
        }
        return self.identity

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actions"] = [action.to_dict() for action in self.actions]
        payload["identity"] = self.identity or self.build_identity()
        return sanitize_value(payload)
