from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


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
        return asdict(self)


@dataclass
class ExperienceRecord:
    task_id: str
    goal: str
    context: dict[str, Any]
    initial_plan: dict[str, Any]
    actions: list[ExperienceAction] = field(default_factory=list)
    final_outcome: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_action(self, action: ExperienceAction) -> None:
        self.actions.append(action)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actions"] = [action.to_dict() for action in self.actions]
        return payload
