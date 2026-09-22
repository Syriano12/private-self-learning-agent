from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from private_agent.core.execution import StepResult


@dataclass
class Observation:
    task_id: str
    step_id: str
    tool_name: str
    execution_status: str
    observed_output: Any = None
    observed_error: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_step_result(cls, task_id: str, result: StepResult) -> "Observation":
        return cls(
            task_id=task_id,
            step_id=result.step_id,
            tool_name=result.tool_name,
            execution_status=result.status,
            observed_output=result.output,
            observed_error=result.error.to_dict() if result.error else None,
            metadata=dict(result.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepOutcome:
    task_id: str
    step_id: str
    tool_name: str
    execution_status: str
    verification_status: str
    observation: Observation
    verification: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "execution_status": self.execution_status,
            "verification_status": self.verification_status,
            "observation": self.observation.to_dict(),
            "verification": self.verification,
        }
