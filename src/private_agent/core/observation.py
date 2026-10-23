from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from private_agent.core.execution import StepResult
from private_agent.core.experience import sanitize_value


MAX_OBSERVATION_BYTES = 128 * 1024


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
        payload = sanitize_value(asdict(self))
        try:
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("observation_not_json_safe") from exc
        if len(encoded.encode("utf-8")) > MAX_OBSERVATION_BYTES:
            raise ValueError("observation_too_large")
        return payload


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
