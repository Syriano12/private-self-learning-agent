from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from private_agent.core.diagnosis import FailureDiagnosis
from private_agent.core.planner import PlanStep


RECOVERY_STRATEGIES = {
    "RETRY",
    "RETRY_WITH_MODIFIED_INPUT",
    "CHANGE_TOOL",
    "CHANGE_PARAMETERS",
    "SKIP_OPTIONAL_STEP",
    "REPLAN",
    "ABORT",
}


@dataclass
class RecoveryDecision:
    strategy: str
    reason: str
    step_id: str
    original_tool: str
    replacement_tool: str = ""
    modified_input: dict[str, Any] = field(default_factory=dict)
    terminal_status: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in RECOVERY_STRATEGIES:
            raise ValueError(f"invalid_recovery_strategy:{self.strategy}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "reason": self.reason,
            "step_id": self.step_id,
            "original_tool": self.original_tool,
            "replacement_tool": self.replacement_tool,
            "modified_input": self.modified_input,
            "terminal_status": self.terminal_status,
            "details": self.details,
        }


@dataclass
class RecoveryAttempt:
    attempt_number: int
    step_id: str
    strategy: str
    tool_name: str
    input_fingerprint: str
    diagnosis: dict[str, Any]
    outcome: str = "planned"
    verification_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "step_id": self.step_id,
            "strategy": self.strategy,
            "tool_name": self.tool_name,
            "input_fingerprint": self.input_fingerprint,
            "diagnosis": self.diagnosis,
            "outcome": self.outcome,
            "verification_status": self.verification_status,
        }


class RecoveryManager:
    """Select finite recovery decisions and reject exact repeats."""

    def __init__(self, *, max_recovery_attempts: int = 3, max_attempts_per_step: int = 2) -> None:
        self.max_recovery_attempts = max(0, max_recovery_attempts)
        self.max_attempts_per_step = max(1, max_attempts_per_step)
        self.attempts: list[RecoveryAttempt] = []

    def choose(
        self,
        step: PlanStep,
        diagnosis: FailureDiagnosis,
        *,
        available_tools: list[dict[str, Any]],
        alternative_plan: PlanStep | None = None,
        allow_replan: bool = False,
    ) -> RecoveryDecision:
        if len(self.attempts) >= self.max_recovery_attempts:
            return self._abort(step, "Maximum recovery attempts reached", "ABORTED")
        step_attempts = [attempt for attempt in self.attempts if attempt.step_id == step.id]
        if len(step_attempts) >= self.max_attempts_per_step:
            return self._abort(step, "Maximum attempts for this step reached", "FAILED")
        if not diagnosis.recoverable:
            terminal = "BLOCKED" if diagnosis.failure_type == "PERMISSION_BLOCKED" else "FAILED"
            return self._abort(step, "Failure is not recoverable", terminal)

        preferred = diagnosis.suggested_recovery
        if preferred == "CHANGE_TOOL":
            replacement = self._find_alternative(step.tool, available_tools, step_attempts)
            if replacement:
                return RecoveryDecision(
                    "CHANGE_TOOL",
                    "Use a different registered tool after diagnosing the original failure",
                    step.id,
                    step.tool,
                    replacement_tool=replacement,
                    details={"failure_type": diagnosis.failure_type},
                )
            if alternative_plan and alternative_plan.tool != step.tool:
                return RecoveryDecision(
                    "REPLAN",
                    "Use a validated alternative plan because no direct replacement was selected",
                    step.id,
                    step.tool,
                    replacement_tool=alternative_plan.tool,
                    details={"failure_type": diagnosis.failure_type},
                )
            if allow_replan:
                return RecoveryDecision(
                    "REPLAN",
                    "Ask the validated replanner for a replacement plan after no direct alternative was selected",
                    step.id,
                    step.tool,
                    details={"failure_type": diagnosis.failure_type},
                )

        if preferred in {"CHANGE_PARAMETERS", "RETRY_WITH_MODIFIED_INPUT"}:
            modified = self._modify_input(step.input, diagnosis)
            if modified != step.input and not self._was_attempted(step, "RETRY_WITH_MODIFIED_INPUT", modified):
                return RecoveryDecision(
                    "RETRY_WITH_MODIFIED_INPUT",
                    "Retry with a deterministic input modification",
                    step.id,
                    step.tool,
                    modified_input=modified,
                    details={"failure_type": diagnosis.failure_type},
                )

        if preferred == "RETRY" and not self._was_attempted(step, "RETRY", step.input):
            return RecoveryDecision(
                "RETRY",
                "Retry once because the diagnosed failure is transient and the fingerprint is new",
                step.id,
                step.tool,
                details={"failure_type": diagnosis.failure_type},
            )

        if alternative_plan and alternative_plan.tool != step.tool:
            return RecoveryDecision(
                "REPLAN",
                "Use the alternative validated plan instead of repeating a failed strategy",
                step.id,
                step.tool,
                replacement_tool=alternative_plan.tool,
                details={"failure_type": diagnosis.failure_type},
            )
        return self._abort(step, "No unused recovery strategy remains", "FAILED")

    def record(self, decision: RecoveryDecision, diagnosis: FailureDiagnosis, *, tool_name: str, inputs: dict[str, Any]) -> RecoveryAttempt:
        attempt = RecoveryAttempt(
            attempt_number=len(self.attempts) + 1,
            step_id=decision.step_id,
            strategy=decision.strategy,
            tool_name=tool_name,
            input_fingerprint=input_fingerprint(inputs),
            diagnosis=diagnosis.to_dict(),
        )
        self.attempts.append(attempt)
        return attempt

    def update_last(self, *, outcome: str, verification_status: str = "") -> None:
        if self.attempts:
            self.attempts[-1].outcome = outcome
            self.attempts[-1].verification_status = verification_status

    def _was_attempted(self, step: PlanStep, strategy: str, inputs: dict[str, Any]) -> bool:
        fingerprint = input_fingerprint(inputs)
        return any(
            attempt.step_id == step.id
            and attempt.strategy == strategy
            and attempt.tool_name == step.tool
            and attempt.input_fingerprint == fingerprint
            for attempt in self.attempts
        )

    @staticmethod
    def _modify_input(inputs: dict[str, Any], diagnosis: FailureDiagnosis) -> dict[str, Any]:
        modified = dict(inputs)
        if isinstance(modified.get("limit"), int) and modified["limit"] > 1:
            modified["limit"] = max(1, modified["limit"] // 2)
        elif diagnosis.failure_type in {"INSUFFICIENT_RESULT", "VERIFICATION_FAILED"} and "limit" in modified:
            modified["limit"] = min(10, int(modified["limit"]) + 1)
        return modified

    @staticmethod
    def _find_alternative(
        original_tool: str,
        available_tools: list[dict[str, Any]],
        attempts: list[RecoveryAttempt],
    ) -> str:
        attempted_tools = {attempt.tool_name for attempt in attempts}
        for metadata in available_tools:
            name = metadata.get("name", "")
            if name and name != original_tool and name not in attempted_tools:
                return name
        return ""

    @staticmethod
    def _abort(step: PlanStep, reason: str, terminal_status: str) -> RecoveryDecision:
        return RecoveryDecision("ABORT", reason, step.id, step.tool, terminal_status=terminal_status)


def input_fingerprint(inputs: dict[str, Any]) -> str:
    serialized = json.dumps(inputs, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
