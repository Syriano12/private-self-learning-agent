from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from private_agent.core.execution import StepResult
from private_agent.core.observation import Observation
from private_agent.core.verification import VerificationResult


FAILURE_TYPES = {
    "TOOL_ERROR",
    "INVALID_INPUT",
    "MISSING_DEPENDENCY",
    "PERMISSION_BLOCKED",
    "TIMEOUT",
    "RATE_LIMITED",
    "NETWORK_ERROR",
    "VERIFICATION_FAILED",
    "INSUFFICIENT_RESULT",
    "INVALID_OUTPUT",
    "UNKNOWN_FAILURE",
}


@dataclass
class FailureDiagnosis:
    failure_type: str
    reason: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    failed_step: str = ""
    tool_name: str = ""
    recoverable: bool = True
    suggested_recovery: str = "ABORT"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.failure_type not in FAILURE_TYPES:
            raise ValueError(f"invalid_failure_type:{self.failure_type}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "reason": self.reason,
            "evidence": self.evidence,
            "failed_step": self.failed_step,
            "tool_name": self.tool_name,
            "recoverable": self.recoverable,
            "suggested_recovery": self.suggested_recovery,
            "details": self.details,
        }


class FailureDiagnoser:
    """Deterministic diagnosis; no LLM is needed to classify known execution failures."""

    def diagnose(
        self,
        *,
        step_result: StepResult,
        observation: Observation,
        verification: VerificationResult,
    ) -> FailureDiagnosis | None:
        if step_result.status == "success" and verification.status == "VERIFIED":
            return None

        if step_result.status != "success":
            error = step_result.error
            code = error.code if error else ""
            details = error.details if error else {}
            failure_type, recoverable, strategy = self._classify_execution(code, details)
            return FailureDiagnosis(
                failure_type,
                error.message if error else "Execution failed without a structured error",
                evidence=[{"type": "execution_error", "value": error.to_dict() if error else None}],
                failed_step=observation.step_id,
                tool_name=observation.tool_name,
                recoverable=recoverable,
                suggested_recovery=strategy,
                details={"execution_status": step_result.status, "error_code": code, "error_details": details},
            )

        if verification.status == "BLOCKED":
            blocked_code = verification.details.get("code") if isinstance(verification.details, dict) else ""
            failure_type = "PERMISSION_BLOCKED" if blocked_code in {"permission_not_approved", "permission_denied"} else "UNKNOWN_FAILURE"
            return FailureDiagnosis(
                failure_type,
                verification.reason,
                evidence=verification.evidence,
                failed_step=observation.step_id,
                tool_name=observation.tool_name,
                recoverable=False,
                suggested_recovery="ABORT",
                details={"verification": verification.to_dict()},
            )
        if verification.status == "FAILED":
            return FailureDiagnosis(
                "VERIFICATION_FAILED",
                verification.reason,
                evidence=verification.evidence,
                failed_step=observation.step_id,
                tool_name=observation.tool_name,
                recoverable=True,
                suggested_recovery="CHANGE_TOOL",
                details={"verification": verification.to_dict()},
            )
        if verification.status == "INSUFFICIENT":
            return FailureDiagnosis(
                "INSUFFICIENT_RESULT",
                verification.reason,
                evidence=verification.evidence,
                failed_step=observation.step_id,
                tool_name=observation.tool_name,
                recoverable=True,
                suggested_recovery="CHANGE_TOOL",
                details={"verification": verification.to_dict()},
            )

        return FailureDiagnosis(
            "UNKNOWN_FAILURE",
            "Unexpected unsuccessful execution/verification combination",
            evidence=[{"type": "step_result", "value": step_result.to_dict()}],
            failed_step=observation.step_id,
            tool_name=observation.tool_name,
            recoverable=False,
            suggested_recovery="ABORT",
            details={"verification": verification.to_dict()},
        )

    @staticmethod
    def _classify_execution(code: str, details: dict[str, Any]) -> tuple[str, bool, str]:
        normalized = code.lower()
        if normalized in {"permission_not_approved", "permission_denied"}:
            return "PERMISSION_BLOCKED", False, "ABORT"
        if normalized in {"missing_dependency", "dependency_failed"}:
            return "MISSING_DEPENDENCY", False, "ABORT"
        if normalized in {"invalid_input"}:
            return "INVALID_INPUT", True, "CHANGE_PARAMETERS"
        if normalized in {"malformed_tool_result", "invalid_output"}:
            return "INVALID_OUTPUT", True, "CHANGE_TOOL"
        if normalized in {"timeout", "timed_out"}:
            return "TIMEOUT", True, "RETRY"
        if normalized in {"rate_limited", "http_429"}:
            return "RATE_LIMITED", True, "RETRY"
        if normalized in {"network_error", "transport_error"}:
            return "NETWORK_ERROR", True, "RETRY"
        if normalized in {"unknown_tool", "tool_exception", "tool_failed", "no_sources"}:
            return "TOOL_ERROR", True, "CHANGE_TOOL"
        if details.get("exception_type") in {"TimeoutError", "ReadTimeout", "ConnectTimeout"}:
            return "TIMEOUT", True, "RETRY"
        return "UNKNOWN_FAILURE", True, "RETRY"
