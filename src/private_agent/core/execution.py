from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.tools.contracts import ToolError, ToolResult, contract_errors, validate_schema_value
from private_agent.tools.research import ToolRegistry


@dataclass
class ExecutionContext:
    task_id: str = ""
    previous_results: dict[str, "StepResult"] = field(default_factory=dict)
    approved_permissions: set[str] = field(default_factory=lambda: {"public_read"})
    limits: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    step_id: str
    tool_name: str
    status: str
    output: Any = None
    error: ToolError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        step_id: str,
        tool_name: str,
        output: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "StepResult":
        return cls(step_id, tool_name, "success", output=output, metadata=metadata or {})

    @classmethod
    def failed(
        cls,
        step_id: str,
        tool_name: str,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "StepResult":
        return cls(
            step_id,
            tool_name,
            "failed",
            error=ToolError(code, message, details or {}),
            metadata=metadata or {},
        )

    @classmethod
    def blocked(
        cls,
        step_id: str,
        tool_name: str,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "StepResult":
        return cls(
            step_id,
            tool_name,
            "blocked",
            error=ToolError(code, message, details or {}),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "output": self.output,
            "error": self.error.to_dict() if self.error else None,
            "metadata": self.metadata,
        }


class GenericExecutor:
    """Execute any registered ToolContract without knowing concrete tool names."""

    def __init__(self, tools: ToolRegistry) -> None:
        self.tools = tools

    def execute_plan(
        self,
        plan: TaskPlan,
        context: ExecutionContext | None = None,
    ) -> list[StepResult]:
        execution_context = context or ExecutionContext()
        results: list[StepResult] = []
        for step in plan.steps:
            result = self.execute_step(step, execution_context)
            results.append(result)
            execution_context.previous_results[step.id] = result
        return results

    def execute_step(self, step: PlanStep, context: ExecutionContext | None = None) -> StepResult:
        execution_context = context or ExecutionContext()
        started = monotonic()
        base_metadata = {"task_id": execution_context.task_id, "tool_name": step.tool}

        try:
            tool = self.tools.get(step.tool)
        except KeyError:
            return StepResult.failed(
                step.id,
                step.tool,
                "unknown_tool",
                f"Tool is not registered: {step.tool}",
                metadata=self._timed(base_metadata, started),
            )

        contract_problems = contract_errors(tool)
        if contract_problems:
            return StepResult.failed(
                step.id,
                step.tool,
                "invalid_tool_contract",
                "Tool does not satisfy the required contract",
                details={"errors": contract_problems},
                metadata=self._timed(base_metadata, started),
            )

        dependencies = self._check_dependencies(step, execution_context)
        if dependencies is not None:
            return StepResult.blocked(
                step.id,
                step.tool,
                dependencies[0],
                dependencies[1],
                details=dependencies[2],
                metadata=self._timed(base_metadata, started),
            )

        permission = getattr(tool, "permission_level", "unknown")
        if permission not in execution_context.approved_permissions:
            return StepResult.blocked(
                step.id,
                step.tool,
                "permission_not_approved",
                f"Permission is not approved: {permission}",
                details={"required_permission": permission},
                metadata=self._timed(base_metadata, started),
            )

        input_errors = validate_schema_value(step.input, tool.input_schema(), path="input")
        if input_errors:
            return StepResult.failed(
                step.id,
                step.tool,
                "invalid_input",
                "Tool input failed schema validation",
                details={"errors": input_errors},
                metadata=self._timed(base_metadata, started),
            )

        try:
            tool_result = tool.execute(step.input, execution_context)
        except Exception as exc:
            return StepResult.failed(
                step.id,
                step.tool,
                "tool_exception",
                f"Tool execution failed: {type(exc).__name__}",
                details={"exception_type": type(exc).__name__},
                metadata=self._timed(base_metadata, started),
            )

        if not isinstance(tool_result, ToolResult) or tool_result.status not in {"success", "failed", "blocked"}:
            return StepResult.failed(
                step.id,
                step.tool,
                "malformed_tool_result",
                "Tool did not return a valid ToolResult",
                metadata=self._timed(base_metadata, started),
            )

        if tool_result.status == "success":
            output_errors = validate_schema_value(tool_result.output, tool.output_schema(), path="output")
            if output_errors:
                return StepResult.failed(
                    step.id,
                    step.tool,
                    "malformed_tool_result",
                    "Successful tool output failed its output schema",
                    details={"errors": output_errors},
                    metadata=self._timed(base_metadata, started),
                )
            return StepResult.success(
                step.id,
                step.tool,
                tool_result.output,
                metadata={**tool_result.metadata, **self._timed(base_metadata, started)},
            )

        return StepResult(
            step_id=step.id,
            tool_name=step.tool,
            status=tool_result.status,
            output=tool_result.output,
            error=tool_result.error
            or ToolError("tool_failed", "Tool returned a failure without a structured error"),
            metadata={**tool_result.metadata, **self._timed(base_metadata, started)},
        )

    @staticmethod
    def _check_dependencies(
        step: PlanStep,
        context: ExecutionContext,
    ) -> tuple[str, str, dict[str, Any]] | None:
        for dependency in step.depends_on:
            if dependency not in context.previous_results:
                return (
                    "missing_dependency",
                    f"Dependency has not executed: {dependency}",
                    {"dependency": dependency},
                )
            dependency_result = context.previous_results[dependency]
            if dependency_result.status != "success":
                return (
                    "dependency_failed",
                    f"Dependency did not succeed: {dependency}",
                    {"dependency": dependency, "dependency_status": dependency_result.status},
                )
        return None

    @staticmethod
    def _timed(metadata: dict[str, Any], started: float) -> dict[str, Any]:
        return {**metadata, "duration_ms": round((monotonic() - started) * 1000, 3)}
