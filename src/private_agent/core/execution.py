from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from private_agent.core.experience import sanitize_value
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.core.planning_provenance import UNKNOWN, normalize_plan_version, normalize_planning_mode
from private_agent.storage import QueueLeaseLost
from private_agent.tools.contracts import ToolError, ToolResult, contract_errors, validate_schema_value
from private_agent.tools.research import ToolRegistry


_SUPPORTED_SCHEMA_TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})
_SUPPORTED_SCHEMA_KEYWORDS = frozenset({
    "type", "required", "properties", "items", "additionalProperties", "minimum", "maximum",
    "title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly",
    "$comment", "$schema", "$id",
})


def _is_supported_schema_shape(schema: Any, *, depth: int = 0) -> bool:
    """Reject malformed or unsupported schema structures before validating bound data."""
    if depth > 32 or not isinstance(schema, dict):
        return False
    try:
        if any(not isinstance(keyword, str) or keyword not in _SUPPORTED_SCHEMA_KEYWORDS for keyword in schema):
            return False
        if "type" in schema:
            schema_type = schema["type"]
            if not isinstance(schema_type, str) or schema_type not in _SUPPORTED_SCHEMA_TYPES:
                return False
        else:
            schema_type = None
        if any(keyword in schema for keyword in ("required", "properties", "additionalProperties")) and schema_type not in {None, "object"}:
            return False
        if "items" in schema and schema_type not in {None, "array"}:
            return False
        if any(keyword in schema for keyword in ("minimum", "maximum")) and schema_type not in {None, "integer", "number"}:
            return False
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(name, str) or not name for name in required):
            return False
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return False
        if any(
            not isinstance(name, str)
            or not name
            or not _is_supported_schema_shape(property_schema, depth=depth + 1)
            for name, property_schema in properties.items()
        ):
            return False
        if "items" in schema and not _is_supported_schema_shape(schema["items"], depth=depth + 1):
            return False
        if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
            return False
        for bound in ("minimum", "maximum"):
            if bound in schema:
                value = schema[bound]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return False
        if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
            return False
        return True
    except Exception:
        return False


_SCHEMA_UNSET = object()


class _ExecutionOutcomeUnknown(RuntimeError):
    """Signal that a tool operation began but its returned value could not be validated."""


@dataclass
class ExecutionContext:
    task_id: str = ""
    previous_results: dict[str, "StepResult"] = field(default_factory=dict)
    approved_permissions: set[str] = field(default_factory=lambda: {"public_read"})
    limits: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    security_controller: Any | None = None
    approval_ids: dict[str, str] = field(default_factory=dict)
    observability: Any | None = None
    verified_result_ids: set[str] = field(default_factory=set)


@dataclass
class StepResult:
    step_id: str
    tool_name: str
    status: str
    output: Any = None
    error: ToolError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    planning_mode: str = UNKNOWN
    plan_version: int = 0

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
        payload = sanitize_value({
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "output": self.output,
            "error": self.error.to_dict() if self.error else None,
            "metadata": self.metadata,
            "planning_mode": normalize_planning_mode(self.planning_mode),
            "plan_version": normalize_plan_version(self.plan_version),
        })
        try:
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("step_result_not_json_safe") from exc
        if len(encoded.encode("utf-8")) > 128 * 1024:
            raise ValueError("step_result_too_large")
        return payload


class GenericExecutor:
    """Execute any registered ToolContract without knowing concrete tool names."""

    def __init__(
        self,
        tools: ToolRegistry,
        security_controller: Any | None = None,
        *,
        lease_store: Any | None = None,
    ) -> None:
        self.tools = tools
        self.security_controller = security_controller
        self.lease_store = lease_store

    def execute_plan(
        self,
        plan: TaskPlan,
        context: ExecutionContext | None = None,
    ) -> list[StepResult]:
        """Run a low-level plan; dependency checks here do not verify outcomes.

        Dependencies are gated only on successful ``StepResult`` values. Task
        flows that require verified dependencies must be driven by
        ``Orchestrator``, which executes and verifies each step before exposing
        its result to later steps.
        """
        execution_context = context or ExecutionContext(security_controller=self.security_controller)
        ordered_steps, dependency_error = self._dependency_order(plan.steps)
        if dependency_error is not None:
            return [
                StepResult.blocked(
                    step.id,
                    step.tool,
                    "invalid_plan_dependencies",
                    dependency_error,
                )
                for step in plan.steps
            ]
        results: list[StepResult] = []
        for step in ordered_steps:
            result = self.execute_step(step, execution_context)
            results.append(result)
            execution_context.previous_results[step.id] = result
        return results

    @staticmethod
    def _dependency_order(steps: list[PlanStep]) -> tuple[list[PlanStep], str | None]:
        by_id: dict[str, PlanStep] = {}
        for step in steps:
            if not isinstance(step.id, str) or not step.id or not isinstance(step.tool, str) or not step.tool:
                return [], "step_identity_invalid"
            if not isinstance(step.depends_on, list) or any(not isinstance(item, str) or not item for item in step.depends_on):
                return [], f"step_dependencies_invalid:{step.id}"
            if step.id in by_id:
                return [], f"duplicate_step_id:{step.id}"
            by_id[step.id] = step
        for step in steps:
            for dependency in step.depends_on:
                if dependency not in by_id:
                    return [], f"missing_step_dependency:{dependency}"

        remaining = dict(by_id)
        ordered: list[PlanStep] = []
        while remaining:
            ready = [
                step
                for step in remaining.values()
                if all(dependency in {item.id for item in ordered} for dependency in step.depends_on)
            ]
            if not ready:
                return [], "cyclic_step_dependencies"
            for step in sorted(ready, key=lambda item: steps.index(item)):
                ordered.append(step)
                remaining.pop(step.id)
        return ordered, None

    def execute_step(self, step: PlanStep, context: ExecutionContext | None = None) -> StepResult:
        execution_context = context or ExecutionContext(security_controller=self.security_controller)
        started = monotonic()
        base_metadata = {"task_id": execution_context.task_id, "tool_name": step.tool}
        observer = execution_context.observability
        if observer is not None:
            observer.emit(
                "TOOL_STARTED",
                task_id=execution_context.task_id,
                action_id=step.id,
                step_id=step.id,
                component="executor",
                metadata={"tool_name": step.tool},
            )

        def finish(result: StepResult) -> StepResult:
            if observer is not None:
                observer.emit(
                    "TOOL_COMPLETED" if result.status == "success" else "TOOL_FAILED",
                    task_id=execution_context.task_id,
                    action_id=step.id,
                    step_id=step.id,
                    component="executor",
                    severity="INFO" if result.status == "success" else "ERROR",
                    metadata={
                        "tool_name": step.tool,
                        "status": result.status,
                        "duration_ms": round((monotonic() - started) * 1000, 3),
                        "error_category": result.error.code if result.error else "",
                    },
                )
            return result

        trusted_controller = self.security_controller
        if trusted_controller is None:
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                "security_not_configured",
                "Execution requires an initialized SecurityController owned by the executor",
                metadata=self._timed(base_metadata, started),
            ))
        if trusted_controller is not None and execution_context.security_controller not in (None, trusted_controller):
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                "security_controller_mismatch",
                "ExecutionContext cannot replace the executor's trusted SecurityController",
                metadata=self._timed(base_metadata, started),
            ))

        try:
            tool = self.tools.get(step.tool)
        except KeyError:
            return finish(StepResult.failed(
                step.id,
                step.tool,
                "unknown_tool",
                f"Tool is not registered: {step.tool}",
                metadata=self._timed(base_metadata, started),
            ))
        except Exception:
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                "tool_registry_unavailable",
                "Tool registry could not resolve the requested tool",
                details={"step_id": step.id},
                metadata=self._timed(base_metadata, started),
            ))

        contract_problems = contract_errors(tool)
        if contract_problems:
            return finish(StepResult.failed(
                step.id,
                step.tool,
                "invalid_tool_contract",
                "Tool does not satisfy the required contract",
                details={"errors": contract_problems},
                metadata=self._timed(base_metadata, started),
            ))

        dependencies = self._check_dependencies(step, execution_context)
        if dependencies is not None:
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                dependencies[0],
                dependencies[1],
                details=dependencies[2],
                metadata=self._timed(base_metadata, started),
            ))

        try:
            tool_input_schema = tool.input_schema()
        except Exception:
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                "tool_input_schema_invalid",
                "Tool input schema could not be read before execution",
                details={"step_id": step.id},
                metadata=self._timed(base_metadata, started),
            ))
        if not _is_supported_schema_shape(tool_input_schema):
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                "tool_input_schema_invalid",
                "Tool input schema is malformed or unsupported",
                details={"step_id": step.id},
                metadata=self._timed(base_metadata, started),
            ))
        try:
            tool_output_schema = tool.output_schema()
        except Exception:
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                "tool_output_schema_invalid",
                "Tool output schema could not be read before execution",
                details={"step_id": step.id},
                metadata=self._timed(base_metadata, started),
            ))
        if not _is_supported_schema_shape(tool_output_schema):
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                "tool_output_schema_invalid",
                "Tool output schema is malformed or unsupported",
                details={"step_id": step.id},
                metadata=self._timed(base_metadata, started),
            ))

        resolved_input, binding_error = self._resolve_input_bindings(
            step,
            tool,
            execution_context,
            target_input_schema=tool_input_schema,
        )
        if binding_error is not None:
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                binding_error[0],
                "A declared input binding could not be resolved from a verified dependency",
                details=binding_error[1],
                metadata=self._timed(base_metadata, started),
            ))

        security_action = None
        security_action = trusted_controller.action_for_tool(
            execution_context.task_id,
            step.id,
            tool,
            resolved_input,
            goal=str(execution_context.metadata.get("goal", "")),
        )
        security_decision = trusted_controller.authorize(
            security_action,
            approval_id=execution_context.approval_ids.get(step.id, "")
            or execution_context.approval_ids.get(step.tool, ""),
            boundary=True,
        )
        if not security_decision.allowed:
            security_code = (
                "permission_not_approved"
                if security_decision.decision == "REQUIRE_APPROVAL"
                else "security_policy_blocked"
            )
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                security_code,
                f"Security policy blocked execution: {security_decision.decision}",
                details=security_decision.to_dict(),
                metadata=self._timed({**base_metadata, "security_decision": security_decision.to_dict()}, started),
            ))

        permission = getattr(tool, "permission_level", "unknown")
        if permission not in execution_context.approved_permissions:
            trusted_controller.record_blocked(
                security_action,
                reason=f"Existing permission set does not include {permission}",
                authorization=security_decision,
                metadata={"required_permission": permission},
            )
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                "permission_not_approved",
                f"Permission is not approved: {permission}",
                details={"required_permission": permission},
                metadata=self._timed(base_metadata, started),
            ))

        try:
            input_errors = validate_schema_value(resolved_input, tool_input_schema, path="input")
        except Exception:
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                "tool_input_schema_invalid",
                "Tool input could not be validated before execution",
                details={"step_id": step.id},
                metadata=self._timed(base_metadata, started),
            ))
        if input_errors:
            return finish(StepResult.failed(
                step.id,
                step.tool,
                "invalid_input",
                "Tool input failed schema validation",
                details={"errors": input_errors},
                metadata=self._timed(base_metadata, started),
            ))

        output_validation_errors: list[str] | None = None
        output_validation_exception: str | None = None

        def validate_tool_result(value: Any) -> tuple[bool, dict[str, Any]]:
            nonlocal output_validation_errors, output_validation_exception
            if not isinstance(value, ToolResult) or value.status not in {"success", "failed", "blocked"}:
                return False, {"error": "malformed_tool_result"}
            if value.status == "success":
                try:
                    output_validation_errors = validate_schema_value(value.output, tool_output_schema, path="output")
                except Exception as exc:
                    output_validation_exception = type(exc).__name__
                    return False, {"error": "result_validation_exception", "exception_type": output_validation_exception}
                if output_validation_errors:
                    return False, {"error": "malformed_tool_result", "details": output_validation_errors}
            return value.status == "success", {
                "tool_status": value.status,
                "no_side_effects": value.no_side_effects,
            }

        operation_binding = None
        trusted_execute = getattr(tool, "_execute_trusted", None)

        def operation() -> Any:
            if callable(trusted_execute):
                return trusted_execute(
                    resolved_input,
                    execution_context,
                    operation_binding=operation_binding,
                    authorization=security_decision,
                    operation=operation,
                )
            return tool.execute(resolved_input, execution_context)

        operation_binding = trusted_controller._bind_operation(
            security_action,
            operation,
            kind="tool",
        )
        execution = trusted_controller.execute_authorized(
            security_action,
            operation,
            authorization=security_decision,
            result_validator=validate_tool_result,
            operation_binding=operation_binding,
            pre_execution_check=(
                lambda: self.lease_store.assert_active_lease(execution_context.task_id)
                if self.lease_store is not None and execution_context.task_id
                else None
            ),
        )
        if execution.blocked:
            code = "queue_lease_lost" if isinstance(execution.exception, QueueLeaseLost) else "execution_boundary_blocked"
            return finish(StepResult.blocked(
                step.id,
                step.tool,
                code,
                "Execution lease was not valid at the side-effect boundary",
                metadata=self._timed(base_metadata, started),
            ))
        if execution.exception is not None:
            # The operation callback may have produced an external effect before raising.
            # Without trusted evidence that it did not, the worker must not retry it.
            raise _ExecutionOutcomeUnknown("Tool operation raised after execution may have started")
        if output_validation_exception is not None:
            # The operation already returned through the authorization boundary.
            # Let the worker persist EXECUTION_UNKNOWN rather than retrying as a normal failure.
            raise _ExecutionOutcomeUnknown("Tool result validation failed after execution may have started")

        tool_result = execution.result
        if not isinstance(tool_result, ToolResult) or tool_result.status not in {"success", "failed", "blocked"}:
            raise _ExecutionOutcomeUnknown("Tool returned an unvalidated result after execution may have started")

        if tool_result.status == "success":
            if output_validation_errors is None:
                raise _ExecutionOutcomeUnknown("Tool result was not validated after execution may have started")
            output_errors = output_validation_errors
            if output_errors:
                raise _ExecutionOutcomeUnknown("Tool output did not satisfy its declared schema after execution may have started")
            return finish(StepResult.success(
                step.id,
                step.tool,
                tool_result.output,
                metadata={**tool_result.metadata, **self._timed(base_metadata, started)},
            ))

        if tool_result.no_side_effects is not True:
            # A failed/blocked return without the explicit no-effect assertion does
            # not prove that no partial side effect occurred.
            raise _ExecutionOutcomeUnknown("Tool reported an unsuccessful result after execution may have started")
        return finish(StepResult(
            step_id=step.id,
            tool_name=step.tool,
            status=tool_result.status,
            output=tool_result.output,
            error=tool_result.error
            or ToolError("tool_failed", "Tool returned a failure without a structured error"),
            metadata={**tool_result.metadata, **self._timed(base_metadata, started)},
        ))

    def _resolve_input_bindings(
        self,
        step: PlanStep,
        tool: Any,
        context: ExecutionContext,
        *,
        target_input_schema: Any = _SCHEMA_UNSET,
    ) -> tuple[dict[str, Any], tuple[str, dict[str, Any]] | None]:
        if not isinstance(step.input, dict):
            return {}, ("input_binding_input_invalid", {"step_id": step.id})
        resolved = dict(step.input)
        bindings = step.input_bindings
        if not isinstance(bindings, dict):
            return resolved, ("input_bindings_invalid", {"step_id": step.id})
        if not bindings:
            return resolved, None
        if target_input_schema is _SCHEMA_UNSET:
            try:
                target_input_schema = tool.input_schema()
            except Exception:
                return resolved, ("input_binding_target_schema_invalid", {"step_id": step.id})
        try:
            if not _is_supported_schema_shape(target_input_schema):
                return resolved, ("input_binding_target_schema_invalid", {"step_id": step.id})
            target_properties = target_input_schema.get("properties", {})
        except Exception:
            return resolved, ("input_binding_target_schema_invalid", {"step_id": step.id})
        if not isinstance(target_properties, dict):
            return resolved, ("input_binding_target_schema_invalid", {"step_id": step.id})
        for target, binding in bindings.items():
            if (
                not isinstance(target, str)
                or target in resolved
                or not isinstance(binding, dict)
                or set(binding) != {"source_step_id", "output_key"}
            ):
                return resolved, ("input_binding_invalid", {"step_id": step.id, "target": str(target)})
            source_id = binding.get("source_step_id")
            output_key = binding.get("output_key")
            if not isinstance(source_id, str) or source_id not in step.depends_on:
                return resolved, ("input_binding_source_not_dependency", {"step_id": step.id, "source_step_id": source_id})
            if source_id not in context.verified_result_ids:
                return resolved, ("input_binding_source_unverified", {"step_id": step.id, "source_step_id": source_id})
            source_result = context.previous_results.get(source_id)
            if (
                not isinstance(source_result, StepResult)
                or source_result.step_id != source_id
                or source_result.status != "success"
                or source_result.error is not None
                or not isinstance(source_result.output, dict)
            ):
                return resolved, ("input_binding_source_untrusted", {"step_id": step.id, "source_step_id": source_id})
            try:
                source_tool = self.tools.get(source_result.tool_name)
            except Exception:
                return resolved, ("input_binding_source_tool_unavailable", {"step_id": step.id, "source_step_id": source_id})
            try:
                source_schema = source_tool.output_schema()
                if not _is_supported_schema_shape(source_schema):
                    return resolved, ("input_binding_source_schema_invalid", {"step_id": step.id, "source_step_id": source_id})
                source_properties = source_schema.get("properties", {})
                source_required = source_schema.get("required", [])
            except Exception:
                return resolved, ("input_binding_source_schema_invalid", {"step_id": step.id, "source_step_id": source_id})
            if not isinstance(output_key, str) or not output_key or not isinstance(source_properties, dict) or not isinstance(source_required, list):
                return resolved, ("input_binding_output_unavailable", {"step_id": step.id, "source_step_id": source_id})
            try:
                target_schema = target_properties.get(target)
                source_field_schema = source_properties.get(output_key)
            except Exception:
                return resolved, ("input_binding_source_schema_invalid", {"step_id": step.id, "source_step_id": source_id})
            if (
                output_key not in source_result.output
                or output_key not in source_required
                or not isinstance(source_field_schema, dict)
                or not isinstance(target_schema, dict)
            ):
                return resolved, ("input_binding_output_unavailable", {"step_id": step.id, "source_step_id": source_id, "output_key": output_key})
            try:
                source_output_errors = validate_schema_value(source_result.output, source_schema, path=f"output.{source_id}")
            except Exception:
                return resolved, ("input_binding_source_schema_invalid", {"step_id": step.id, "source_step_id": source_id})
            if source_output_errors:
                return resolved, ("input_binding_source_output_invalid", {"step_id": step.id, "source_step_id": source_id})
            value = source_result.output[output_key]
            try:
                target_value_errors = validate_schema_value(value, target_schema, path=f"input.{target}")
            except Exception:
                return resolved, ("input_binding_target_schema_invalid", {"step_id": step.id})
            if target_value_errors:
                return resolved, ("input_binding_value_invalid", {"step_id": step.id, "source_step_id": source_id, "target": target})
            resolved[target] = value
        return resolved, None

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
            if not isinstance(dependency_result, StepResult):
                return (
                    "dependency_result_invalid",
                    f"Dependency result is malformed: {dependency}",
                    {"dependency": dependency},
                )
            if dependency_result.step_id != dependency:
                return (
                    "dependency_identity_mismatch",
                    f"Dependency result does not belong to the requested step: {dependency}",
                    {"dependency": dependency, "result_step_id": dependency_result.step_id},
                )
            if dependency_result.status != "success" or dependency_result.error is not None:
                return (
                    "dependency_failed",
                    f"Dependency did not succeed: {dependency}",
                    {"dependency": dependency, "dependency_status": dependency_result.status},
                )
        return None

    @staticmethod
    def _timed(metadata: dict[str, Any], started: float) -> dict[str, Any]:
        return {**metadata, "duration_ms": round((monotonic() - started) * 1000, 3)}
