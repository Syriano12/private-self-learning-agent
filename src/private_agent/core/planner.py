from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from private_agent.core.llm import LLMProvider
from private_agent.tools.research import ToolRegistry


class PlanValidationError(ValueError):
    """Raised when an LLM-generated plan violates deterministic constraints."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Invalid plan: " + "; ".join(errors))


@dataclass
class PlanStep:
    id: str
    objective: str
    tool: str
    input: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    depends_on: list[str] = field(default_factory=list)
    verifier: str = ""
    success_criteria: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    output: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    goal: str
    steps: list[PlanStep]
    rationale: str


PLAN_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "rationale": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "objective": {"type": "string"},
                    "tool": {"type": "string"},
                    "input": {"type": "object"},
                    "reason": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "permission_level": {"type": "string"},
                    "verifier": {"type": "string"},
                    "success_criteria": {"type": "object"},
                },
                "required": ["id", "objective", "tool", "input", "reason", "depends_on"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["goal", "rationale", "steps"],
    "additionalProperties": False,
}


class Planner:
    """Generate plans through an injected LLM provider and validate them deterministically."""

    def __init__(self, provider: LLMProvider | None = None, *, max_steps: int = 12) -> None:
        self.provider = provider
        self.max_steps = max_steps

    def create(
        self,
        goal: str,
        tools: ToolRegistry,
        prior_knowledge: list[dict[str, Any]],
        *,
        constraints: dict[str, Any] | None = None,
        permissions: dict[str, str] | None = None,
        retrieved_experiences: list[dict[str, Any]] | None = None,
    ) -> TaskPlan:
        if not goal.strip():
            raise PlanValidationError(["goal_required"])
        if self.provider is None:
            return self._unavailable_fallback(goal, tools, prior_knowledge)

        context = {
            "goal": goal,
            "available_tools": tools.describe(),
            "relevant_memory": prior_knowledge,
            "relevant_experiences": retrieved_experiences or [],
            "constraints": constraints or {"max_steps": self.max_steps},
            "permissions": permissions or {},
        }
        response = self.provider.generate_json(
            system_prompt=(
                "You are a planning component. Return only a JSON object matching the supplied schema. "
                "Propose a minimal safe plan using only registered tools. Do not execute tools."
            ),
            user_prompt=json.dumps(context, ensure_ascii=False, sort_keys=True),
            response_schema=PLAN_RESPONSE_SCHEMA,
        )
        return self.validate(response.data, goal=goal, tools=tools, permissions=permissions)

    def validate(
        self,
        payload: Any,
        *,
        goal: str,
        tools: ToolRegistry,
        permissions: dict[str, str] | None = None,
    ) -> TaskPlan:
        errors: list[str] = []
        if not isinstance(payload, dict):
            raise PlanValidationError(["plan_must_be_object"])
        allowed_plan_keys = {"goal", "rationale", "steps"}
        errors.extend(f"unknown_plan_field:{key}" for key in payload if key not in allowed_plan_keys)

        payload_goal = payload.get("goal")
        if not isinstance(payload_goal, str) or not payload_goal.strip():
            errors.append("goal_required")
        elif payload_goal != goal:
            errors.append("goal_mismatch")
        steps_payload = payload.get("steps")
        if not isinstance(steps_payload, list):
            errors.append("steps_must_be_array")
            steps_payload = []
        elif not steps_payload:
            errors.append("steps_required")
        if len(steps_payload) > self.max_steps:
            errors.append(f"too_many_steps:max={self.max_steps}")
        rationale = payload.get("rationale")
        if not isinstance(rationale, str):
            errors.append("rationale_must_be_string")

        step_ids: set[str] = set()
        dependencies: dict[str, list[str]] = {}
        steps: list[PlanStep] = []
        for index, raw_step in enumerate(steps_payload):
            if not isinstance(raw_step, dict):
                errors.append(f"step_{index}_must_be_object")
                continue
            allowed_step_keys = {
                "id",
                "objective",
                "tool",
                "input",
                "reason",
                "depends_on",
                "permission_level",
                "verifier",
                "success_criteria",
            }
            errors.extend(f"unknown_step_field:{index}.{key}" for key in raw_step if key not in allowed_step_keys)
            step_id = raw_step.get("id")
            objective = raw_step.get("objective")
            tool_name = raw_step.get("tool")
            step_input = raw_step.get("input")
            reason = raw_step.get("reason")
            depends_on = raw_step.get("depends_on")
            verifier = raw_step.get("verifier", "")
            success_criteria = raw_step.get("success_criteria", {})

            if not isinstance(step_id, str) or not step_id.strip():
                errors.append(f"step_{index}_id_required")
                step_id = f"invalid_{index}"
            elif step_id in step_ids:
                errors.append(f"duplicate_step_id:{step_id}")
            step_ids.add(step_id)
            if not isinstance(objective, str) or not objective.strip():
                errors.append(f"step_{step_id}_objective_required")
                objective = ""
            if not isinstance(tool_name, str) or not tool_name.strip():
                errors.append(f"step_{step_id}_tool_required")
                tool_name = ""
            if not isinstance(step_input, dict):
                errors.append(f"step_{step_id}_input_must_be_object")
                step_input = {}
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"step_{step_id}_reason_required")
                reason = ""
            if not isinstance(depends_on, list) or not all(isinstance(dep, str) for dep in depends_on):
                errors.append(f"step_{step_id}_depends_on_must_be_string_array")
                depends_on = []
            if not isinstance(verifier, str):
                errors.append(f"step_{step_id}_verifier_must_be_string")
                verifier = ""
            if not isinstance(success_criteria, dict):
                errors.append(f"step_{step_id}_success_criteria_must_be_object")
                success_criteria = {}

            if tool_name:
                try:
                    tool = tools.get(tool_name)
                except KeyError:
                    tool = None
                    errors.append(f"unknown_tool:{tool_name}")
                if tool is not None:
                    tool_schema = getattr(tool, "input_schema", lambda: {})()
                    self._validate_input(step_id, step_input, tool_schema, errors)
                    declared_permission = getattr(tool, "permission_level", None)
                    requested_permission = raw_step.get("permission_level", declared_permission)
                    if requested_permission != declared_permission:
                        errors.append(f"permission_mismatch:{tool_name}")
                    if permissions and permissions.get(tool_name) != declared_permission:
                        errors.append(f"permission_not_granted:{tool_name}")

            dependencies[step_id] = depends_on
            steps.append(
                PlanStep(
                    id=step_id,
                    objective=objective,
                    tool=tool_name,
                    input=step_input,
                    reason=reason,
                    depends_on=depends_on,
                    verifier=verifier,
                    success_criteria=success_criteria,
                )
            )

        for step_id, depends_on in dependencies.items():
            for dependency in depends_on:
                if dependency not in step_ids:
                    errors.append(f"unknown_dependency:{step_id}->{dependency}")
        if self._has_cycle(dependencies):
            errors.append("circular_dependency")

        if errors:
            raise PlanValidationError(errors)
        return TaskPlan(goal=goal, steps=steps, rationale=rationale)

    @staticmethod
    def _validate_input(step_id: str, values: dict[str, Any], schema: dict[str, Any], errors: list[str]) -> None:
        required = schema.get("required", []) if isinstance(schema, dict) else []
        for name in required:
            if name not in values:
                errors.append(f"missing_required_input:{step_id}.{name}")
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if schema.get("additionalProperties") is False:
            for name in values:
                if name not in properties:
                    errors.append(f"unknown_input:{step_id}.{name}")
        for name, definition in properties.items():
            if name not in values:
                continue
            value = values[name]
            expected = definition.get("type") if isinstance(definition, dict) else None
            if expected == "string" and not isinstance(value, str):
                errors.append(f"invalid_input_type:{step_id}.{name}:string")
            elif expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                errors.append(f"invalid_input_type:{step_id}.{name}:integer")
            if isinstance(value, int):
                minimum = definition.get("minimum")
                maximum = definition.get("maximum")
                if minimum is not None and value < minimum:
                    errors.append(f"input_below_minimum:{step_id}.{name}")
                if maximum is not None and value > maximum:
                    errors.append(f"input_above_maximum:{step_id}.{name}")

    @staticmethod
    def _has_cycle(dependencies: dict[str, list[str]]) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(visit(dep) for dep in dependencies.get(node, [])):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in dependencies)

    @staticmethod
    def _unavailable_fallback(
        goal: str,
        tools: ToolRegistry,
        prior_knowledge: list[dict[str, Any]],
    ) -> TaskPlan:
        available = tools.available()
        if not available:
            raise PlanValidationError(["no_tools_registered", "llm_provider_unavailable"])
        first_tool = tools.get(available[0]["name"])
        input_values: dict[str, Any] = {}
        schema = getattr(first_tool, "input_schema", lambda: {})()
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if "query" in properties:
            input_values["query"] = goal
        if "limit" in properties:
            input_values["limit"] = 6
        return TaskPlan(
            goal=goal,
            steps=[
                PlanStep(
                    id=str(uuid.uuid4()),
                    objective="Execute the first registered safe tool as a compatibility fallback",
                    tool=available[0]["name"],
                    input=input_values,
                    reason="LLM provider is not configured; this fallback is not dynamic planning.",
                )
            ],
            rationale=(
                f"استُرجعت {len(prior_knowledge)} عناصر معرفة سابقة. "
                "LLM planner unavailable; "
                "The compatibility fallback is not an LLM-generated plan."
            ),
        )
