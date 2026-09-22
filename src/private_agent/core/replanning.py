from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from private_agent.core.diagnosis import FailureDiagnosis
from private_agent.core.llm import LLMProvider
from private_agent.core.observation import Observation
from private_agent.core.planner import PLAN_RESPONSE_SCHEMA, PlanStep, PlanValidationError, Planner, TaskPlan
from private_agent.tools.research import ToolRegistry


class ReplanError(RuntimeError):
    pass


@dataclass
class ReplanRequest:
    current_plan: TaskPlan
    failure_observation: Observation
    diagnosis: FailureDiagnosis
    executed_history: list[dict[str, Any]]
    constraints: dict[str, Any]
    permissions: dict[str, str]


class Replanner:
    """Propose a replacement plan, then force it through the existing Planner validator."""

    def __init__(self, provider: LLMProvider | None = None, *, max_steps: int = 12) -> None:
        self.provider = provider
        self.planner = Planner(provider, max_steps=max_steps)

    def replan(
        self,
        request: ReplanRequest,
        tools: ToolRegistry,
        *,
        replacement_tool: str = "",
    ) -> TaskPlan:
        if self.provider is not None:
            return self._llm_replan(request, tools)
        return self._deterministic_replan(request, tools, replacement_tool=replacement_tool)

    def replan_step(
        self,
        current_plan: TaskPlan,
        *,
        step_id: str,
        tool_name: str,
        inputs: dict[str, Any],
        tools: ToolRegistry,
        permissions: dict[str, str],
    ) -> TaskPlan:
        steps = []
        found = False
        for step in current_plan.steps:
            payload = self._step_payload(step)
            if step.id == step_id:
                payload["tool"] = tool_name
                payload["input"] = inputs
                found = True
            steps.append(payload)
        if not found:
            raise ReplanError("failed_step_not_in_current_plan")
        payload = {
            "goal": current_plan.goal,
            "rationale": f"Validated recovery update for step {step_id}",
            "steps": steps,
        }
        try:
            return self.planner.validate(payload, goal=current_plan.goal, tools=tools, permissions=permissions)
        except PlanValidationError as exc:
            raise ReplanError(f"Recovery plan failed validation: {exc.errors}") from exc

    def _llm_replan(self, request: ReplanRequest, tools: ToolRegistry) -> TaskPlan:
        context = {
            "goal": request.current_plan.goal,
            "current_plan": self._plan_payload(request.current_plan),
            "failure_observation": request.failure_observation.to_dict(),
            "failure_diagnosis": request.diagnosis.to_dict(),
            "executed_history": request.executed_history,
            "available_tools": tools.describe(),
            "constraints": request.constraints,
            "permissions": request.permissions,
        }
        response = self.provider.generate_json(
            system_prompt=(
                "You are a recovery planning component. Propose a replacement plan only. "
                "Do not execute tools. Use only registered tools and obey permissions. "
                "Return JSON matching the supplied plan schema."
            ),
            user_prompt=json.dumps(context, ensure_ascii=False, sort_keys=True),
            response_schema=PLAN_RESPONSE_SCHEMA,
        )
        try:
            return self.planner.validate(
                response.data,
                goal=request.current_plan.goal,
                tools=tools,
                permissions=request.permissions,
            )
        except PlanValidationError as exc:
            raise ReplanError(f"LLM re-plan failed validation: {exc.errors}") from exc

    def _deterministic_replan(
        self,
        request: ReplanRequest,
        tools: ToolRegistry,
        *,
        replacement_tool: str = "",
    ) -> TaskPlan:
        failed_step_id = request.diagnosis.failed_step
        failed_step = next((step for step in request.current_plan.steps if step.id == failed_step_id), None)
        if failed_step is None:
            raise ReplanError("failed_step_not_in_current_plan")
        failed_tool = failed_step.tool
        alternatives = [metadata for metadata in tools.describe() if metadata["name"] != failed_tool]
        if not alternatives:
            raise ReplanError("no_alternative_tool_registered")

        replacement_metadata = next(
            (metadata for metadata in alternatives if metadata["name"] == replacement_tool),
            alternatives[0],
        )
        replacement_name = replacement_metadata["name"]
        replacement_input = self._inputs_for_tool(
            replacement_metadata["input_schema"],
            failed_step.input,
            request.current_plan.goal,
        )
        replacement = PlanStep(
            id=failed_step.id,
            objective=f"Recovery alternative for {failed_step.objective}",
            tool=replacement_name,
            input=replacement_input,
            reason=f"Replace failed tool {failed_tool} after {request.diagnosis.failure_type}",
            depends_on=[dep for dep in failed_step.depends_on if dep != failed_step.id],
            verifier="",
            success_criteria=dict(failed_step.success_criteria),
        )
        steps: list[dict[str, Any]] = []
        replaced = False
        for step in request.current_plan.steps:
            if step.id == failed_step.id:
                steps.append(self._step_payload(replacement))
                replaced = True
            else:
                steps.append(self._step_payload(step))
        if not replaced:
            raise ReplanError("failed_step_not_replaced")
        payload = {
            "goal": request.current_plan.goal,
            "rationale": f"Deterministic re-plan replaced {failed_tool} with {replacement_name}",
            "steps": steps,
        }
        try:
            return self.planner.validate(
                payload,
                goal=request.current_plan.goal,
                tools=tools,
                permissions=request.permissions,
            )
        except PlanValidationError as exc:
            raise ReplanError(f"Deterministic re-plan failed validation: {exc.errors}") from exc

    @staticmethod
    def _inputs_for_tool(schema: dict[str, Any], previous: dict[str, Any], goal: str) -> dict[str, Any]:
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = schema.get("required", []) if isinstance(schema, dict) else []
        values: dict[str, Any] = {}
        for name in required:
            if name in previous:
                values[name] = previous[name]
            elif name == "query":
                values[name] = goal
            elif properties.get(name, {}).get("type") == "integer":
                values[name] = 3
            elif properties.get(name, {}).get("type") == "boolean":
                values[name] = False
            else:
                values[name] = ""
        for name in properties:
            if name in previous and name not in values:
                values[name] = previous[name]
        return values

    @staticmethod
    def _step_payload(step: PlanStep) -> dict[str, Any]:
        return {
            "id": step.id,
            "objective": step.objective,
            "tool": step.tool,
            "input": step.input,
            "reason": step.reason,
            "depends_on": step.depends_on,
            "verifier": step.verifier,
            "success_criteria": step.success_criteria,
        }

    def _plan_payload(self, plan: TaskPlan) -> dict[str, Any]:
        return {
            "goal": plan.goal,
            "rationale": plan.rationale,
            "steps": [self._step_payload(step) for step in plan.steps],
        }
