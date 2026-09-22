from __future__ import annotations

from pathlib import Path

from private_agent.core.execution import ExecutionContext, GenericExecutor, StepResult
from private_agent.core.observation import Observation
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.core.verification import (
    GenericVerifier,
    VerificationResult,
    VerifierRegistry,
    WebResearchVerifier,
)
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class OutputTool:
    permission_level = "public_read"
    risk_level = "low"

    def __init__(self, name: str, output, *, output_schema: dict | None = None):
        self.name = name
        self.output = output
        self.calls = 0
        self._output_schema = output_schema or {"type": "object"}

    def description(self) -> str:
        return self.name

    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    def output_schema(self) -> dict:
        return self._output_schema

    def execute(self, inputs: dict, context: ExecutionContext) -> ToolResult:
        self.calls += 1
        return ToolResult.success(self.output)


class RaisingVerifier:
    name = "raising"

    def verify(self, observation: Observation, criteria: dict) -> VerificationResult:
        raise RuntimeError("verification broke")


class CustomVerifier:
    name = "custom"

    def verify(self, observation: Observation, criteria: dict) -> VerificationResult:
        return VerificationResult("VERIFIED", "custom criterion met", verifier_type=self.name)


def make_registry(tool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def make_step(tool: str, *, verifier: str = "", criteria: dict | None = None) -> PlanStep:
    return PlanStep(
        id="step-1",
        objective="test",
        tool=tool,
        input={},
        reason="test",
        verifier=verifier,
        success_criteria=criteria or {},
    )


def test_observation_is_structured_and_separate_from_tool_result() -> None:
    result = StepResult.success("step-1", "empty", {"value": 1}, metadata={"duration_ms": 2})
    observation = Observation.from_step_result("task-1", result)
    assert observation.task_id == "task-1"
    assert observation.step_id == "step-1"
    assert observation.tool_name == "empty"
    assert observation.execution_status == "success"
    assert observation.observed_output == {"value": 1}
    assert observation is not result
    assert observation.to_dict()["timestamp"]


def test_successful_execution_with_empty_output_is_insufficient_not_verified() -> None:
    tool = OutputTool("empty", {})
    registry = make_registry(tool)
    execution = GenericExecutor(registry).execute_step(make_step("empty"))
    assert execution.status == "success"
    observation = Observation.from_step_result("task", execution)
    verification = VerifierRegistry().verify("generic", observation, {"required_fields": []})
    assert verification.status == "INSUFFICIENT"
    assert verification.verified is False
    assert verification.details["code"] == "missing_output"


def test_successful_execution_with_non_empty_output_missing_required_field_fails() -> None:
    tool = OutputTool("partial", {"other": "evidence"})
    execution = GenericExecutor(make_registry(tool)).execute_step(make_step("partial"))
    assert execution.status == "success"
    observation = Observation.from_step_result("task", execution)
    verification = VerifierRegistry().verify("generic", observation, {"required_fields": ["value"]})
    assert verification.status == "FAILED"
    assert verification.details["code"] == "missing_required_output"


def test_missing_success_criteria_is_insufficient() -> None:
    tool = OutputTool("valid", {"value": "evidence"})
    execution = GenericExecutor(make_registry(tool)).execute_step(make_step("valid"))
    observation = Observation.from_step_result("task", execution)
    verification = VerifierRegistry().verify("generic", observation, {})
    assert execution.status == "success"
    assert verification.status == "INSUFFICIENT"
    assert verification.details["code"] == "missing_success_criteria"


def test_successful_execution_with_valid_output_is_verified() -> None:
    tool = OutputTool("valid", {"value": "evidence"})
    execution = GenericExecutor(make_registry(tool)).execute_step(make_step("valid", criteria={"required_fields": ["value"]}))
    observation = Observation.from_step_result("task", execution)
    verification = VerifierRegistry().verify("generic", observation, {"required_fields": ["value"]})
    assert execution.status == "success"
    assert verification.status == "VERIFIED"
    assert verification.evidence


def test_blocked_observation_becomes_blocked_verification() -> None:
    tool = OutputTool("private", {"value": "secret"})
    tool.permission_level = "private_write"
    execution = GenericExecutor(make_registry(tool)).execute_step(make_step("private"))
    observation = Observation.from_step_result("task", execution)
    verification = VerifierRegistry().verify("generic", observation, {"required_fields": ["value"]})
    assert execution.status == "blocked"
    assert verification.status == "BLOCKED"


def test_web_research_verifier_empty_and_valid_sources() -> None:
    verifier = WebResearchVerifier()
    empty = Observation("task", "step", "web_research", "success", {"sources": [], "errors": []})
    insufficient = verifier.verify(empty, {"minimum_sources": 2, "require_non_empty_content": True})
    assert insufficient.status == "INSUFFICIENT"

    valid = Observation(
        "task",
        "step",
        "web_research",
        "success",
        {
            "sources": [
                {"url": "https://a.test", "title": "A", "text": "relevant evidence"},
                {"url": "https://b.test", "title": "B", "text": "more evidence"},
            ],
            "errors": [],
        },
    )
    verified = verifier.verify(valid, {"minimum_sources": 2, "required_terms": ["evidence"]})
    assert verified.status == "VERIFIED"
    assert len(verified.evidence) == 3


def test_web_research_http_or_fetch_error_is_not_verified() -> None:
    observation = Observation(
        "task",
        "step",
        "web_research",
        "success",
        {"sources": [{"url": "https://a.test", "text": "evidence"}], "errors": ["fetch_failed"]},
    )
    result = WebResearchVerifier().verify(observation, {"minimum_sources": 1})
    assert result.status == "FAILED"
    assert result.verified is False


def test_unknown_verifier_is_blocked_and_verifier_exception_is_failed() -> None:
    registry = VerifierRegistry([GenericVerifier(), RaisingVerifier()])
    observation = Observation("task", "step", "tool", "success", {"value": 1})
    unknown = registry.verify("does_not_exist", observation, {"required_fields": ["value"]})
    assert unknown.status == "BLOCKED"
    assert unknown.details["code"] == "unknown_verifier"
    raised = registry.verify("raising", observation, {})
    assert raised.status == "FAILED"
    assert raised.details["code"] == "verification_exception"


def test_adding_a_verifier_does_not_require_generic_executor_changes() -> None:
    registry = VerifierRegistry()
    registry.register(CustomVerifier())
    tool = OutputTool("custom_tool", {"value": "ok"})
    execution = GenericExecutor(make_registry(tool)).execute_step(make_step("custom_tool", verifier="custom"))
    observation = Observation.from_step_result("task", execution)
    result = registry.verify("custom", observation, {})
    assert execution.status == "success"
    assert result.status == "VERIFIED"


class StubPlanner:
    def __init__(self, tool_name: str, *, criteria: dict | None = None):
        self.tool_name = tool_name
        self.criteria = criteria or {}
        self.calls = 0

    def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict]) -> TaskPlan:
        self.calls += 1
        return TaskPlan(goal, [make_step(self.tool_name, criteria=self.criteria)], "stub")


class RecordingExecutor:
    def __init__(self, output: dict):
        self.output = output
        self.calls = 0

    def execute_plan(self, plan: TaskPlan, context: ExecutionContext) -> list[StepResult]:
        self.calls += 1
        return [StepResult.success(plan.steps[0].id, plan.steps[0].tool, self.output)]


def test_orchestrator_uses_verification_and_does_not_retry_or_replan_on_failure(tmp_path: Path) -> None:
    tool = OutputTool("empty", {})
    planner = StubPlanner("empty")
    executor = RecordingExecutor({})
    store = Store(tmp_path / "agent.sqlite3")
    agent = Orchestrator(store, make_registry(tool), planner=planner, executor=executor)
    result = agent.run("goal")
    assert result["step_results"][0]["status"] == "success"
    assert result["step_outcomes"][0]["execution_status"] == "success"
    assert result["step_outcomes"][0]["verification_status"] == "INSUFFICIENT"
    assert result["verification"]["status"] == "INSUFFICIENT"
    assert result["verification"]["verified"] is False
    assert result["attempts"] == 1
    assert executor.calls == 1
    assert planner.calls == 1
    assert len(store.observations_for_task(result["task_id"])) == 1
    assert len(store.verifications_for_task(result["task_id"])) == 1
    assert {event["event_type"] for event in store.events_for_task(result["task_id"])} >= {
        "plan_created",
        "observation_created",
        "verification_completed",
        "step_completed",
    }


def test_orchestrator_reports_verified_only_after_valid_research_evidence(tmp_path: Path) -> None:
    tool = OutputTool("web_research", {"sources": [{"url": "https://a", "text": "one"}, {"url": "https://b", "text": "two"}], "errors": []})
    executor = RecordingExecutor(tool.output)
    agent = Orchestrator(
        Store(tmp_path / "agent.sqlite3"),
        make_registry(tool),
        planner=StubPlanner("web_research", criteria={"minimum_sources": 2}),
        executor=executor,
    )
    result = agent.run("goal")
    assert result["step_outcomes"][0]["execution_status"] == "success"
    assert result["step_outcomes"][0]["verification_status"] == "VERIFIED"
    assert result["verification"]["status"] == "VERIFIED"
    assert result["verification"]["verified"] is True
