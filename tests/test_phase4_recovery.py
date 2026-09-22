from __future__ import annotations

from pathlib import Path

import pytest

from private_agent.core.diagnosis import FailureDiagnoser
from private_agent.core.execution import ExecutionContext, GenericExecutor, StepResult
from private_agent.core.observation import Observation
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.core.recovery import RecoveryManager, input_fingerprint
from private_agent.core.replanning import ReplanError, ReplanRequest, Replanner
from private_agent.core.verification import VerificationResult, VerifierRegistry
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class ControlledTool:
    risk_level = "low"
    permission_level = "public_read"

    def __init__(self, name: str, outcomes: list[ToolResult], *, required: list[str] | None = None):
        self.name = name
        self.outcomes = list(outcomes)
        self.required = required or []
        self.calls = 0

    def description(self) -> str:
        return self.name

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {field: {"type": "string"} for field in self.required},
            "required": self.required,
            "additionalProperties": False,
        }

    def output_schema(self) -> dict:
        return {"type": "object"}

    def execute(self, inputs: dict, context: ExecutionContext) -> ToolResult:
        self.calls += 1
        if self.outcomes:
            return self.outcomes.pop(0)
        return ToolResult.failed("tool_failed", "no configured outcome")


def registry(*tools) -> ToolRegistry:
    result = ToolRegistry()
    for tool in tools:
        result.register(tool)
    return result


def observation(status: str = "failed", output=None, error=None, tool: str = "tool") -> Observation:
    return Observation("task", "step", tool, status, output, error)


def failure_step(tool: str = "tool", *, inputs: dict | None = None, criteria: dict | None = None) -> PlanStep:
    return PlanStep(
        id="step",
        objective="perform task",
        tool=tool,
        input=inputs or {},
        reason="test",
        success_criteria=criteria or {"required_fields": ["result"]},
    )


def test_failure_diagnoser_classifies_execution_and_verification_failures() -> None:
    diagnoser = FailureDiagnoser()
    tool_error = StepResult.failed("step", "tool", "tool_exception", "boom", details={"exception_type": "RuntimeError"})
    diagnosis = diagnoser.diagnose(
        step_result=tool_error,
        observation=observation(error=tool_error.error.to_dict()),
        verification=VerificationResult("FAILED", "execution failed"),
    )
    assert diagnosis is not None
    assert diagnosis.failure_type == "TOOL_ERROR"
    assert diagnosis.suggested_recovery == "CHANGE_TOOL"

    invalid = StepResult.failed("step", "tool", "invalid_input", "bad input")
    invalid_diagnosis = diagnoser.diagnose(
        step_result=invalid,
        observation=observation(error=invalid.error.to_dict()),
        verification=VerificationResult("FAILED", "execution failed"),
    )
    assert invalid_diagnosis is not None
    assert invalid_diagnosis.failure_type == "INVALID_INPUT"
    assert invalid_diagnosis.suggested_recovery == "CHANGE_PARAMETERS"

    verification_failure = diagnoser.diagnose(
        step_result=StepResult.success("step", "tool", {"other": "x"}),
        observation=observation("success", {"other": "x"}),
        verification=VerificationResult("VERIFIED", "should be ignored") if False else VerificationResult("FAILED", "evidence missing"),
    )
    assert verification_failure is not None
    assert verification_failure.failure_type == "VERIFICATION_FAILED"


def test_failure_diagnoser_classifies_permission_unknown_dependency_and_unknown() -> None:
    diagnoser = FailureDiagnoser()
    cases = [
        ("permission_not_approved", "PERMISSION_BLOCKED", False),
        ("missing_dependency", "MISSING_DEPENDENCY", False),
        ("unexpected", "UNKNOWN_FAILURE", True),
    ]
    for code, expected, recoverable in cases:
        result = StepResult.failed("step", "tool", code, "failure")
        diagnosis = diagnoser.diagnose(
            step_result=result,
            observation=observation(error=result.error.to_dict()),
            verification=VerificationResult("FAILED", "execution failed"),
        )
        assert diagnosis is not None
        assert diagnosis.failure_type == expected
        assert diagnosis.recoverable is recoverable


def test_failure_diagnoser_classifies_timeout_rate_limit_and_network_errors() -> None:
    diagnoser = FailureDiagnoser()
    for code, expected in [("timeout", "TIMEOUT"), ("rate_limited", "RATE_LIMITED"), ("network_error", "NETWORK_ERROR")]:
        result = StepResult.failed("step", "tool", code, "transient failure")
        diagnosis = diagnoser.diagnose(
            step_result=result,
            observation=observation(error=result.error.to_dict()),
            verification=VerificationResult("FAILED", "execution failed"),
        )
        assert diagnosis is not None
        assert diagnosis.failure_type == expected


def test_recovery_retry_is_bounded_and_exact_strategy_does_not_repeat() -> None:
    manager = RecoveryManager(max_recovery_attempts=3, max_attempts_per_step=3)
    step = failure_step(inputs={"query": "x"})
    diagnosis = FailureDiagnoser().diagnose(
        step_result=StepResult.failed("step", "tool", "network_error", "temporary"),
        observation=observation(error={"code": "network_error"}),
        verification=VerificationResult("FAILED", "execution failed"),
    )
    assert diagnosis is not None
    first = manager.choose(step, diagnosis, available_tools=[{"name": "tool"}])
    assert first.strategy == "RETRY"
    manager.record(first, diagnosis, tool_name="tool", inputs=step.input)
    second = manager.choose(step, diagnosis, available_tools=[{"name": "tool"}])
    assert second.strategy == "ABORT"
    assert second.terminal_status == "FAILED"
    assert input_fingerprint(step.input) == manager.attempts[0].input_fingerprint


def test_recovery_global_limit_produces_aborted_status() -> None:
    manager = RecoveryManager(max_recovery_attempts=1, max_attempts_per_step=3)
    step = failure_step(inputs={"query": "x"})
    diagnosis = FailureDiagnoser().diagnose(
        step_result=StepResult.failed("step", "tool", "network_error", "temporary"),
        observation=observation(error={"code": "network_error"}),
        verification=VerificationResult("FAILED", "execution failed"),
    )
    assert diagnosis is not None
    first = manager.choose(step, diagnosis, available_tools=[{"name": "tool"}])
    manager.record(first, diagnosis, tool_name="tool", inputs=step.input)
    terminal = manager.choose(step, diagnosis, available_tools=[{"name": "tool"}])
    assert terminal.strategy == "ABORT"
    assert terminal.terminal_status == "ABORTED"


def test_recovery_changes_tool_or_parameters_without_executor_hardcoding() -> None:
    manager = RecoveryManager(max_recovery_attempts=3, max_attempts_per_step=2)
    step = failure_step("a", inputs={"query": "x"})
    diagnosis = FailureDiagnoser().diagnose(
        step_result=StepResult.failed("step", "a", "tool_exception", "unavailable"),
        observation=observation(error={"code": "tool_exception"}, tool="a"),
        verification=VerificationResult("FAILED", "execution failed"),
    )
    assert diagnosis is not None
    decision = manager.choose(step, diagnosis, available_tools=[{"name": "a"}, {"name": "b"}])
    assert decision.strategy == "CHANGE_TOOL"
    assert decision.replacement_tool == "b"

    input_step = failure_step("a", inputs={"query": "x", "limit": 6})
    input_diagnosis = FailureDiagnoser().diagnose(
        step_result=StepResult.failed("step", "a", "invalid_input", "bad"),
        observation=observation(error={"code": "invalid_input"}, tool="a"),
        verification=VerificationResult("FAILED", "execution failed"),
    )
    assert input_diagnosis is not None
    modified = manager.choose(input_step, input_diagnosis, available_tools=[{"name": "a"}])
    assert modified.strategy == "RETRY_WITH_MODIFIED_INPUT"
    assert modified.modified_input["limit"] == 3


def test_replanner_replaces_tool_and_revalidates_permissions() -> None:
    a = ControlledTool("a", [ToolResult.failed("tool_failed", "unavailable")])
    b = ControlledTool("b", [ToolResult.success({"result": "ok"})])
    tools = registry(a, b)
    current = TaskPlan("goal", [failure_step("a")], "initial")
    diagnosis = FailureDiagnoser().diagnose(
        step_result=StepResult.failed("step", "a", "tool_exception", "unavailable"),
        observation=observation(error={"code": "tool_exception"}, tool="a"),
        verification=VerificationResult("FAILED", "execution failed"),
    )
    assert diagnosis is not None
    request = ReplanRequest(current, observation(error={"code": "tool_exception"}, tool="a"), diagnosis, [], {}, {"a": "public_read", "b": "public_read"})
    replanned = Replanner().replan(request, tools, replacement_tool="b")
    assert replanned.steps[0].tool == "b"
    assert replanned.steps[0].id == "step"

    private = ControlledTool("private", [ToolResult.success({"result": "no"})])
    private.permission_level = "private_write"
    private_tools = registry(a, private)
    with pytest.raises(ReplanError, match="permission_not_granted"):
        Replanner().replan(request, private_tools, replacement_tool="private")


def test_llm_replanner_only_proposes_and_validation_still_runs() -> None:
    class FakeProvider:
        def __init__(self):
            self.calls = 0

        def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict):
            self.calls += 1
            from private_agent.core.llm import LLMResponse

            return LLMResponse(
                {
                    "goal": "goal",
                    "rationale": "use b",
                    "steps": [
                        {
                            "id": "step",
                            "objective": "perform task",
                            "tool": "b",
                            "input": {},
                            "reason": "alternative",
                            "depends_on": [],
                        }
                    ],
                },
                "{}",
                "fake",
            )

    a = ControlledTool("a", [])
    b = ControlledTool("b", [])
    tools = registry(a, b)
    provider = FakeProvider()
    current = TaskPlan("goal", [failure_step("a")], "initial")
    diagnosis = FailureDiagnoser().diagnose(
        step_result=StepResult.failed("step", "a", "tool_exception", "unavailable"),
        observation=observation(error={"code": "tool_exception"}, tool="a"),
        verification=VerificationResult("FAILED", "execution failed"),
    )
    assert diagnosis is not None
    replanned = Replanner(provider).replan(
        ReplanRequest(current, observation(error={"code": "tool_exception"}, tool="a"), diagnosis, [], {}, {"a": "public_read", "b": "public_read"}),
        tools,
    )
    assert provider.calls == 1
    assert replanned.steps[0].tool == "b"
    assert a.calls == 0 and b.calls == 0


def test_llm_replanner_rejects_invalid_plan_instead_of_executing_it() -> None:
    class InvalidProvider:
        def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict):
            from private_agent.core.llm import LLMResponse

            return LLMResponse(
                {"goal": "goal", "rationale": "unsafe", "steps": [{"id": "x", "tool": "not_registered"}]},
                "{}",
                "fake",
            )

    a = ControlledTool("a", [])
    tools = registry(a)
    current = TaskPlan("goal", [failure_step("a")], "initial")
    diagnosis = FailureDiagnoser().diagnose(
        step_result=StepResult.failed("step", "a", "tool_exception", "unavailable"),
        observation=observation(error={"code": "tool_exception"}, tool="a"),
        verification=VerificationResult("FAILED", "execution failed"),
    )
    assert diagnosis is not None
    with pytest.raises(ReplanError, match="failed validation"):
        Replanner(InvalidProvider()).replan(
            ReplanRequest(current, observation(error={"code": "tool_exception"}, tool="a"), diagnosis, [], {}, {"a": "public_read"}),
            tools,
        )
    assert a.calls == 0


def test_orchestrator_recovery_succeeds_then_verifies_and_records_experience(tmp_path: Path) -> None:
    a = ControlledTool("a", [ToolResult.failed("tool_failed", "unavailable")])
    b = ControlledTool("b", [ToolResult.success({"result": "evidence"})])
    tools = registry(a, b)

    class PlannerStub:
        provider = None
        max_steps = 12

        def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict]) -> TaskPlan:
            return TaskPlan(goal, [failure_step("a")], "initial")

    store = Store(tmp_path / "agent.sqlite3")
    result = Orchestrator(store, tools, planner=PlannerStub(), max_recovery_attempts=2).run("goal")
    assert result["final_status"] == "COMPLETED"
    assert result["verification"]["status"] == "VERIFIED"
    assert result["attempts"] == 2
    assert a.calls == 1 and b.calls == 1
    assert result["recovery_attempts"][0]["strategy"] == "CHANGE_TOOL"
    assert result["experience"]["actions"][0]["diagnosis"]["failure_type"] == "TOOL_ERROR"
    assert result["experience"]["actions"][0]["recovery"]["strategy"] == "CHANGE_TOOL"
    assert len(result["experience"]["actions"]) == 2
    assert store.get_experience(result["task_id"]) is not None


def test_orchestrator_recovery_failure_does_not_claim_success(tmp_path: Path) -> None:
    a = ControlledTool("a", [ToolResult.failed("tool_failed", "unavailable")])
    b = ControlledTool("b", [ToolResult.success({"other": "not enough"})])
    tools = registry(a, b)

    class PlannerStub:
        provider = None
        max_steps = 12

        def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict]) -> TaskPlan:
            return TaskPlan(goal, [failure_step("a")], "initial")

    result = Orchestrator(Store(tmp_path / "agent.sqlite3"), tools, planner=PlannerStub(), max_recovery_attempts=3).run("goal")
    assert result["final_status"] == "FAILED"
    assert result["verification"]["verified"] is False
    assert result["verification"]["status"] == "FAILED"
    assert result["attempts"] == 2
    assert a.calls == 1 and b.calls == 1
    assert len(result["recovery_attempts"]) == 2
    assert result["recovery_attempts"][-1]["outcome"] == "FAILED"


def test_permission_blocked_recovery_is_not_allowed_to_bypass_security(tmp_path: Path) -> None:
    a = ControlledTool("a", [ToolResult.failed("tool_failed", "unavailable")])
    private = ControlledTool("private", [ToolResult.success({"result": "should not run"})])
    private.permission_level = "private_write"
    tools = registry(a, private)

    class PlannerStub:
        provider = None
        max_steps = 12

        def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict]) -> TaskPlan:
            return TaskPlan(goal, [failure_step("a")], "initial")

    result = Orchestrator(Store(tmp_path / "agent.sqlite3"), tools, planner=PlannerStub(), max_recovery_attempts=2).run("goal")
    assert result["final_status"] == "BLOCKED"
    assert private.calls == 0
    assert any("replan" in error.lower() or "permission" in error.lower() for error in result["errors"])
