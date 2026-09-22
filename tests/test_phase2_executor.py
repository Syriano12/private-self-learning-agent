from __future__ import annotations

from pathlib import Path

from private_agent.core.execution import ExecutionContext, GenericExecutor, StepResult
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class FakeContractTool:
    permission_level = "public_read"
    risk_level = "low"

    def __init__(self, name: str, *, required: list[str] | None = None, failure: str | None = None, malformed: bool = False):
        self.name = name
        self.calls: list[dict] = []
        self.required = required or []
        self.failure = failure
        self.malformed = malformed

    def description(self) -> str:
        return f"Fake contract tool {self.name}"

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {name: {"type": "string"} for name in self.required},
            "required": self.required,
            "additionalProperties": False,
        }

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        }

    def execute(self, inputs: dict, context: ExecutionContext):
        self.calls.append(inputs)
        if self.failure == "exception":
            raise RuntimeError("boom")
        if self.malformed:
            return {"result": "not wrapped"}
        if self.failure:
            return ToolResult.failed(self.failure, "fake tool failure")
        return ToolResult.success({"result": f"{self.name}:{inputs}"})


def step(step_id: str, tool: str, *, inputs: dict | None = None, depends_on: list[str] | None = None) -> PlanStep:
    return PlanStep(
        id=step_id,
        objective=f"Run {tool}",
        tool=tool,
        input=inputs or {},
        reason="test contract execution",
        depends_on=depends_on or [],
    )


def registry(*tools: FakeContractTool) -> ToolRegistry:
    result = ToolRegistry()
    for tool in tools:
        result.register(tool)
    return result


def test_executor_executes_registered_tool_dynamically_without_web_research_name() -> None:
    alpha = FakeContractTool("alpha", required=["value"])
    tools = registry(alpha)
    executor = GenericExecutor(tools)
    result = executor.execute_step(step("s1", "alpha", inputs={"value": "hello"}))
    assert result.status == "success"
    assert result.tool_name == "alpha"
    assert result.output == {"result": "alpha:{'value': 'hello'}"}
    assert alpha.calls == [{"value": "hello"}]


def test_registry_discovery_and_executor_do_not_need_executor_changes_for_new_tool() -> None:
    beta = FakeContractTool("beta")
    tools = registry(beta)
    assert tools.describe()[0]["name"] == "beta"
    assert tools.describe()[0]["output_schema"]["required"] == ["result"]
    result = GenericExecutor(tools).execute_plan(TaskPlan("goal", [step("b", "beta")], "test"))[0]
    assert result.status == "success"
    assert result.tool_name == "beta"


def test_unknown_tool_returns_failed_and_is_not_executed() -> None:
    result = GenericExecutor(registry()).execute_step(step("missing", "not_registered"))
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "unknown_tool"


def test_invalid_input_returns_failed_without_running_tool() -> None:
    alpha = FakeContractTool("alpha", required=["value"])
    result = GenericExecutor(registry(alpha)).execute_step(
        step("s1", "alpha", inputs={"value": "ok", "forbidden": "no"})
    )
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "invalid_input"
    assert alpha.calls == []


def test_missing_required_input_returns_failed_without_running_tool() -> None:
    alpha = FakeContractTool("alpha", required=["value"])
    result = GenericExecutor(registry(alpha)).execute_step(step("s1", "alpha"))
    assert result.status == "failed"
    assert result.error is not None
    assert "missing_required_input:input.value" in result.error.details["errors"]
    assert alpha.calls == []


def test_successful_dependency_allows_dependent_step() -> None:
    first = FakeContractTool("first")
    second = FakeContractTool("second")
    plan = TaskPlan("goal", [step("one", "first"), step("two", "second", depends_on=["one"])], "test")
    results = GenericExecutor(registry(first, second)).execute_plan(plan)
    assert [result.status for result in results] == ["success", "success"]
    assert len(second.calls) == 1


def test_failed_dependency_blocks_dependent_step_without_running_it() -> None:
    first = FakeContractTool("first", failure="upstream_failed")
    second = FakeContractTool("second")
    plan = TaskPlan("goal", [step("one", "first"), step("two", "second", depends_on=["one"])], "test")
    results = GenericExecutor(registry(first, second)).execute_plan(plan)
    assert results[0].status == "failed"
    assert results[1].status == "blocked"
    assert results[1].error is not None
    assert results[1].error.code == "dependency_failed"
    assert second.calls == []


def test_missing_dependency_is_blocked() -> None:
    second = FakeContractTool("second")
    result = GenericExecutor(registry(second)).execute_step(step("two", "second", depends_on=["one"]))
    assert result.status == "blocked"
    assert result.error is not None
    assert result.error.code == "missing_dependency"
    assert second.calls == []


def test_tool_exception_is_structured_failure() -> None:
    failing = FakeContractTool("failing", failure="exception")
    result = GenericExecutor(registry(failing)).execute_step(step("s1", "failing"))
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "tool_exception"
    assert result.error.details == {"exception_type": "RuntimeError"}


def test_malformed_tool_result_is_structured_failure() -> None:
    malformed = FakeContractTool("malformed", malformed=True)
    result = GenericExecutor(registry(malformed)).execute_step(step("s1", "malformed"))
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "malformed_tool_result"


def test_unapproved_permission_is_blocked_before_execution() -> None:
    private_tool = FakeContractTool("private")
    private_tool.permission_level = "private_write"
    result = GenericExecutor(registry(private_tool)).execute_step(step("s1", "private"))
    assert result.status == "blocked"
    assert result.error is not None
    assert result.error.code == "permission_not_approved"
    assert private_tool.calls == []


def test_unregistered_code_tool_is_not_an_arbitrary_execution_fallback() -> None:
    result = GenericExecutor(registry()).execute_step(step("s1", "python", inputs={"code": "raise SystemExit"}))
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "unknown_tool"


class NonCallingLegacyResearch:
    name = "web_research"
    risk_level = "low"
    permission_level = "public_read"

    def search(self, query: str, limit: int = 5):
        raise AssertionError("Orchestrator must route through its executor")

    def fetch(self, url: str, max_chars: int = 8000):
        raise AssertionError("Orchestrator must route through its executor")


class StubPlanner:
    def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict]) -> TaskPlan:
        return TaskPlan(goal, [step("research", "web_research", inputs={"query": goal})], "stub plan")


class RecordingExecutor:
    def __init__(self) -> None:
        self.plans: list[TaskPlan] = []

    def execute_plan(self, plan: TaskPlan, context: ExecutionContext) -> list[StepResult]:
        self.plans.append(plan)
        output = {
            "query": plan.goal,
            "sources": [
                {"title": "A", "url": "https://example.com/a", "text": "evidence A", "status": 200},
                {"title": "B", "url": "https://example.com/b", "text": "evidence B", "status": 200},
            ],
            "errors": [],
        }
        return [StepResult.success(plan.steps[0].id, plan.steps[0].tool, output)]


def test_orchestrator_routes_plan_through_injected_generic_executor(tmp_path: Path) -> None:
    tools = ToolRegistry()
    tools.register(NonCallingLegacyResearch())
    executor = RecordingExecutor()
    agent = Orchestrator(Store(tmp_path / "agent.sqlite3"), tools, planner=StubPlanner(), executor=executor)
    result = agent.run("collect evidence")
    assert result["verification"]["verified"] is True
    assert len(executor.plans) == 1
    assert executor.plans[0].steps[0].tool == "web_research"
