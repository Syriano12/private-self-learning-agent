from __future__ import annotations

from pathlib import Path

from private_agent.core.execution import ExecutionContext, GenericExecutor
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.security import SecurityController
from private_agent.skills.lifecycle import (
    MutationReport,
    MutationTestingEngine,
    RuntimeVerificationReport,
    SandboxResult,
    SkillCandidate,
    SkillContract,
    SkillLifecycleManager,
)
from private_agent.skills.registry import SkillRegistry
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class BoundaryTool:
    name = "boundary_tool"
    permission_level = "public_read"
    risk_level = "low"
    required_capabilities: list[str] = []
    declared_capabilities: list[str] = []
    calls = 0

    def description(self) -> str:
        return "tool used to test the execution boundary"

    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        }

    def execute(self, inputs: dict, context: ExecutionContext) -> ToolResult:
        self.calls += 1
        return ToolResult.success({"result": "executed"})


def registry(tool: BoundaryTool) -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(tool)
    return tools


def step() -> PlanStep:
    return PlanStep("boundary-step", "boundary test", "boundary_tool", {}, "test", [])


def test_generic_executor_fails_closed_without_security_controller() -> None:
    tool = BoundaryTool()
    result = GenericExecutor(registry(tool)).execute_step(step(), ExecutionContext(task_id="missing-security"))

    assert result.status == "blocked"
    assert result.error is not None
    assert result.error.code == "security_not_configured"
    assert tool.calls == 0


def test_generic_executor_still_executes_through_security_boundary() -> None:
    tool = BoundaryTool()
    controller = SecurityController()
    result = GenericExecutor(registry(tool), controller).execute_step(
        step(),
        ExecutionContext(task_id="configured-security", security_controller=controller),
    )

    assert result.status == "success"
    assert tool.calls == 1
    assert controller.audit.list(task_id="configured-security")


class SpySandbox:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, candidate, contract, inputs) -> SandboxResult:
        self.calls += 1
        raise AssertionError("sandbox must not execute without SecurityController")


class MinimalImplementation:
    def __call__(self, inputs: dict) -> dict:
        return {"result": "active"}


def make_candidate() -> SkillCandidate:
    return SkillCandidate(
        skill_id="fail-closed-skill",
        name="fail_closed_skill",
        version=1,
        description="boundary test skill",
        source="def run(value):\n    if value > 0:\n        return {'result': 'active'}\n    return {'result': 'inactive'}\n",
        entry_point="run",
        declared_capabilities=[],
        expected_inputs={"type": "object", "properties": {}, "additionalProperties": False},
        expected_outputs={
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        },
        tests=["returns active"],
        metadata={"invariants": ["non_empty_result"]},
        provenance={"source": "phase14a7-test"},
        implementation=MinimalImplementation(),
        mutation_test_runner=lambda source, mutant: mutant.category == "comparison_inversion" and "<= 0" in source,
    )


def test_skill_execution_fails_closed_without_calling_sandbox(tmp_path: Path) -> None:
    store = Store(tmp_path / "skill-boundary.sqlite3")
    controller = SecurityController(store=store)
    lifecycle = SkillLifecycleManager(
        SkillRegistry(store),
        mutation_engine=MutationTestingEngine(max_mutants=1, threshold=1.0),
        security_controller=controller,
    )
    candidate = make_candidate()
    submitted = lifecycle.submit(candidate, inputs={})
    assert submitted.approval.status == "QUARANTINED"
    assert submitted.sandbox.error == "external_sandbox_required"
    lifecycle.registry.persist_candidate(candidate, status="ACTIVE")

    spy = SpySandbox()
    lifecycle.sandbox = spy
    lifecycle.security = None
    result = lifecycle.execute_active("fail_closed_skill", {})

    assert result.status == "BLOCKED"
    assert result.sandbox.error == "security_not_configured"
    assert spy.calls == 0
    store.close()


def test_orchestrator_default_path_has_an_initialized_security_boundary(tmp_path: Path) -> None:
    tool = BoundaryTool()
    store = Store(tmp_path / "orchestrator-boundary.sqlite3")
    planner = type(
        "PlannerStub",
        (),
        {
            "create": lambda self, goal, tools, prior_knowledge: TaskPlan(
                goal,
                [PlanStep("boundary-step", "boundary test", "boundary_tool", {}, "test", [], success_criteria={"required_fields": ["result"]})],
                "boundary plan",
            )
        },
    )()
    agent = Orchestrator(store, registry(tool), planner=planner)

    result = agent.run("orchestrated boundary")

    assert result["final_status"] == "COMPLETED"
    assert tool.calls == 1
    assert agent.security is not None
    store.close()


def test_skill_submit_fails_closed_before_sandbox_when_activation_is_denied(
    tmp_path: Path, monkeypatch,
) -> None:
    store = Store(tmp_path / "skill-submit-boundary.sqlite3")
    controller = SecurityController(store=store)
    lifecycle = SkillLifecycleManager(
        SkillRegistry(store),
        mutation_engine=MutationTestingEngine(max_mutants=1, threshold=1.0),
        security_controller=controller,
    )
    candidate = make_candidate()
    candidate.metadata["required_capabilities"] = ["SYSTEM_COMMAND"]
    candidate.metadata["risk_level"] = "CRITICAL"
    store.create_owner_permission_rule(
        {
            "rule_id": "deny-submit-skill",
            "effect": "DENY",
            "scope_type": "SKILL",
            "scope_key": candidate.name,
            "conditions": {},
            "reason": "skill submit boundary test",
        },
        actor="owner",
        reason="skill submit boundary test",
    )
    spy = SpySandbox()
    lifecycle.sandbox = spy
    passed_mutation = MutationReport(
        True,
        1,
        1,
        0,
        1.0,
        lifecycle.mutation_engine.threshold,
        reason="trusted_static_test_gate",
    )
    monkeypatch.setattr(
        lifecycle.mutation_engine,
        "run",
        lambda _candidate, _runner=None: passed_mutation,
    )

    submitted = lifecycle.submit(candidate, inputs={})

    assert submitted.approval.approved is False
    assert submitted.sandbox.status == "BLOCKED"
    assert submitted.sandbox.error.startswith("security_policy_blocked:")
    assert spy.calls == 0
    store.close()


def test_orchestrator_rejects_untrusted_injected_executor(tmp_path: Path) -> None:
    store = Store(tmp_path / "executor-injection.sqlite3")
    tools = ToolRegistry()

    class UntrustedExecutor:
        def execute_plan(self, plan, context):
            raise AssertionError("untrusted executor must never run")

    try:
        Orchestrator(store, tools, executor=UntrustedExecutor())
    except TypeError as exc:
        assert str(exc) == "unsupported_executor: production execution owns GenericExecutor"
    else:
        raise AssertionError("Orchestrator accepted an untrusted executor")
    store.close()
