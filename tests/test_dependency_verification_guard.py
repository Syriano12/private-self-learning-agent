from __future__ import annotations

from pathlib import Path

import pytest

from private_agent.core.execution import ExecutionContext, GenericExecutor, StepResult
from private_agent.core.orchestrator import Orchestrator, TaskInterrupted
from private_agent.core.planner import PlanStep, Planner, TaskPlan
from private_agent.core.recovery import RecoveryManager
from private_agent.core.task_state import TaskState
from private_agent.security import SecurityController
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class RegisteredProbeTool:
    permission_level = "public_read"
    risk_level = "low"

    def __init__(self, name: str, *, required_inputs: list[str] | None = None, malformed: bool = False) -> None:
        self.name = name
        self.required_inputs = required_inputs or []
        self.malformed = malformed
        self.calls: list[dict] = []

    def description(self) -> str:
        return f"Benign registered probe {self.name}"

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {key: {"type": "string"} for key in self.required_inputs},
            "required": self.required_inputs,
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
        self.calls.append(dict(inputs))
        if self.malformed:
            return {"result": "not a ToolResult"}
        return ToolResult.success({"result": f"{self.name}:{inputs}"})


class StaticPlanner:
    max_steps = 12

    def __init__(self, plan: TaskPlan) -> None:
        self.plan = plan

    def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict]) -> TaskPlan:
        return self.plan


class StaticReplanner:
    provider = object()

    def __init__(self, plan: TaskPlan) -> None:
        self.plan = plan

    def replan(self, request, tools: ToolRegistry, *, replacement_tool: str = "") -> TaskPlan:
        return self.plan


class RecoveryProbeTool(RegisteredProbeTool):
    def __init__(self, name: str, result: str, *, fail_first: bool = False) -> None:
        super().__init__(name)
        self.result = result
        self.fail_first = fail_first
        self.previous_results_seen: list[dict] = []

    def execute(self, inputs: dict, context: ExecutionContext):
        self.calls.append(dict(inputs))
        self.previous_results_seen.append(
            {key: result.to_dict() for key, result in context.previous_results.items() if isinstance(result, StepResult)}
        )
        if self.fail_first:
            self.fail_first = False
            return ToolResult.failed("tool_failed", "synthetic first-attempt failure")
        return ToolResult.success({"result": self.result})


def make_registry(*tools) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def make_raw_plan() -> dict:
    return {
        "goal": "prove verified dependency handling",
        "rationale": "synthetic local regression fixture",
        "steps": [
            {
                "id": "source",
                "objective": "produce source evidence",
                "tool": "repeat",
                "input": {"role": "source"},
                "reason": "registered benign fixture",
                "depends_on": [],
                "success_criteria": {"required_fields": ["result"]},
            },
            {
                "id": "consumer",
                "objective": "consume the later dependency",
                "tool": "repeat",
                "input": {"role": "consumer"},
                "reason": "registered benign fixture",
                "depends_on": ["later"],
                "success_criteria": {"required_fields": ["result"]},
            },
            {
                "id": "later",
                "objective": "produce the actual later result",
                "tool": "later_tool",
                "input": {},
                "reason": "registered benign fixture",
                "depends_on": [],
                "success_criteria": {"required_fields": ["result"]},
            },
        ],
    }


class RepeatedToolProbe(RegisteredProbeTool):
    def __init__(self, store: Store, call_order: list[tuple[str, str]]) -> None:
        super().__init__("repeat", required_inputs=["role"])
        self.store = store
        self.call_order = call_order
        self.consumer_snapshot: dict | None = None

    def execute(self, inputs: dict, context: ExecutionContext) -> ToolResult:
        role = inputs["role"]
        self.calls.append(dict(inputs))
        self.call_order.append((self.name, role))
        if role == "source":
            # Deliberately try to plant a future dependency in this action's mutable context.
            context.previous_results["later"] = StepResult.success(
                "later", "later_tool", {"result": "FORGED_BEFORE_VERIFICATION"}
            )
            return ToolResult.success({"result": "ACTUAL-SOURCE"})
        if role == "consumer":
            previous = {
                key: value.to_dict()
                for key, value in context.previous_results.items()
                if isinstance(value, StepResult)
            }
            state = self.store.get_task_state(context.task_id)
            records = list((state or {}).get("action_records", {}).values())
            later_record = next(record for record in records if record.get("step_id") == "later")
            events = self.store.events_for_task(context.task_id)
            self.consumer_snapshot = {
                "previous_results": previous,
                "later_action_status": later_record.get("status"),
                "later_execution_status": later_record.get("execution_status"),
                "later_verification_status": (later_record.get("verification") or {}).get("status"),
                "later_step_completed_seen": any(
                    event.get("event_type") == "step_completed" and event.get("step_id") == "later"
                    for event in events
                ),
            }
            return ToolResult.success({"result": "CONSUMER-RAN"})
        raise AssertionError(f"unexpected test role: {role}")


class LaterToolProbe(RegisteredProbeTool):
    def __init__(self, call_order: list[tuple[str, str]]) -> None:
        super().__init__("later_tool")
        self.call_order = call_order
        self.previous_results: dict = {}

    def execute(self, inputs: dict, context: ExecutionContext) -> ToolResult:
        self.calls.append(dict(inputs))
        self.call_order.append((self.name, "actual"))
        self.previous_results = {
            key: value.to_dict()
            for key, value in context.previous_results.items()
            if isinstance(value, StepResult)
        }
        # Attempt to mutate an earlier action's nested output. The later consumer
        # must still receive the authoritative copy held by the orchestrator.
        if "source" in context.previous_results:
            context.previous_results["source"].output["result"] = "MUTATED-BY-LATER-TOOL"
        return ToolResult.success({"result": "ACTUAL-LATER"})


def test_orchestrator_isolates_context_and_orders_forward_dependency_with_repeated_tool_names(tmp_path: Path) -> None:
    store = Store(tmp_path / "dependency-guard.sqlite3")
    call_order: list[tuple[str, str]] = []
    repeat = RepeatedToolProbe(store, call_order)
    later = LaterToolProbe(call_order)
    tools = make_registry(repeat, later)
    plan = Planner().validate(make_raw_plan(), goal="prove verified dependency handling", tools=tools)

    # Planner currently accepts a forward reference; execution must still order it safely.
    assert [step.id for step in plan.steps] == ["source", "consumer", "later"]
    agent = Orchestrator(
        store,
        tools,
        planner=StaticPlanner(plan),
        recovery_manager=RecoveryManager(max_recovery_attempts=0, max_attempts_per_step=0),
    )
    result = agent.run("prove verified dependency handling")

    assert result["final_status"] == "COMPLETED"
    assert call_order == [("repeat", "source"), ("later_tool", "actual"), ("repeat", "consumer")]
    assert set(later.previous_results) == {"source"}
    assert later.previous_results["source"]["step_id"] == "source"
    assert later.previous_results["source"]["output"] == {"result": "ACTUAL-SOURCE"}

    snapshot = repeat.consumer_snapshot
    assert snapshot is not None
    assert set(snapshot["previous_results"]) == {"source", "later"}
    assert snapshot["previous_results"]["source"]["step_id"] == "source"
    assert snapshot["previous_results"]["source"]["tool_name"] == "repeat"
    assert snapshot["previous_results"]["source"]["output"] == {"result": "ACTUAL-SOURCE"}
    assert snapshot["previous_results"]["later"]["step_id"] == "later"
    assert snapshot["previous_results"]["later"]["tool_name"] == "later_tool"
    assert snapshot["previous_results"]["later"]["output"] == {"result": "ACTUAL-LATER"}
    assert snapshot["later_action_status"] == "VERIFIED"
    assert snapshot["later_execution_status"] == "success"
    assert snapshot["later_verification_status"] == "VERIFIED"
    assert snapshot["later_step_completed_seen"] is True

    persisted = TaskState.from_dict(store.get_task_state(result["task_id"]))
    source_step = next(step for step in plan.steps if step.id == "source")
    verified = agent._verified_result_from_action_record(persisted, source_step)
    assert verified is not None and verified.step_id == "source"

    unverified_state = TaskState.from_dict(persisted.to_dict())
    source_key = unverified_state.action_key_for_step(source_step)
    unverified_state.action_records[source_key]["verification"]["status"] = "FAILED"
    unverified_state.touch()
    assert agent._verified_result_from_action_record(unverified_state, source_step) is None

    mismatched_state = TaskState.from_dict(persisted.to_dict())
    mismatched_state.action_records[source_key]["output"]["step_id"] = "different-step"
    mismatched_state.touch()
    assert agent._verified_result_from_action_record(mismatched_state, source_step) is None


def test_resume_rehydrates_verified_result_before_forward_dependent_action(tmp_path: Path) -> None:
    path = tmp_path / "dependency-resume.sqlite3"
    store = Store(path)
    call_order: list[tuple[str, str]] = []
    repeat = RepeatedToolProbe(store, call_order)
    later = LaterToolProbe(call_order)
    tools = make_registry(repeat, later)
    plan = Planner().validate(make_raw_plan(), goal="prove verified dependency handling", tools=tools)

    def interrupt_after_source_verification(checkpoint: str, state: dict) -> None:
        if checkpoint == "AFTER_VERIFICATION" and state.get("current_step") == "source":
            raise TaskInterrupted("restart after source verification")

    first_run = Orchestrator(
        store,
        tools,
        planner=StaticPlanner(plan),
        recovery_manager=RecoveryManager(max_recovery_attempts=0, max_attempts_per_step=0),
        checkpoint_hook=interrupt_after_source_verification,
    )
    with pytest.raises(TaskInterrupted, match="after source verification"):
        first_run.run("prove verified dependency handling")
    task_id = store.db.execute("SELECT task_id FROM task_states").fetchone()[0]
    interrupted_state = store.get_task_state(task_id)
    source_record = next(
        record for record in interrupted_state["action_records"].values()
        if record.get("step_id") == "source"
    )
    assert source_record["status"] == "VERIFIED"
    assert source_record["verification"]["verified"] is True
    store.close()

    restarted_store = Store(path)
    repeat.store = restarted_store
    restarted = Orchestrator(
        restarted_store,
        tools,
        planner=StaticPlanner(plan),
        recovery_manager=RecoveryManager(max_recovery_attempts=0, max_attempts_per_step=0),
    )
    result = restarted.resume_task(task_id)

    assert result["final_status"] == "COMPLETED"
    assert call_order == [("repeat", "source"), ("later_tool", "actual"), ("repeat", "consumer")]
    assert [call["role"] for call in repeat.calls] == ["source", "consumer"]
    assert later.calls == [{}]
    assert repeat.consumer_snapshot is not None
    assert repeat.consumer_snapshot["previous_results"]["source"]["output"] == {"result": "ACTUAL-SOURCE"}
    assert repeat.consumer_snapshot["previous_results"]["later"]["output"] == {"result": "ACTUAL-LATER"}
    assert repeat.consumer_snapshot["later_action_status"] == "VERIFIED"
    assert repeat.consumer_snapshot["later_verification_status"] == "VERIFIED"
    assert repeat.consumer_snapshot["later_step_completed_seen"] is True
    restarted_store.close()


def test_recovery_replan_discards_old_verified_result_before_new_predecessor(tmp_path: Path) -> None:
    old_source = RecoveryProbeTool("source_old", "OLD-SOURCE")
    fail_once = RecoveryProbeTool("fail_once", "UNUSED", fail_first=True)
    new_source = RecoveryProbeTool("source_new", "NEW-SOURCE")
    replacement = RecoveryProbeTool("replacement", "RECOVERED")
    consumer = RecoveryProbeTool("consumer", "CONSUMED")
    tools = make_registry(old_source, fail_once, new_source, replacement, consumer)

    original = Planner().validate(
        {
            "goal": "replace a failed step and reverify dependencies",
            "rationale": "synthetic recovery test",
            "steps": [
                {"id": "source", "objective": "old source", "tool": "source_old", "input": {}, "reason": "fixture", "depends_on": [], "success_criteria": {"required_fields": ["result"]}},
                {"id": "failure", "objective": "fail once", "tool": "fail_once", "input": {}, "reason": "fixture", "depends_on": ["source"], "success_criteria": {"required_fields": ["result"]}},
                {"id": "consumer", "objective": "consume current results", "tool": "consumer", "input": {}, "reason": "fixture", "depends_on": ["source", "failure"], "success_criteria": {"required_fields": ["result"]}},
            ],
        },
        goal="replace a failed step and reverify dependencies",
        tools=tools,
    )
    recovered = Planner().validate(
        {
            "goal": "replace a failed step and reverify dependencies",
            "rationale": "synthetic recovery replacement plan",
            "steps": [
                {"id": "consumer", "objective": "consume current results", "tool": "consumer", "input": {}, "reason": "fixture", "depends_on": ["source", "failure"], "success_criteria": {"required_fields": ["result"]}},
                {"id": "source", "objective": "new source", "tool": "source_new", "input": {}, "reason": "fixture", "depends_on": [], "success_criteria": {"required_fields": ["result"]}},
                {"id": "failure", "objective": "recovered action", "tool": "replacement", "input": {}, "reason": "fixture", "depends_on": ["source"], "success_criteria": {"required_fields": ["result"]}},
            ],
        },
        goal="replace a failed step and reverify dependencies",
        tools=tools,
    )
    store = Store(tmp_path / "recovery-dependency.sqlite3")
    agent = Orchestrator(
        store,
        tools,
        planner=StaticPlanner(original),
        replanner=StaticReplanner(recovered),
        recovery_manager=RecoveryManager(max_recovery_attempts=1, max_attempts_per_step=2),
    )

    result = agent.run("replace a failed step and reverify dependencies")

    assert result["final_status"] == "COMPLETED"
    assert old_source.calls == [{}]
    assert fail_once.calls == [{}]
    assert new_source.calls == [{}]
    assert new_source.previous_results_seen == [{}]
    assert replacement.calls == [{}]
    assert consumer.previous_results_seen[-1]["source"]["output"] == {"result": "NEW-SOURCE"}
    assert consumer.previous_results_seen[-1]["failure"]["output"] == {"result": "RECOVERED"}
    store.close()


def test_dependency_result_identity_mismatch_blocks_registered_dependent() -> None:
    target = RegisteredProbeTool("target")
    security = SecurityController()
    context = ExecutionContext(
        previous_results={
            "expected-source": StepResult.success(
                "different-source", "registered-upstream", {"result": "forged"}
            )
        },
        security_controller=security,
    )
    dependent = PlanStep(
        id="dependent",
        objective="must not run",
        tool="target",
        input={},
        reason="synthetic identity check",
        depends_on=["expected-source"],
    )

    result = GenericExecutor(make_registry(target), security).execute_step(dependent, context)

    assert result.status == "blocked"
    assert result.error is not None
    assert result.error.code == "dependency_identity_mismatch"
    assert target.calls == []


def test_malformed_tool_result_cannot_enable_a_dependent_step() -> None:
    malformed = RegisteredProbeTool("malformed", malformed=True)
    dependent = RegisteredProbeTool("dependent")
    security = SecurityController()
    plan = TaskPlan(
        "malformed dependency control",
        [
            PlanStep("source", "malformed output", "malformed", {}, "synthetic malformed output"),
            PlanStep("child", "dependent action", "dependent", {}, "must be blocked", depends_on=["source"]),
        ],
        "bounded local regression fixture",
    )

    results = GenericExecutor(make_registry(malformed, dependent), security).execute_plan(
        plan, ExecutionContext(security_controller=security)
    )

    assert results[0].status == "failed"
    assert results[0].error is not None and results[0].error.code == "malformed_tool_result"
    assert results[1].status == "blocked"
    assert results[1].error is not None and results[1].error.code == "dependency_failed"
    assert dependent.calls == []
