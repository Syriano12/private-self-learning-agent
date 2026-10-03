from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from private_agent.core.execution import ExecutionContext
from private_agent.core.orchestrator import Orchestrator, TaskInterrupted
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.core.task_state import InvalidStateTransition, StateIntegrityError, TaskState
from private_agent.security import SecurityController, SecurityPolicy
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class DurableTool:
    permission_level = "public_read"
    risk_level = "low"
    required_capabilities: list[str] = []
    declared_capabilities: list[str] = []
    planned_effect = "controlled durable test action"

    def __init__(self, name: str, outcomes: list[ToolResult] | None = None, *, capabilities: list[str] | None = None, risk: str = "low") -> None:
        self.name = name
        self.outcomes = list(outcomes or [ToolResult.success({"result": name})])
        self.calls = 0
        self.required_capabilities = list(capabilities or [])
        self.declared_capabilities = list(capabilities or [])
        self.risk_level = risk
        self.network_required = "NETWORK_ACCESS" in self.required_capabilities

    def description(self) -> str:
        return f"durable tool {self.name}"

    def input_schema(self) -> dict:
        return {"type": "object", "properties": {"value": {"type": "string"}}, "additionalProperties": False}

    def output_schema(self) -> dict:
        return {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"], "additionalProperties": False}

    def execute(self, inputs: dict, context: ExecutionContext) -> ToolResult:
        self.calls += 1
        return self.outcomes.pop(0) if self.outcomes else ToolResult.success({"result": self.name})


class StubPlanner:
    provider = None
    max_steps = 12

    def __init__(self, steps: list[PlanStep]) -> None:
        self.steps = steps
        self.calls = 0

    def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict]) -> TaskPlan:
        self.calls += 1
        return TaskPlan(goal, [PlanStep(**step.__dict__) for step in self.steps], "durable test plan")


def registry(*items: DurableTool) -> ToolRegistry:
    result = ToolRegistry()
    for item in items:
        result.register(item)
    return result


def step(tool: str = "safe", *, step_id: str = "step-1", value: str = "x", criteria: dict | None = None) -> PlanStep:
    return PlanStep(
        step_id,
        "durable action",
        tool,
        {"value": value},
        "controlled test",
        [],
        success_criteria=criteria or {"required_fields": ["result"]},
    )


def make_agent(path: Path, tool: DurableTool, *, hook=None, security: SecurityController | None = None, planner=None) -> tuple[Store, Orchestrator]:
    store = Store(path)
    agent = Orchestrator(
        store,
        registry(tool),
        planner=planner or StubPlanner([step(tool.name)]),
        security_controller=security,
        checkpoint_hook=hook,
        max_recovery_attempts=2,
    )
    return store, agent


def test_task_creation_and_durable_checkpoints(tmp_path: Path) -> None:
    tool = DurableTool("safe")
    store, agent = make_agent(tmp_path / "creation.sqlite3", tool)
    result = agent.run("persist creation")
    state = store.get_task_state(result["task_id"])
    assert state is not None
    assert state["task_id"] == result["task_id"]
    assert state["goal"] == "persist creation"
    assert state["state_version"] == 1
    assert state["status"] == "COMPLETED"
    assert state["last_successful_checkpoint"] == "TASK_COMPLETED"
    checkpoint_types = [row["checkpoint_type"] for row in store.task_checkpoints(result["task_id"])]
    assert checkpoint_types[:3] == ["TASK_CREATED", "TASK_PLANNING", "PLAN_CREATED"]
    assert "BEFORE_EXECUTION" in checkpoint_types
    assert "AFTER_EXECUTION" in checkpoint_types
    assert "AFTER_OBSERVATION" in checkpoint_types
    assert "AFTER_VERIFICATION" in checkpoint_types
    assert "TASK_COMPLETED" in checkpoint_types
    assert store.get_task_state_row(result["task_id"])["state_fingerprint"] == state["state_fingerprint"]
    store.close()


def test_successful_resume_after_normal_interruption(tmp_path: Path) -> None:
    tool = DurableTool("safe")

    def interrupt(checkpoint: str, state: dict) -> None:
        if checkpoint == "PLAN_CREATED":
            raise TaskInterrupted("normal interruption after plan checkpoint")

    store, first = make_agent(tmp_path / "normal-resume.sqlite3", tool, hook=interrupt)
    with pytest.raises(TaskInterrupted):
        first.run("resume me")
    task_id = store.db.execute("SELECT task_id FROM task_states").fetchone()[0]
    store.close()

    restarted_store = Store(tmp_path / "normal-resume.sqlite3")
    restarted = Orchestrator(restarted_store, registry(tool), planner=StubPlanner([step("safe")]))
    result = restarted.resume_task(task_id)
    assert result["final_status"] == "COMPLETED"
    assert tool.calls == 1
    assert restarted_store.get_task_state(task_id)["status"] == "COMPLETED"
    restarted_store.close()


def test_resume_after_crash_like_interruption_fails_closed_as_execution_unknown(tmp_path: Path) -> None:
    tool = DurableTool("safe")

    def interrupt(checkpoint: str, state: dict) -> None:
        if checkpoint == "BEFORE_EXECUTION":
            raise TaskInterrupted("crash before tool returned")

    store, first = make_agent(tmp_path / "unknown.sqlite3", tool, hook=interrupt)
    with pytest.raises(TaskInterrupted):
        first.run("unknown execution")
    task_id = store.db.execute("SELECT task_id FROM task_states").fetchone()[0]
    raw = store.get_task_state(task_id)
    assert raw["status"] == "EXECUTING"
    assert any(record["status"] == "EXECUTING" for record in raw["action_records"].values())
    store.close()

    restarted_store = Store(tmp_path / "unknown.sqlite3")
    restarted = Orchestrator(restarted_store, registry(tool), planner=StubPlanner([step("safe")]))
    result = restarted.resume_task(task_id)
    assert result["final_status"] == "BLOCKED"
    assert "ambiguous_execution_state_requires_observation" in result["errors"]
    assert tool.calls == 0
    assert restarted_store.get_task_state(task_id)["status"] == "EXECUTION_UNKNOWN"
    assert any(row["checkpoint_type"] == "EXECUTION_UNKNOWN" for row in restarted_store.task_checkpoints(task_id))
    restarted_store.close()


def test_execution_unknown_can_only_retry_after_explicit_not_executed_observation(tmp_path: Path) -> None:
    tool = DurableTool("safe")

    def interrupt(checkpoint: str, state: dict) -> None:
        if checkpoint == "BEFORE_EXECUTION":
            raise TaskInterrupted("interrupted")

    store, first = make_agent(tmp_path / "unknown-retry.sqlite3", tool, hook=interrupt)
    with pytest.raises(TaskInterrupted):
        first.run("safe retry")
    task_id = store.db.execute("SELECT task_id FROM task_states").fetchone()[0]
    key = next(iter(store.get_task_state(task_id)["action_records"]))
    store.close()
    restarted_store = Store(tmp_path / "unknown-retry.sqlite3")
    restarted = Orchestrator(restarted_store, registry(tool), planner=StubPlanner([step("safe")]))
    blocked = restarted.resume_task(task_id)
    assert blocked["final_status"] == "BLOCKED"
    assert restarted_store.get_task_state(task_id)["status"] == "EXECUTION_UNKNOWN"
    assert tool.calls == 0
    result = restarted.resume_task(task_id, unknown_resolutions={key: {"decision": "retry", "observed_not_executed": True}})
    assert result["final_status"] == "COMPLETED"
    assert tool.calls == 1
    restarted_store.close()


def test_verified_action_is_not_executed_twice_after_resume(tmp_path: Path) -> None:
    tool = DurableTool("safe")
    store, agent = make_agent(tmp_path / "duplicate.sqlite3", tool)
    result = agent.run("no duplicate")
    assert tool.calls == 1
    resumed = agent.resume_task(result["task_id"])
    assert resumed["final_status"] == "COMPLETED"
    assert tool.calls == 1
    state = TaskState.from_dict(store.get_task_state(result["task_id"]))
    assert len(state.completed_actions) == 1
    assert all(record["status"] == "VERIFIED" for record in state.action_records.values())
    store.close()


def test_state_fingerprint_detects_stale_mutation(tmp_path: Path) -> None:
    tool = DurableTool("safe")
    store, agent = make_agent(tmp_path / "stale.sqlite3", tool)
    with pytest.raises(TaskInterrupted):
        agent.checkpoint_hook = lambda checkpoint, state: (_ for _ in ()).throw(TaskInterrupted()) if checkpoint == "PLAN_CREATED" else None
        agent.run("stale")
    task_id = store.db.execute("SELECT task_id FROM task_states").fetchone()[0]
    raw = store.get_task_state(task_id)
    raw["goal"] = "tampered goal"
    store.db.execute("UPDATE task_states SET state_json = ? WHERE task_id = ?", (json.dumps(raw), task_id))
    store.db.commit()
    result = agent.resume_task(task_id)
    assert result["final_status"] == "BLOCKED"
    assert any("state_fingerprint_mismatch" in error for error in result["errors"])
    store.close()


def test_resume_continues_from_last_verified_step_without_reexecuting_it(tmp_path: Path) -> None:
    first_tool = DurableTool("first")
    second_tool = DurableTool("second")
    path = tmp_path / "multi-step-resume.sqlite3"

    def interrupt(checkpoint: str, state: dict) -> None:
        if checkpoint == "AFTER_VERIFICATION" and state.get("current_step") == "first-step":
            raise TaskInterrupted("interrupted between verified steps")

    store = Store(path)
    agent = Orchestrator(
        store,
        registry(first_tool, second_tool),
        planner=StubPlanner([
            step("first", step_id="first-step"),
            step("second", step_id="second-step"),
        ]),
        checkpoint_hook=interrupt,
    )
    with pytest.raises(TaskInterrupted):
        agent.run("two-step resume")
    task_id = store.db.execute("SELECT task_id FROM task_states").fetchone()[0]
    state_before = TaskState.from_dict(store.get_task_state(task_id))
    assert len(state_before.completed_actions) == 1
    first_key = state_before.completed_actions[0]
    store.close()

    restarted_store = Store(path)
    restarted = Orchestrator(
        restarted_store,
        registry(first_tool, second_tool),
        planner=StubPlanner([
            step("first", step_id="first-step"),
            step("second", step_id="second-step"),
        ]),
    )
    result = restarted.resume_task(task_id)
    assert result["final_status"] == "COMPLETED"
    assert first_tool.calls == 1
    assert second_tool.calls == 1
    state_after = TaskState.from_dict(restarted_store.get_task_state(task_id))
    assert first_key in state_after.completed_actions
    assert len(state_after.completed_actions) == 2
    restarted_store.close()


def test_corrupted_state_and_unknown_schema_fail_closed(tmp_path: Path) -> None:
    tool = DurableTool("safe")
    store, agent = make_agent(tmp_path / "corrupt.sqlite3", tool)
    with pytest.raises(TaskInterrupted):
        agent.checkpoint_hook = lambda checkpoint, state: (_ for _ in ()).throw(TaskInterrupted()) if checkpoint == "PLAN_CREATED" else None
        agent.run("corrupt")
    task_id = store.db.execute("SELECT task_id FROM task_states").fetchone()[0]
    store.db.execute("UPDATE task_states SET state_json = ? WHERE task_id = ?", ("{broken", task_id))
    store.db.commit()
    assert agent.resume_task(task_id)["final_status"] == "BLOCKED"
    store.db.execute("UPDATE task_states SET state_json = ? WHERE task_id = ?", ("{broken", task_id))
    store.db.commit()
    store.close()

    store2, agent2 = make_agent(tmp_path / "version.sqlite3", DurableTool("safe"))
    with pytest.raises(TaskInterrupted):
        agent2.checkpoint_hook = lambda checkpoint, state: (_ for _ in ()).throw(TaskInterrupted()) if checkpoint == "PLAN_CREATED" else None
        agent2.run("version")
    task2 = store2.db.execute("SELECT task_id FROM task_states").fetchone()[0]
    raw = store2.get_task_state(task2)
    raw["state_version"] = 999
    store2.db.execute("UPDATE task_states SET state_json = ? WHERE task_id = ?", (json.dumps(raw), task2))
    store2.db.commit()
    result = agent2.resume_task(task2)
    assert result["final_status"] == "BLOCKED"
    assert any("unsupported_state_version" in error for error in result["errors"])
    store2.close()


def _high_risk_setup(path: Path, *, tool_name: str = "network") -> tuple[Store, SecurityController, Orchestrator, DurableTool]:
    tool = DurableTool(tool_name, capabilities=["NETWORK_ACCESS"], risk="high")
    store = Store(path)
    security = SecurityController(store=store)
    planner = StubPlanner([step(tool_name)])
    agent = Orchestrator(store, registry(tool), planner=planner, security_controller=security)
    return store, security, agent, tool


def test_approval_state_persists_pending_across_restart(tmp_path: Path) -> None:
    path = tmp_path / "pending-approval.sqlite3"
    store, security, agent, tool = _high_risk_setup(path)
    first = agent.run("pending approval", task_id="pending-task")
    approval_id = first["approval_requests"][0]["approval_id"]
    assert first["final_status"] == "BLOCKED"
    assert store.get_task_state("pending-task")["status"] == "WAITING_APPROVAL"
    store.close()

    restarted_store = Store(path)
    restarted_security = SecurityController(store=restarted_store)
    restarted = Orchestrator(restarted_store, registry(tool), planner=StubPlanner([step(tool.name)]), security_controller=restarted_security)
    result = restarted.resume_task("pending-task")
    assert result["final_status"] == "BLOCKED"
    assert result["approval_requests"][0]["approval_id"] == approval_id
    assert restarted_security.approval_gate.get(approval_id).status == "PENDING"
    assert tool.calls == 0
    restarted_store.close()


def test_approved_high_risk_action_resumes_only_with_valid_persisted_approval(tmp_path: Path) -> None:
    path = tmp_path / "approved-restart.sqlite3"
    store, security, agent, tool = _high_risk_setup(path)
    first = agent.run("approved resume", task_id="approved-task")
    approval_id = first["approval_requests"][0]["approval_id"]
    security.approval_gate.approve(approval_id)
    store.close()

    restarted_store = Store(path)
    restarted = Orchestrator(restarted_store, registry(tool), planner=StubPlanner([step(tool.name)]), security_controller=SecurityController(store=restarted_store))
    result = restarted.resume_task("approved-task", approval_ids={"step-1": approval_id})
    assert result["final_status"] == "COMPLETED"
    assert tool.calls == 1
    assert any(event["event_type"] == "APPROVAL_REVALIDATED" for event in restarted.security.audit.list(task_id="approved-task"))
    restarted_store.close()


def test_expired_approval_after_restart_is_invalidated(tmp_path: Path) -> None:
    path = tmp_path / "expired-restart.sqlite3"
    store, security, agent, tool = _high_risk_setup(path)
    first = agent.run("expired approval", task_id="expired-task")
    approval_id = first["approval_requests"][0]["approval_id"]
    request = security.approval_gate.get(approval_id)
    request.expires_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    store.save_approval_request(request.to_dict())
    store.close()

    restarted_store = Store(path)
    restarted_security = SecurityController(store=restarted_store)
    restarted = Orchestrator(restarted_store, registry(tool), planner=StubPlanner([step(tool.name)]), security_controller=restarted_security)
    result = restarted.resume_task("expired-task")
    assert result["final_status"] == "BLOCKED"
    assert restarted_security.approval_gate.get(approval_id).status == "EXPIRED"
    assert any(event["event_type"] == "APPROVAL_INVALIDATED" for event in restarted_security.audit.list(task_id="expired-task"))
    assert tool.calls == 0
    restarted_store.close()


def test_policy_version_change_after_restart_invalidates_old_approval(tmp_path: Path) -> None:
    path = tmp_path / "policy-restart.sqlite3"
    store, security, agent, tool = _high_risk_setup(path)
    first = agent.run("policy changed", task_id="policy-task")
    approval_id = first["approval_requests"][0]["approval_id"]
    security.approval_gate.approve(approval_id)
    security.update_policy(SecurityPolicy(policy_version=2))
    store.close()

    restarted_store = Store(path)
    restarted_security = SecurityController(store=restarted_store)
    restarted = Orchestrator(restarted_store, registry(tool), planner=StubPlanner([step(tool.name)]), security_controller=restarted_security)
    result = restarted.resume_task("policy-task", approval_ids={"step-1": approval_id})
    assert result["final_status"] == "BLOCKED"
    assert restarted_security.policy_engine.policy.policy_version == 2
    assert tool.calls == 0
    restarted_store.close()


def test_capability_and_action_fingerprint_change_after_restart_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "binding-restart.sqlite3"
    store, security, agent, tool = _high_risk_setup(path)
    first = agent.run("binding changed", task_id="binding-task")
    approval_id = first["approval_requests"][0]["approval_id"]
    security.approval_gate.approve(approval_id)
    state = TaskState.from_dict(store.get_task_state("binding-task"))
    state.plan["steps"][0]["input"]["value"] = "changed"
    state.set_plan(state.plan, version=state.plan_version + 1)
    state.checkpoint("PLAN_UPDATED", status="READY", phase="PLAN_UPDATED")
    store.save_task_state(state, "PLAN_UPDATED")
    store.close()

    tool.required_capabilities = ["EXTERNAL_API"]
    tool.declared_capabilities = ["EXTERNAL_API"]
    restarted_store = Store(path)
    restarted_security = SecurityController(store=restarted_store)
    restarted = Orchestrator(restarted_store, registry(tool), planner=StubPlanner([step(tool.name)]), security_controller=restarted_security)
    result = restarted.resume_task("binding-task", approval_ids={"step-1": approval_id})
    assert result["final_status"] == "BLOCKED"
    assert tool.calls == 0
    assert any(event["event_type"] == "APPROVAL_INVALIDATED" for event in restarted_security.audit.list(task_id="binding-task"))
    restarted_store.close()


def test_low_risk_resume_continues_after_restart_and_recovery_replanning_is_durable(tmp_path: Path) -> None:
    path = tmp_path / "low-recovery.sqlite3"
    tool_a = DurableTool("a", [ToolResult.failed("tool_failed", "temporary")])
    tool_b = DurableTool("b", [ToolResult.success({"result": "recovered"})])

    def interrupt(checkpoint: str, state: dict) -> None:
        if checkpoint == "AFTER_RECOVERY":
            raise TaskInterrupted("restart after replanning")

    store = Store(path)
    planner = StubPlanner([step("a")])
    agent = Orchestrator(store, registry(tool_a, tool_b), planner=planner, checkpoint_hook=interrupt, max_recovery_attempts=2)
    with pytest.raises(TaskInterrupted):
        agent.run("recovery resume")
    task_id = store.db.execute("SELECT task_id FROM task_states").fetchone()[0]
    persisted = store.get_task_state(task_id)
    assert persisted["plan_version"] == 2
    assert persisted["plan"]["steps"][0]["tool"] == "b"
    store.close()

    restarted_store = Store(path)
    restarted = Orchestrator(restarted_store, registry(tool_a, tool_b), planner=StubPlanner([step("a")]), max_recovery_attempts=2)
    result = restarted.resume_task(task_id)
    assert result["final_status"] == "COMPLETED"
    assert tool_a.calls == 1
    assert tool_b.calls == 1
    assert any(row["checkpoint_type"] == "AFTER_RECOVERY" for row in restarted_store.task_checkpoints(task_id))
    restarted_store.close()


def test_atomic_checkpoint_rows_are_self_consistent_and_auditable(tmp_path: Path) -> None:
    tool = DurableTool("safe")
    store, agent = make_agent(tmp_path / "atomic.sqlite3", tool)
    result = agent.run("atomic checkpoints")
    checkpoints = store.task_checkpoints(result["task_id"])
    assert checkpoints
    for checkpoint in checkpoints:
        payload = json.loads(checkpoint["state_json"])
        assert payload["state_fingerprint"] == checkpoint["state_fingerprint"]
        assert payload["state_version"] == checkpoint["state_version"] == 1
    audit_types = {event["event_type"] for event in agent.security.audit.list(task_id=result["task_id"])} if agent.security else set()
    assert not audit_types or "CHECKPOINT_SAVED" in audit_types
    store.close()


def test_security_audit_contains_task_lifecycle_events(tmp_path: Path) -> None:
    tool = DurableTool("safe")
    store = Store(tmp_path / "audit-lifecycle.sqlite3")
    security = SecurityController(store=store)
    agent = Orchestrator(
        store,
        registry(tool),
        planner=StubPlanner([step("safe")]),
        security_controller=security,
    )
    result = agent.run("audit lifecycle")
    agent.resume_task(result["task_id"])
    event_types = {event["event_type"] for event in security.audit.list(task_id=result["task_id"])}
    assert {"TASK_CREATED", "CHECKPOINT_SAVED", "TASK_COMPLETED", "TASK_RESUMED"}.issubset(event_types)
    assert "task-state" in {event["tool_or_skill"] for event in security.audit.list(task_id=result["task_id"])}
    store.close()


def test_invalid_transition_and_missing_verification_fail_closed() -> None:
    state = TaskState.create("task", "goal")
    with pytest.raises(InvalidStateTransition):
        state.transition("COMPLETED")
    state.set_plan({"goal": "goal", "rationale": "r", "steps": [{"id": "s", "tool": "t", "input": {}}]}, version=1)
    state.status = "COMPLETED"
    state.touch()
    with pytest.raises(StateIntegrityError, match="completed_without_persisted_verification"):
        state.validate()
