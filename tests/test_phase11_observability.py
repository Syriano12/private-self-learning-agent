from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from private_agent.core.execution import ExecutionContext, GenericExecutor
from private_agent.core.orchestrator import Orchestrator, TaskInterrupted
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.observability import (
    EVENT_SCHEMA_VERSION,
    ObservabilityMonitor,
    ObservabilityStorageError,
    ObservabilityValidationError,
    StructuredEvent,
)
from private_agent.security import SecurityController
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class DemoTool:
    permission_level = "public_read"
    risk_level = "low"
    required_capabilities: list[str] = []
    declared_capabilities: list[str] = []
    network_required = False
    planned_effect = "controlled observability test"

    def __init__(self, name: str, outcomes: list[ToolResult] | None = None, *, capabilities: list[str] | None = None, risk: str = "low") -> None:
        self.name = name
        self.outcomes = list(outcomes or [ToolResult.success({"result": name})])
        self.calls = 0
        self.required_capabilities = list(capabilities or [])
        self.declared_capabilities = list(capabilities or [])
        self.risk_level = risk
        self.network_required = "NETWORK_ACCESS" in self.required_capabilities

    def description(self) -> str:
        return f"demo {self.name}"

    def input_schema(self) -> dict:
        return {"type": "object", "properties": {"value": {"type": "string"}}, "additionalProperties": False}

    def output_schema(self) -> dict:
        return {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"], "additionalProperties": False}

    def execute(self, inputs: dict, context: ExecutionContext) -> ToolResult:
        self.calls += 1
        return self.outcomes.pop(0) if self.outcomes else ToolResult.success({"result": self.name})


class PlannerStub:
    provider = None
    max_steps = 12

    def __init__(self, steps: list[PlanStep]) -> None:
        self.steps = steps

    def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict]) -> TaskPlan:
        return TaskPlan(goal, [PlanStep(**step.__dict__) for step in self.steps], "observability test plan")


def registry(*tools: DemoTool) -> ToolRegistry:
    result = ToolRegistry()
    for tool in tools:
        result.register(tool)
    return result


def make_step(tool: str, step_id: str = "step-1") -> PlanStep:
    return PlanStep(
        step_id,
        "observe a controlled action",
        tool,
        {"value": "safe"},
        "observability test",
        [],
        success_criteria={"required_fields": ["result"]},
    )


def build_agent(path: Path, tool: DemoTool, *, monitor: ObservabilityMonitor | None = None, security=None, hook=None) -> tuple[Store, ObservabilityMonitor, Orchestrator]:
    store = Store(path)
    monitor = monitor or ObservabilityMonitor(store)
    agent = Orchestrator(
        store,
        registry(tool),
        planner=PlannerStub([make_step(tool.name)]),
        security_controller=security,
        observability=monitor,
        checkpoint_hook=hook,
        max_recovery_attempts=2,
    )
    return store, monitor, agent


def test_structured_event_creation_contains_required_schema() -> None:
    event = StructuredEvent(
        event_type="ACTION_STARTED",
        component="test",
        task_id="task-1",
        action_id="action-1",
        step_id="step-1",
        metadata={"safe": True},
    )
    payload = event.to_dict()
    assert payload["event_id"]
    assert payload["timestamp"]
    assert payload["correlation_id"] == "task:task-1"
    assert payload["schema_version"] == EVENT_SCHEMA_VERSION
    assert payload["metadata"] == {"safe": True}


def test_event_schema_validation_rejects_invalid_type_version_and_shape() -> None:
    with pytest.raises(ObservabilityValidationError):
        StructuredEvent(event_type="arbitrary free form log", component="test")
    with pytest.raises(ObservabilityValidationError):
        StructuredEvent(event_type="ACTION_STARTED", component="test", schema_version=999)
    with pytest.raises(ObservabilityValidationError):
        StructuredEvent.from_dict({"event_type": "ACTION_STARTED"})


def test_event_ordering_is_sequence_based_even_when_timestamps_match(tmp_path: Path) -> None:
    store = Store(tmp_path / "ordering.sqlite3")
    monitor = ObservabilityMonitor(store)
    first = monitor.emit("TASK_CREATED", task_id="t", metadata={"n": 1}, event_id="same-1")
    second = monitor.emit("PLAN_CREATED", task_id="t", metadata={"n": 2}, event_id="same-2")
    assert first is not None and second is not None
    timeline = monitor.get_task_timeline("t")
    assert [event["event_id"] for event in timeline] == ["same-1", "same-2"]
    assert timeline[0]["sequence"] < timeline[1]["sequence"]
    store.close()


def test_task_and_action_correlation_are_explicit(tmp_path: Path) -> None:
    store = Store(tmp_path / "correlation.sqlite3")
    monitor = ObservabilityMonitor(store)
    event = monitor.emit("ACTION_STARTED", task_id="task-x", action_id="action-x", step_id="step-x")
    assert event is not None
    persisted = monitor.get_task_timeline("task-x")[0]
    assert persisted["task_id"] == "task-x"
    assert persisted["action_id"] == "action-x"
    assert persisted["step_id"] == "step-x"
    assert persisted["correlation_id"] == "task:task-x"
    store.close()


def test_task_timeline_reconstructs_complete_runtime(tmp_path: Path) -> None:
    tool = DemoTool("safe")
    store, monitor, agent = build_agent(tmp_path / "timeline.sqlite3", tool)
    result = agent.run("timeline goal")
    timeline = monitor.get_task_timeline(result["task_id"])
    event_types = [event["event_type"] for event in timeline]
    assert event_types[0] == "TASK_CREATED"
    assert event_types.index("PLAN_CREATED") < event_types.index("ACTION_STARTED")
    assert event_types.index("ACTION_STARTED") < event_types.index("VERIFICATION_PASSED")
    assert event_types[-1] in {"LEARNING_REJECTED", "LEARNING_COMPLETED"}
    assert monitor.get_recent_events(3)[-1]["sequence"] == timeline[-1]["sequence"]
    store.close()


def test_metrics_calculate_counts_and_durations(tmp_path: Path) -> None:
    tool = DemoTool("safe")
    store, monitor, agent = build_agent(tmp_path / "metrics.sqlite3", tool)
    result = agent.run("metrics goal")
    metrics = monitor.get_task_metrics(result["task_id"])
    assert metrics["task_count"] == 1
    assert metrics["completed_tasks"] == 1
    assert metrics["action_count"] == 1
    assert metrics["successful_actions"] == 1
    assert metrics["verification_success"] == 1
    assert metrics["tool_failures"] == 0
    assert metrics["task_duration_ms"] >= 0
    assert metrics["action_duration_ms"] >= 0
    assert metrics["tool_duration_ms"] >= 0
    assert monitor.get_system_metrics()["completed_tasks"] >= 1
    store.close()


def test_duration_metadata_is_monotonic_measurement(tmp_path: Path) -> None:
    store = Store(tmp_path / "duration.sqlite3")
    monitor = ObservabilityMonitor(store)
    monitor.emit("PLAN_CREATED", task_id="t", metadata={"duration_ms": 12.5})
    monitor.emit("TASK_COMPLETED", task_id="t", metadata={"duration_ms": 4.25})
    metrics = monitor.get_task_metrics("t")
    assert metrics["planning_duration_ms"] == 12.5
    assert metrics["task_duration_ms"] == 4.25
    store.close()


def test_security_policy_and_approval_events_are_observable(tmp_path: Path) -> None:
    path = tmp_path / "security.sqlite3"
    tool = DemoTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    store = Store(path)
    monitor = ObservabilityMonitor(store)
    security = SecurityController(store=store, observability=monitor)
    agent = Orchestrator(store, registry(tool), planner=PlannerStub([make_step("network")]), security_controller=security, observability=monitor)
    result = agent.run("high risk", task_id="security-task")
    assert result["final_status"] == "BLOCKED"
    types = {event["event_type"] for event in monitor.get_task_timeline("security-task")}
    assert {"SECURITY_REQUIRE_APPROVAL", "APPROVAL_REQUESTED", "TASK_BLOCKED"}.issubset(types)
    approval_id = result["approval_requests"][0]["approval_id"]
    security.approval_gate.approve(approval_id)
    assert "APPROVAL_APPROVED" in {event["event_type"] for event in monitor.get_security_events("security-task")}
    store.close()


def test_tool_lifecycle_events_cover_success_and_failure(tmp_path: Path) -> None:
    store = Store(tmp_path / "tools.sqlite3")
    monitor = ObservabilityMonitor(store)
    success_tool = DemoTool("success")
    failure_tool = DemoTool("failure", [ToolResult.failed("tool_failed", "controlled failure")])
    tools = registry(success_tool, failure_tool)
    executor = GenericExecutor(tools)
    context = ExecutionContext(task_id="tool-task", observability=monitor)
    executor.execute_step(make_step("success"), context)
    executor.execute_step(make_step("failure", "failure-step"), context)
    types = [event["event_type"] for event in monitor.get_tool_events("tool-task")]
    assert types.count("TOOL_STARTED") == 2
    assert "TOOL_COMPLETED" in types
    assert "TOOL_FAILED" in types
    store.close()


def test_verification_events_include_pass_and_failure_categories(tmp_path: Path) -> None:
    good = DemoTool("good")
    bad = DemoTool("bad", [ToolResult.success({"wrong": "shape"})])
    store = Store(tmp_path / "verification.sqlite3")
    monitor = ObservabilityMonitor(store)
    good_agent = Orchestrator(store, registry(good), planner=PlannerStub([make_step("good")]), observability=monitor)
    good_result = good_agent.run("good verification")
    bad_agent = Orchestrator(store, registry(bad), planner=PlannerStub([make_step("bad")]), observability=monitor)
    bad_result = bad_agent.run("bad verification")
    assert good_result["final_status"] == "COMPLETED"
    assert bad_result["final_status"] != "COMPLETED"
    assert "VERIFICATION_PASSED" in {event["event_type"] for event in monitor.get_task_timeline(good_result["task_id"])}
    assert "VERIFICATION_FAILED" in {event["event_type"] for event in monitor.get_task_timeline(bad_result["task_id"])}
    store.close()


def test_recovery_and_replanning_events_are_observable(tmp_path: Path) -> None:
    first = DemoTool("first", [ToolResult.failed("tool_failed", "temporary")])
    replacement = DemoTool("replacement", [ToolResult.success({"result": "recovered"})])
    store = Store(tmp_path / "recovery.sqlite3")
    monitor = ObservabilityMonitor(store)
    agent = Orchestrator(
        store,
        registry(first, replacement),
        planner=PlannerStub([make_step("first")]),
        observability=monitor,
        max_recovery_attempts=2,
    )
    result = agent.run("recovery telemetry")
    types = {event["event_type"] for event in monitor.get_task_timeline(result["task_id"])}
    assert result["final_status"] == "COMPLETED"
    assert {"RECOVERY_STARTED", "RECOVERY_COMPLETED", "REPLAN_STARTED", "REPLAN_COMPLETED", "PLAN_UPDATED"}.issubset(types)
    store.close()


def test_memory_reflection_learning_events_are_observable(tmp_path: Path) -> None:
    tool = DemoTool("safe")
    store, monitor, agent = build_agent(tmp_path / "memory.sqlite3", tool)
    result = agent.run("memory lifecycle")
    types = {event["event_type"] for event in monitor.get_task_timeline(result["task_id"])}
    assert "MEMORY_RETRIEVED" in types
    assert "MEMORY_STORED" in types
    assert "REFLECTION_STARTED" in types
    assert "REFLECTION_COMPLETED" in types
    assert "LEARNING_STARTED" in types
    assert "LEARNING_COMPLETED" in types
    assert "LEARNING_REJECTED" in types or result["learned_strategies"]
    store.close()


def test_skill_lifecycle_events_are_observable_without_changing_registry_behavior(tmp_path: Path) -> None:
    store = Store(tmp_path / "skills.sqlite3")
    monitor = ObservabilityMonitor(store)
    store.save_skill_candidate(
        {
            "skill_id": "skill-1",
            "name": "demo",
            "version": 1,
            "status": "CANDIDATE",
            "candidate": {"name": "demo"},
            "contract": {},
            "analysis": {},
            "mutation": {},
            "runtime": {},
            "cross_verification": {},
            "approval": {},
        }
    )
    store.save_skill_event("skill-1", "activated", {"name": "demo", "version": 1})
    store.save_skill_event("skill-1", "status_changed", {"status": "QUARANTINED"})
    store.save_skill_event("skill-1", "rollback", {"from_version": 1, "to_version": 0})
    types = {event["event_type"] for event in monitor.get_tool_events()}
    all_types = {event["event_type"] for event in store.all_observability_events()}
    assert {"SKILL_CANDIDATE", "SKILL_ACTIVATED", "SKILL_BLOCKED", "SKILL_ROLLED_BACK"}.issubset(all_types)
    assert not types  # skill events are not misclassified as tool events
    store.close()


def test_sensitive_metadata_is_redacted_and_raw_payload_is_summarized(tmp_path: Path) -> None:
    store = Store(tmp_path / "redaction.sqlite3")
    monitor = ObservabilityMonitor(store)
    secret = "Bearer VERY_SECRET_TOKEN_123456789"
    monitor.emit(
        "ACTION_STARTED",
        task_id="redact-task",
        metadata={
            "api_key": "AIza123456789012345678901234567890",
            "password": "plain-password",
            "authorization": secret,
            "input": {"private": "full sensitive payload"},
            "note": secret,
        },
    )
    raw = json.dumps(store.all_observability_events(), ensure_ascii=False)
    assert "VERY_SECRET_TOKEN_123456789" not in raw
    assert "plain-password" not in raw
    assert "full sensitive payload" not in raw
    assert "[REDACTED]" in raw or "redacted" in raw
    store.close()


def test_event_persistence_and_query_filters_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "persist.sqlite3"
    store = Store(path)
    monitor = ObservabilityMonitor(store)
    monitor.emit("TASK_FAILED", task_id="persist-task", severity="ERROR", metadata={"reason": "failure"})
    store.close()
    restarted = Store(path)
    restarted_monitor = ObservabilityMonitor(restarted)
    assert restarted_monitor.get_failures("persist-task")[0]["event_type"] == "TASK_FAILED"
    assert restarted_monitor.get_recent_events(10)[0]["task_id"] == "persist-task"
    restarted.close()


def test_resume_continues_same_timeline_and_task_identity(tmp_path: Path) -> None:
    tool = DemoTool("safe")

    def interrupt(checkpoint: str, state: dict) -> None:
        if checkpoint == "PLAN_CREATED":
            raise TaskInterrupted("controlled restart")

    store = Store(tmp_path / "resume-observability.sqlite3")
    monitor = ObservabilityMonitor(store)
    first = Orchestrator(store, registry(tool), planner=PlannerStub([make_step("safe")]), observability=monitor, checkpoint_hook=interrupt)
    with pytest.raises(TaskInterrupted):
        first.run("resume timeline")
    task_id = store.db.execute("SELECT task_id FROM task_states").fetchone()[0]
    restarted = Orchestrator(store, registry(tool), planner=PlannerStub([make_step("safe")]), observability=monitor)
    result = restarted.resume_task(task_id)
    assert result["final_status"] == "COMPLETED"
    timeline = monitor.get_task_timeline(task_id)
    assert "TASK_RESUMED" in [event["event_type"] for event in timeline]
    assert all(event["task_id"] == task_id for event in timeline)
    assert "TASK_COMPLETED" in [event["event_type"] for event in timeline]
    assert {"LEARNING_REJECTED", "LEARNING_COMPLETED"}.intersection(event["event_type"] for event in timeline)
    store.close()


def test_observability_storage_failure_degrades_without_authorization_or_execution_change(tmp_path: Path) -> None:
    class FailingStore(Store):
        def save_observability_event(self, payload: dict) -> int:
            raise sqlite3.OperationalError("telemetry disk unavailable")

    store = FailingStore(tmp_path / "degraded.sqlite3")
    monitor = ObservabilityMonitor(store)
    tool = DemoTool("safe")
    agent = Orchestrator(store, registry(tool), planner=PlannerStub([make_step("safe")]), observability=monitor)
    result = agent.run("degraded telemetry")
    assert result["final_status"] == "COMPLETED"
    assert tool.calls == 1
    assert monitor.degraded_writes > 0
    assert monitor.last_storage_error == "OperationalError"
    store.close()


def test_duplicate_event_id_is_idempotent_and_conflict_is_rejected(tmp_path: Path) -> None:
    store = Store(tmp_path / "dedupe.sqlite3")
    monitor = ObservabilityMonitor(store)
    first = monitor.emit("TASK_CREATED", task_id="dedupe", event_id="fixed-event", metadata={"x": 1})
    second = monitor.emit("TASK_CREATED", task_id="dedupe", event_id="fixed-event", metadata={"x": 1})
    assert first is not None and second is not None
    assert first.sequence == second.sequence
    assert len(store.all_observability_events()) == 1
    with pytest.raises(sqlite3.IntegrityError):
        monitor.emit("TASK_CREATED", task_id="dedupe", event_id="fixed-event", metadata={"x": 2})
    store.close()


def test_observability_never_authorizes_or_executes_blocked_action(tmp_path: Path) -> None:
    path = tmp_path / "authority.sqlite3"
    tool = DemoTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    store = Store(path)
    monitor = ObservabilityMonitor(store)
    security = SecurityController(store=store, observability=monitor)
    agent = Orchestrator(store, registry(tool), planner=PlannerStub([make_step("network")]), security_controller=security, observability=monitor)
    result = agent.run("blocked authority", task_id="authority-task")
    assert result["final_status"] == "BLOCKED"
    assert tool.calls == 0
    assert "SECURITY_REQUIRE_APPROVAL" in {event["event_type"] for event in monitor.get_security_events("authority-task")}
    store.close()


def test_phase11_query_surface_is_internal_and_complete(tmp_path: Path) -> None:
    store = Store(tmp_path / "queries.sqlite3")
    monitor = ObservabilityMonitor(store)
    monitor.emit("TOOL_FAILED", task_id="query-task", action_id="a", metadata={"error_category": "timeout", "duration_ms": 2})
    assert monitor.get_task_timeline("query-task")
    assert monitor.get_recent_events(1)
    assert monitor.get_task_metrics("query-task")["tool_failures"] == 1
    assert monitor.get_system_metrics()["tool_failures"] == 1
    assert monitor.get_failures("query-task")
    assert monitor.get_security_events("query-task") == []
    assert monitor.get_tool_events("query-task")[0]["event_type"] == "TOOL_FAILED"
    store.close()
