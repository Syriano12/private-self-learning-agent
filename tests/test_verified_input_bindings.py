from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from private_agent.api import AgentAPIService, create_app
from private_agent.core.execution import ExecutionContext, GenericExecutor, StepResult
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import PlanStep, PlanValidationError, Planner, TaskPlan
from private_agent.core.recovery import RecoveryManager, input_fingerprint
from private_agent.observability import ObservabilityMonitor
from private_agent.security import SecurityController
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry
from private_agent.worker import BackgroundWorker


API_TOKEN = "verified-binding-test-derived_value"


class BindingTool:
    permission_level = "public_read"
    risk_level = "low"
    planned_effect = "bounded local binding test"

    def __init__(self, name: str, calls: list[dict[str, Any]] | None = None) -> None:
        self.name = name
        self.calls = calls if calls is not None else []

    def description(self) -> str:
        return f"Benign input-binding fixture: {self.name}"

    def input_schema(self) -> dict[str, Any]:
        if self.name == "source":
            return {
                "type": "object",
                "properties": {"seed": {"type": "string"}},
                "required": ["seed"],
                "additionalProperties": False,
            }
        return {
            "type": "object",
            "properties": {"derived_value": {"type": "string"}},
            "required": ["derived_value"],
            "additionalProperties": False,
        }

    def output_schema(self) -> dict[str, Any]:
        if self.name == "source":
            return {
                "type": "object",
                "properties": {"derived_value": {"type": "string"}},
                "required": ["derived_value"],
                "additionalProperties": False,
            }
        return {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        }

    def execute(self, inputs: dict[str, Any], context: ExecutionContext) -> ToolResult:
        self.calls.append(dict(inputs))
        if self.name == "source":
            return ToolResult.success({"derived_value": f"derived:{inputs['seed']}"})
        return ToolResult.success({"result": f"consumed:{inputs['derived_value']}"})


class BindingPlanner:
    provider = None
    max_steps = 12

    def __init__(self, *, source_verifier_should_fail: bool = False) -> None:
        self.source_verifier_should_fail = source_verifier_should_fail

    def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict[str, Any]]) -> TaskPlan:
        required_fields = ["derived_value", "absent"] if self.source_verifier_should_fail else ["derived_value"]
        return Planner().validate(
            {
                "goal": goal,
                "rationale": "deterministic local input-binding integration fixture",
                "steps": [
                    {
                        "id": "source-step",
                        "objective": "produce a declared derived_value",
                        "tool": "source",
                        "input": {"seed": "fixture"},
                        "reason": "registered benign test fixture",
                        "depends_on": [],
                        "success_criteria": {"required_fields": required_fields},
                    },
                    {
                        "id": "consumer-step",
                        "objective": "consume the verified derived_value",
                        "tool": "consumer",
                        "input": {},
                        "reason": "registered benign test fixture",
                        "depends_on": ["source-step"],
                        "input_bindings": {
                            "derived_value": {
                                "source_step_id": "source-step",
                                "output_key": "derived_value",
                            }
                        },
                        "success_criteria": {"required_fields": ["result"]},
                    },
                ],
            },
            goal=goal,
            tools=tools,
        )


def make_registry(*tools: BindingTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def make_api_task(
    tmp_path: Path,
    *,
    source_verifier_should_fail: bool = False,
    monkeypatch: pytest.MonkeyPatch | None = None,
):
    store = Store(tmp_path / "input-bindings.sqlite3")
    monitor = ObservabilityMonitor(store)
    security = SecurityController(store=store, observability=monitor)
    tools = make_registry(BindingTool("source"), BindingTool("consumer"))
    orchestrator = Orchestrator(
        store,
        tools,
        planner=BindingPlanner(source_verifier_should_fail=source_verifier_should_fail),
        security_controller=security,
        observability=monitor,
        recovery_manager=RecoveryManager(max_recovery_attempts=0, max_attempts_per_step=0),
    )
    service = AgentAPIService(orchestrator, token=API_TOKEN)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    observed_inputs: dict[str, dict[str, Any]] = {}
    if monkeypatch is not None:
        original = SecurityController.action_for_tool

        def record_action(controller: SecurityController, task_id: str, step_id: str, tool: Any, inputs: dict[str, Any], *, goal: str = ""):
            observed_inputs[step_id] = dict(inputs)
            return original(controller, task_id, step_id, tool, inputs, goal=goal)

        monkeypatch.setattr(SecurityController, "action_for_tool", record_action)
    return client, orchestrator, store, tools, observed_inputs


def test_api_task_binds_only_verified_source_output_before_authorization_and_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, orchestrator, store, tools, authorized_inputs = make_api_task(tmp_path, monkeypatch=monkeypatch)
    goal = "produce and consume a verified local derived_value"

    created = client.post(
        "/api/v1/tasks",
        json={"goal": goal},
        headers={"Authorization": f"Bearer {API_TOKEN}", "Idempotency-Key": "verified-input-binding"},
    )
    assert created.status_code == 201, created.text
    task_id = created.json()["task_id"]
    assert created.json()["status"] == "QUEUED"

    worker = BackgroundWorker(orchestrator, worker_id="input-binding-worker")
    result = worker.run_once()

    assert result["status"] == "COMPLETED"
    assert result["task_id"] == task_id
    assert tools.get("source").calls == [{"seed": "fixture"}]
    assert tools.get("consumer").calls == [{"derived_value": "derived:fixture"}]
    assert authorized_inputs["consumer-step"] == {"derived_value": "derived:fixture"}
    state = store.get_task_state(task_id)
    restored_plan = orchestrator._plan_from_payload(state["plan"])
    assert restored_plan.steps[1].input_bindings == {
        "derived_value": {"source_step_id": "source-step", "output_key": "derived_value"}
    }
    records = list(state["action_records"].values())
    consumer_record = next(record for record in records if record.get("step_id") == "consumer-step")
    assert consumer_record["status"] == "VERIFIED"
    assert consumer_record["verification"]["verified"] is True
    store.close()


def test_bound_high_risk_action_requires_exact_resolved_input_approval_before_resume(tmp_path: Path) -> None:
    client, orchestrator, store, tools, _ = make_api_task(tmp_path)
    consumer = tools.get("consumer")
    consumer.required_capabilities = ["NETWORK_ACCESS"]
    consumer.declared_capabilities = ["NETWORK_ACCESS"]
    consumer.risk_level = "high"
    goal = "approve the exact value produced by a verified dependency"

    created = client.post(
        "/api/v1/tasks",
        json={"goal": goal},
        headers={"Authorization": f"Bearer {API_TOKEN}", "Idempotency-Key": "bound-action-approval"},
    )
    assert created.status_code == 201, created.text
    task_id = created.json()["task_id"]
    worker = BackgroundWorker(orchestrator, worker_id="bound-approval-worker")

    waiting = worker.run_once()

    assert waiting["status"] == "WAITING_APPROVAL"
    assert waiting["task_id"] == task_id
    assert tools.get("source").calls == [{"seed": "fixture"}]
    assert consumer.calls == []
    state = store.get_task_state(task_id)
    assert state["status"] == "WAITING_APPROVAL"
    approval_id = state["approval_ids"]["consumer-step"]
    approval_response = client.get(
        f"/api/v1/approvals/{approval_id}",
        headers={"Authorization": f"Bearer {API_TOKEN}"},
    )
    assert approval_response.status_code == 200, approval_response.text
    approval = approval_response.json()
    resolved_fingerprint = input_fingerprint({"derived_value": "derived:fixture"})
    assert approval["input_fingerprint"] == resolved_fingerprint
    assert approval["input_fingerprint"] != input_fingerprint({})
    assert approval["status"] == "PENDING"

    approved = client.post(
        f"/api/v1/approvals/{approval_id}/approve",
        json={},
        headers={"Authorization": f"Bearer {API_TOKEN}"},
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "APPROVED"
    store.requeue_waiting_approval_task(task_id)

    resumed = worker.run_once()

    assert resumed["status"] == "COMPLETED", resumed
    assert resumed["task_id"] == task_id
    assert consumer.calls == [{"derived_value": "derived:fixture"}]
    completed_state = store.get_task_state(task_id)
    bound_decision = completed_state["security_decisions"]["consumer-step"]
    assert bound_decision["input_fingerprint"] == resolved_fingerprint
    assert bound_decision["decision"] == "ALLOW"
    assert completed_state["status"] == "COMPLETED"
    store.close()


def test_api_worker_never_passes_source_output_when_source_verification_fails(
    tmp_path: Path,
) -> None:
    client, orchestrator, store, tools, _ = make_api_task(tmp_path, source_verifier_should_fail=True)
    goal = "produce a derived_value that does not satisfy verification"
    created = client.post(
        "/api/v1/tasks",
        json={"goal": goal},
        headers={"Authorization": f"Bearer {API_TOKEN}", "Idempotency-Key": "unverified-source-binding"},
    )
    assert created.status_code == 201, created.text
    task_id = created.json()["task_id"]

    result = BackgroundWorker(orchestrator, worker_id="unverified-binding-worker").run_once()

    assert result["status"] == "FAILED"
    assert result["task_id"] == task_id
    assert tools.get("source").calls == [{"seed": "fixture"}]
    assert tools.get("consumer").calls == []
    state = store.get_task_state(task_id)
    source_record = next(
        record for record in state["action_records"].values() if record.get("step_id") == "source-step"
    )
    assert source_record["status"] != "VERIFIED"
    assert any(
        isinstance(value, dict) and value.get("status") == "FAILED"
        for value in state["verification_results"].values()
    )
    store.close()


def test_unverified_dependency_result_cannot_satisfy_an_input_binding() -> None:
    source = BindingTool("source")
    consumer = BindingTool("consumer")
    tools = make_registry(source, consumer)
    security = SecurityController()
    step = PlanStep(
        id="consumer-step",
        objective="must not consume an unverified derived_value",
        tool="consumer",
        input={},
        reason="direct fail-closed regression",
        depends_on=["source-step"],
        input_bindings={"derived_value": {"source_step_id": "source-step", "output_key": "derived_value"}},
    )
    context = ExecutionContext(
        previous_results={
            "source-step": StepResult.success("source-step", "source", {"derived_value": "UNVERIFIED"})
        },
        security_controller=security,
    )

    result = GenericExecutor(tools, security).execute_step(step, context)

    assert result.status == "blocked"
    assert result.error is not None and result.error.code == "input_binding_source_unverified"
    assert consumer.calls == []


def test_planner_rejects_input_bindings_that_do_not_match_tool_contracts() -> None:
    source = BindingTool("source")
    consumer = BindingTool("consumer")
    tools = make_registry(source, consumer)
    invalid_plan = {
        "goal": "reject an undeclared source field",
        "rationale": "negative schema regression",
        "steps": [
            {
                "id": "source-step",
                "objective": "produce derived_value",
                "tool": "source",
                "input": {"seed": "fixture"},
                "reason": "fixture",
                "depends_on": [],
                "success_criteria": {"required_fields": ["derived_value"]},
            },
            {
                "id": "consumer-step",
                "objective": "consume derived_value",
                "tool": "consumer",
                "input": {},
                "reason": "fixture",
                "depends_on": ["source-step"],
                "input_bindings": {
                    "derived_value": {"source_step_id": "source-step", "output_key": "not_declared"}
                },
            },
        ],
    }

    with pytest.raises(PlanValidationError, match="input_binding_source_output_unavailable"):
        Planner().validate(invalid_plan, goal=invalid_plan["goal"], tools=tools)
