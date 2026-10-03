from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from private_agent.api import AgentAPIService, create_app
from private_agent.core.orchestrator import Orchestrator, TaskInterrupted
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.observability import ObservabilityMonitor
from private_agent.security import SecurityController
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


API_TOKEN = "phase12-owner-token-7d9e3b"


class APITestTool:
    permission_level = "public_read"
    planned_effect = "controlled API test action"

    def __init__(
        self,
        name: str,
        outcomes: list[ToolResult] | None = None,
        *,
        capabilities: list[str] | None = None,
        declared_capabilities: list[str] | None = None,
        risk: str = "low",
    ) -> None:
        self.name = name
        self.outcomes = list(outcomes or [ToolResult.success({"result": name})])
        self.calls = 0
        self.required_capabilities = list(capabilities or [])
        self.declared_capabilities = list(
            self.required_capabilities if declared_capabilities is None else declared_capabilities
        )
        self.risk_level = risk
        self.network_required = "NETWORK_ACCESS" in self.required_capabilities

    def description(self) -> str:
        return f"API test tool {self.name}"

    def input_schema(self) -> dict:
        return {"type": "object", "properties": {"value": {"type": "string"}}, "additionalProperties": False}

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        }

    def execute(self, inputs: dict, context) -> ToolResult:
        self.calls += 1
        return self.outcomes.pop(0) if self.outcomes else ToolResult.success({"result": self.name})


class APIPlanner:
    provider = None
    max_steps = 12

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name

    def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict]) -> TaskPlan:
        step = PlanStep(
            "api-step-1",
            "controlled API action",
            self.tool_name,
            {"value": "safe"},
            "API lifecycle test",
            [],
            success_criteria={"required_fields": ["result"]},
        )
        return TaskPlan(goal, [step], "API test plan")


def make_registry(*tools: APITestTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def make_service(
    tmp_path: Path,
    tool: APITestTool,
    *,
    token: str | None = API_TOKEN,
    hook=None,
) -> tuple[AgentAPIService, Store, ObservabilityMonitor]:
    store = Store(tmp_path / "api.sqlite3")
    monitor = ObservabilityMonitor(store)
    security = SecurityController(store=store, observability=monitor)
    core = Orchestrator(
        store,
        make_registry(tool),
        planner=APIPlanner(tool.name),
        security_controller=security,
        observability=monitor,
        checkpoint_hook=hook,
        max_recovery_attempts=2,
    )
    return AgentAPIService(core, token=token), store, monitor


def client_for(service: AgentAPIService) -> TestClient:
    return TestClient(create_app(service=service), raise_server_exceptions=False)


def auth_headers(*, key: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    if key is not None:
        headers["Idempotency-Key"] = key
    return headers


def create_low_task(client: TestClient, *, key: str = "create-1"):
    response = client.post("/api/v1/tasks", json={"goal": "complete a safe API task"}, headers=auth_headers(key=key))
    assert response.status_code == 201, response.text
    return response, response.json()["task_id"]


def create_high_task(client: TestClient, *, key: str = "high-1"):
    response = client.post("/api/v1/tasks", json={"goal": "request a controlled high risk action"}, headers=auth_headers(key=key))
    assert response.status_code == 201, response.text
    return response, response.json()["task_id"]


def test_authentication_rejects_missing_invalid_and_malformed_credentials(tmp_path: Path) -> None:
    service, store, monitor = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    for headers in ({}, {"Authorization": "Bearer wrong-owner-token"}, {"Authorization": "Basic phase12-owner-token"}, {"Authorization": "Bearer"}, {"Authorization": f"Bearer {API_TOKEN} extra"}):
        response = client.post("/api/v1/tasks", json={"goal": "must not run"}, headers=headers)
        assert response.status_code == 401
        body = response.json()
        assert set(("error_code", "message", "request_id")).issubset(body)
        assert API_TOKEN not in response.text
    assert monitor.get_system_metrics()["task_count"] == 0
    store.close()


def test_valid_authentication_creates_real_typed_task_response(tmp_path: Path) -> None:
    service, store, monitor = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    response, task_id = create_low_task(client)
    body = response.json()
    assert body["task_id"] == task_id
    assert body["status"] == "COMPLETED"
    assert body["current_phase"] == "TASK_COMPLETED"
    assert body["created_at"]
    assert body["api_version"] == "v1"
    assert body["schema_version"] == 1
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert store.get_task_state(task_id)["status"] == "COMPLETED"
    store.close()


def test_request_validation_and_error_model_are_deterministic(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    for payload in ({"goal": "   "}, {"goal": "valid", "unknown": "field"}, {}):
        response = client.post("/api/v1/tasks", json=payload, headers=auth_headers())
        assert response.status_code == 422
        body = response.json()
        assert body["error_code"] == "validation_error"
        assert body["message"] == "Request validation failed"
        assert "traceback" not in response.text.lower()
    store.close()


def test_task_get_is_safe_and_unknown_task_is_404(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    _, task_id = create_low_task(client)
    response = client.get(f"/api/v1/tasks/{task_id}", headers=auth_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == task_id
    assert body["status"] == "COMPLETED"
    assert body["plan_version"] == 1
    assert body["checkpoint"]["last_successful_checkpoint"] == "TASK_COMPLETED"
    assert body["verification_status"] == "VERIFIED"
    assert '"value":"safe"' not in response.text
    assert "raw sensitive payload" not in response.text
    unknown = client.get("/api/v1/tasks/not-a-real-task", headers=auth_headers())
    assert unknown.status_code == 404
    assert unknown.json()["error_code"] == "task_not_found"
    store.close()


def test_task_idempotency_replays_without_duplicate_execution_and_conflicts_safely(tmp_path: Path) -> None:
    tool = APITestTool("safe")
    service, store, _ = make_service(tmp_path, tool)
    client = client_for(service)
    first, task_id = create_low_task(client, key="stable-key")
    replay = client.post("/api/v1/tasks", json={"goal": "complete a safe API task"}, headers=auth_headers(key="stable-key"))
    assert replay.status_code == 201
    assert replay.json()["task_id"] == task_id
    assert replay.json()["idempotent_replay"] is True
    assert tool.calls == 1
    conflict = client.post("/api/v1/tasks", json={"goal": "a different goal"}, headers=auth_headers(key="stable-key"))
    assert conflict.status_code == 409
    assert conflict.json()["error_code"] == "idempotency_conflict"
    store.close()


def test_concurrent_same_idempotency_key_returns_one_task(tmp_path: Path) -> None:
    tool = APITestTool("safe")
    service, store, _ = make_service(tmp_path, tool)
    client = client_for(service)

    def submit() -> tuple[int, str]:
        response = client.post("/api/v1/tasks", json={"goal": "concurrent safe task"}, headers=auth_headers(key="concurrent-key"))
        return response.status_code, response.json().get("task_id", "")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: submit(), range(2)))
    assert all(status == 201 for status, _ in results)
    assert len({task_id for _, task_id in results}) == 1
    assert tool.calls == 1
    store.close()


def test_resume_calls_core_and_terminal_resume_does_not_execute_again(tmp_path: Path) -> None:
    tool = APITestTool("safe")
    service, store, _ = make_service(tmp_path, tool)
    client = client_for(service)
    _, task_id = create_low_task(client)
    resumed = client.post(f"/api/v1/tasks/{task_id}/resume", json={}, headers=auth_headers())
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "COMPLETED"
    assert tool.calls == 1
    assert client.post("/api/v1/tasks/not-real/resume", json={}, headers=auth_headers()).status_code == 404
    store.close()


def test_execution_unknown_resume_does_not_blindly_retry(tmp_path: Path) -> None:
    tool = APITestTool("safe")

    def interrupt(checkpoint: str, state: dict) -> None:
        if checkpoint == "BEFORE_EXECUTION":
            raise TaskInterrupted("controlled API interruption")

    service, store, _ = make_service(tmp_path, tool, hook=interrupt)
    client = client_for(service)
    failed_create = client.post("/api/v1/tasks", json={"goal": "unknown execution task"}, headers=auth_headers())
    assert failed_create.status_code == 500
    task_id = store.db.execute("SELECT task_id FROM task_states").fetchone()[0]
    resumed = client.post(f"/api/v1/tasks/{task_id}/resume", json={}, headers=auth_headers())
    assert resumed.status_code == 200
    body = resumed.json()
    assert body["status"] == "EXECUTION_UNKNOWN"
    assert body["verification_status"] in {"NOT_STARTED", "BLOCKED"}
    assert tool.calls == 0
    event_types = [event["event_type"] for event in service.observability.get_task_timeline(task_id)]
    assert "EXECUTION_UNKNOWN" in event_types
    store.close()


def test_approval_list_inspect_approve_and_resume_preserve_binding(tmp_path: Path) -> None:
    tool = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, monitor = make_service(tmp_path, tool)
    client = client_for(service)
    created, task_id = create_high_task(client)
    assert created.json()["status"] == "WAITING_APPROVAL"
    pending = client.get("/api/v1/approvals", headers=auth_headers())
    assert pending.status_code == 200
    assert len(pending.json()) == 1
    approval_id = pending.json()[0]["approval_id"]
    inspected = client.get(f"/api/v1/approvals/{approval_id}", headers=auth_headers())
    assert inspected.status_code == 200
    approval = inspected.json()
    assert approval["task_id"] == task_id
    assert approval["risk_level"] == "HIGH"
    assert approval["requested_capabilities"] == ["NETWORK_ACCESS"]
    cannot_modify = client.post(
        f"/api/v1/approvals/{approval_id}/approve",
        json={"risk_level": "LOW"},
        headers=auth_headers(),
    )
    assert cannot_modify.status_code == 422
    approved = client.post(f"/api/v1/approvals/{approval_id}/approve", json={}, headers=auth_headers())
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"
    resumed = client.post(f"/api/v1/tasks/{task_id}/resume", json={}, headers=auth_headers())
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "COMPLETED"
    assert tool.calls == 1
    assert "APPROVAL_APPROVED" in {event["event_type"] for event in monitor.get_security_events(task_id)}
    duplicate = client.post(f"/api/v1/approvals/{approval_id}/approve", json={}, headers=auth_headers())
    assert duplicate.status_code == 409
    store.close()


def test_approval_deny_and_expiry_are_safe_and_non_executable(tmp_path: Path) -> None:
    tool = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, tool)
    client = client_for(service)
    _, task_id = create_high_task(client, key="deny-task")
    approval_id = client.get("/api/v1/approvals", headers=auth_headers()).json()[0]["approval_id"]
    denied = client.post(f"/api/v1/approvals/{approval_id}/deny", json={}, headers=auth_headers())
    assert denied.status_code == 200
    assert denied.json()["status"] == "DENIED"
    assert tool.calls == 0
    _, expired_task_id = create_high_task(client, key="expired-task")
    expired_id = client.get("/api/v1/approvals", headers=auth_headers()).json()[0]["approval_id"]
    request = service.security.approval_gate.get(expired_id)
    request.expires_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    service.security.approval_gate._persist(request)
    expired = client.post(f"/api/v1/approvals/{expired_id}/approve", json={}, headers=auth_headers())
    assert expired.status_code == 409
    assert expired.json()["error_code"] == "approval_not_pending"
    assert client.get("/api/v1/approvals", headers=auth_headers()).json() == []
    assert store.get_task_state(task_id)["status"] == "WAITING_APPROVAL"
    assert store.get_task_state(expired_task_id)["status"] == "WAITING_APPROVAL"
    store.close()


def test_api_cannot_bypass_capability_or_quarantined_skill_boundaries(tmp_path: Path) -> None:
    mismatch = APITestTool(
        "undeclared-network",
        capabilities=["NETWORK_ACCESS"],
        declared_capabilities=[],
        risk="low",
    )
    service, store, monitor = make_service(tmp_path, mismatch)
    client = client_for(service)
    created, task_id = create_low_task(client, key="mismatch")
    assert created.json()["status"] == "BLOCKED"
    assert mismatch.calls == 0
    security_events = client.get(f"/api/v1/security-events?task_id={task_id}", headers=auth_headers())
    assert security_events.status_code == 200
    assert any(item["event_type"] == "SECURITY_DENY" for item in security_events.json())
    assert client.post("/api/v1/skills/quarantined/activate", json={}, headers=auth_headers()).status_code == 404
    store.close()


def test_observability_timeline_metrics_failures_security_and_tools_are_read_only(tmp_path: Path) -> None:
    tool = APITestTool("safe", [ToolResult.failed("tool_failed", "controlled failure")])
    service, store, monitor = make_service(tmp_path, tool)
    client = client_for(service)
    _, task_id = create_low_task(client, key="failure-task")
    timeline = client.get(f"/api/v1/tasks/{task_id}/timeline", headers=auth_headers())
    assert timeline.status_code == 200
    events = timeline.json()
    assert [item["sequence"] for item in events] == sorted(item["sequence"] for item in events)
    assert any(item["event_type"] == "TOOL_FAILED" for item in events)
    task_metrics = client.get(f"/api/v1/tasks/{task_id}/metrics", headers=auth_headers())
    assert task_metrics.status_code == 200
    assert task_metrics.json()["tool_failures"] >= 1
    for path in ("/api/v1/failures", "/api/v1/security-events", "/api/v1/tool-events", "/api/v1/metrics"):
        response = client.get(path, headers=auth_headers())
        assert response.status_code == 200
    assert client.get(f"/api/v1/tool-events?task_id={task_id}", headers=auth_headers()).json()
    assert monitor.get_system_metrics()["tool_failures"] >= 1
    store.close()


def test_request_correlation_uses_header_and_never_contains_credentials(tmp_path: Path) -> None:
    service, store, monitor = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    headers = auth_headers(key="correlation")
    headers["X-Request-ID"] = "request-phase12-001"
    response = client.post("/api/v1/tasks", json={"goal": "correlated task"}, headers=headers)
    assert response.status_code == 201
    assert response.headers["X-Request-ID"] == "request-phase12-001"
    task_id = response.json()["task_id"]
    timeline = monitor.get_task_timeline(task_id)
    completed = [event for event in timeline if event["event_type"] == "API_REQUEST_COMPLETED"]
    assert completed
    assert completed[-1]["correlation_id"] == "request-phase12-001"
    raw = json.dumps(store.all_observability_events(), ensure_ascii=False)
    assert API_TOKEN not in raw
    store.close()


def test_health_and_readiness_distinguish_process_liveness_from_dependencies(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert client.get("/health").status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    unconfigured = client_for(AgentAPIService(service.orchestrator, token=""))
    not_ready = unconfigured.get("/ready")
    assert not_ready.status_code == 503
    assert not_ready.json()["error_code"] == "not_ready"
    store.close()


def test_api_created_task_timeline_and_approval_survive_service_restart(tmp_path: Path) -> None:
    path = tmp_path / "api.sqlite3"
    tool = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, monitor = make_service(tmp_path, tool)
    client = client_for(service)
    created, task_id = create_high_task(client, key="restart-high")
    approval_id = client.get("/api/v1/approvals", headers=auth_headers()).json()[0]["approval_id"]
    store.close()
    reopened = Store(path)
    restarted_monitor = ObservabilityMonitor(reopened)
    restarted_security = SecurityController(store=reopened, observability=restarted_monitor)
    restarted_core = Orchestrator(
        reopened,
        make_registry(tool),
        planner=APIPlanner(tool.name),
        security_controller=restarted_security,
        observability=restarted_monitor,
    )
    restarted_client = client_for(AgentAPIService(restarted_core, token=API_TOKEN))
    state = restarted_client.get(f"/api/v1/tasks/{task_id}", headers=auth_headers())
    assert state.status_code == 200
    assert state.json()["status"] == "WAITING_APPROVAL"
    assert restarted_client.get(f"/api/v1/approvals/{approval_id}", headers=auth_headers()).status_code == 200
    timeline = restarted_client.get(f"/api/v1/tasks/{task_id}/timeline", headers=auth_headers())
    assert timeline.status_code == 200
    assert len(timeline.json()) > 0
    reopened.close()


def test_duplicate_approval_requests_are_serialized_and_state_remains_valid(tmp_path: Path) -> None:
    tool = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, tool)
    client = client_for(service)
    _, task_id = create_high_task(client, key="approval-race")
    approval_id = client.get("/api/v1/approvals", headers=auth_headers()).json()[0]["approval_id"]

    def approve() -> int:
        return client.post(f"/api/v1/approvals/{approval_id}/approve", json={}, headers=auth_headers()).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda _: approve(), range(2)))
    assert sorted(statuses) == [200, 409]
    state = client.get(f"/api/v1/tasks/{task_id}", headers=auth_headers())
    assert state.status_code == 200
    assert state.json()["approvals"][0]["status"] == "APPROVED"
    store.close()


def test_approval_resume_race_has_no_500_and_never_executes_twice(tmp_path: Path) -> None:
    tool = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, tool)
    client = client_for(service)
    _, task_id = create_high_task(client, key="approval-resume-race")
    approval_id = client.get("/api/v1/approvals", headers=auth_headers()).json()[0]["approval_id"]

    def approve() -> int:
        return client.post(f"/api/v1/approvals/{approval_id}/approve", json={}, headers=auth_headers()).status_code

    def resume() -> int:
        return client.post(f"/api/v1/tasks/{task_id}/resume", json={}, headers=auth_headers()).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda fn: fn(), (approve, resume)))
    assert all(status in {200, 409} for status in statuses)
    final = client.post(f"/api/v1/tasks/{task_id}/resume", json={}, headers=auth_headers())
    assert final.status_code == 200
    assert final.json()["status"] in {"COMPLETED", "WAITING_APPROVAL", "BLOCKED"}
    assert tool.calls <= 1
    store.close()


def test_public_api_does_not_expose_cancel_or_skill_activation_as_fake_operations(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    response = client.post("/api/v1/tasks/not-real/cancel", json={}, headers=auth_headers())
    assert response.status_code in {404, 405}
    skill = client.post("/api/v1/skills/demo/activate", json={}, headers=auth_headers())
    assert skill.status_code == 404
    store.close()
