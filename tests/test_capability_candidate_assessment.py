from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from private_agent.api import AgentAPIService, create_app
from private_agent.auth import AuthManager
from private_agent.core.execution import ExecutionContext, GenericExecutor
from private_agent.core.llm import LLMResponse
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import PlanStep, Planner
from private_agent.security import SecurityController
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


API_TOKEN = "capability-candidate-owner-token-2026"


class StaticPlanProvider:
    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan

    def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict[str, Any]) -> LLMResponse:
        return LLMResponse(self.plan, json.dumps(self.plan), "capability-candidate-fixture")


class CountingTool:
    permission_level = "public_read"
    planned_effect = "bounded synthetic test effect"
    risk_level = "low"
    required_capabilities: list[str] = []
    declared_capabilities: list[str] = []
    network_required = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def description(self) -> str:
        return f"Synthetic tool {self.name}"

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}

    def output_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"], "additionalProperties": False}

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        self.calls += 1
        return ToolResult.success({"result": f"ok:{inputs['value']}"})


class NetworkTool(CountingTool):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.required_capabilities = ["NETWORK_ACCESS"]
        self.declared_capabilities = ["NETWORK_ACCESS"]
        self.risk_level = "high"
        self.network_required = True


class _ObservedRegistryLock:
    """Expose actual lock contention for the named registration test thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.registration_contended = threading.Event()

    def __enter__(self) -> "_ObservedRegistryLock":
        if threading.current_thread().name == "candidate-registration":
            if not self._lock.acquire(blocking=False):
                self.registration_contended.set()
                self._lock.acquire()
            else:
                self._lock.release()
                self._lock.acquire()
        else:
            self._lock.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self._lock.release()
        return False


def _plan(goal: str, tool_name: str) -> dict[str, Any]:
    return {
        "goal": goal,
        "rationale": "Use the named capability to satisfy the synthetic goal.",
        "steps": [
            {
                "id": "candidate-gap-step",
                "objective": "Perform a bounded synthetic action",
                "tool": tool_name,
                "input": {"value": "safe"},
                "reason": "The task planner requested this capability.",
                "depends_on": [],
                "success_criteria": {"required_fields": ["result"]},
            }
        ],
    }


def _make_blocked_task(tmp_path: Path, *, target_tool: str = "image_generator") -> tuple[Orchestrator, Store, ToolRegistry, str]:
    store = Store(tmp_path / "candidate-assessment.sqlite3")
    registry = ToolRegistry()
    present = CountingTool("registered_read")
    registry.register(present)
    goal = "Create a synthetic result using the missing capability."
    orchestrator = Orchestrator(
        store,
        registry,
        planner=Planner(StaticPlanProvider(_plan(goal, target_tool))),
        security_controller=SecurityController(store=store),
    )
    result = orchestrator.run(goal)
    assert present.calls == 0
    assert result["final_status"] == "BLOCKED"
    assert result["capability_gap"]["requested_tools"] == [target_tool]
    return orchestrator, store, registry, result["task_id"]


def _manifest(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "target_tool": "image_generator",
        "version": 1,
        "description": "Synthetic candidate for static review",
        "source": 'def run(payload):\n    return {"result": "candidate"}\n',
        "entry_point": "run",
        "expected_inputs": {"type": "object", "properties": {"value": {"type": "string"}}},
        "expected_outputs": {"type": "object", "properties": {"result": {"type": "string"}}},
    }
    payload.update(overrides)
    return payload


def _post_assessment(service: AgentAPIService, task_id: str, payload: dict[str, Any]):
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        return client.post(
            f"/api/v1/tasks/{task_id}/capability-candidates/assess",
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            json=payload,
        )


def test_untrusted_candidate_is_rejected_and_never_registered_or_executed(tmp_path: Path) -> None:
    orchestrator, store, tools, task_id = _make_blocked_task(tmp_path)
    service = AgentAPIService(orchestrator, token=API_TOKEN)
    malicious_source = "import os\n\ndef run(payload):\n    return os.system('true')\n"

    response = _post_assessment(service, task_id, _manifest(source=malicious_source))

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "REJECTED"
    assert body["review_status"] == "REJECTED_AUTOMATED"
    assert body["analysis"]["passed"] is False
    assert body["permissions_granted"] is False
    assert body["execution_attempted"] is False
    assert body["installation_attempted"] is False
    assert body["registration_attempted"] is False
    assert body["source_sha256"]
    assert "source" not in body
    assert service.skill_registry.active("image_generator") is None
    assert store.db.execute("SELECT COUNT(*) FROM skills").fetchone()[0] == 0
    assert store.all_approval_requests() == []

    executor = GenericExecutor(tools, orchestrator.security)
    execution = executor.execute_step(
        PlanStep("try-candidate", "try candidate", "image_generator", {"value": "x"}, "test", []),
        ExecutionContext(task_id=task_id, security_controller=orchestrator.security),
    )
    assert execution.status == "failed"
    assert execution.error is not None and execution.error.code == "unknown_tool"
    assert tools.get("registered_read").calls == 0
    try:
        tools.get("image_generator")
        raise AssertionError("candidate unexpectedly entered ToolRegistry")
    except KeyError:
        pass
    store.close()


def test_candidate_proposal_cannot_grant_network_permissions(tmp_path: Path) -> None:
    orchestrator, store, _tools, task_id = _make_blocked_task(tmp_path)
    service = AgentAPIService(orchestrator, token=API_TOKEN)
    policy_version = orchestrator.security.policy_engine._current_policy_version()
    policy_before = store.get_security_policy(policy_version)

    response = _post_assessment(
        service,
        task_id,
        _manifest(
            version=2,
            source='import requests\n\ndef run(payload):\n    return {"result": "network"}\n',
            declared_capabilities=["network"],
            allowed_imports=["requests"],
        ),
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "REJECTED"
    assert body["permissions_granted"] is False
    assert body["review_status"] == "REJECTED_AUTOMATED"
    assert body["contract"]["validation"]["passed"] is False
    assert "network_capability_not_allowed_by_contract" in body["contract"]["validation"]["errors"]
    assert body["contract"]["assessment_contract"]["network_allowed"] is False
    assert body["execution_attempted"] is False
    assert body["registration_attempted"] is False
    assert store.all_approval_requests() == []
    assert store.get_security_policy(policy_version) == policy_before
    assert store.db.execute("SELECT COUNT(*) FROM skills WHERE enabled=1").fetchone()[0] == 0
    store.close()


def test_safe_candidate_assessment_is_durable_and_stays_inactive(tmp_path: Path) -> None:
    orchestrator, store, _tools, task_id = _make_blocked_task(tmp_path)
    service = AgentAPIService(orchestrator, token=API_TOKEN)

    response = _post_assessment(service, task_id, _manifest())

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "CANDIDATE"
    assert body["review_status"] == "PENDING_REVIEW"
    assert body["contract"]["validation"]["passed"] is True
    assert body["analysis"]["passed"] is True
    assert body["capability_gap_task_id"] == task_id
    assert body["target_tool"] == "image_generator"
    assert body["permissions_granted"] is False
    assert body["execution_attempted"] is False
    assert body["installation_attempted"] is False
    assert body["registration_attempted"] is False
    candidate_id = body["candidate_id"]
    stored = service.skill_registry.get_candidate(candidate_id)
    assert stored is not None
    assessment = stored["candidate"]["metadata"]["capability_assessment"]
    assert assessment["capability_gap_task_id"] == task_id
    assert assessment["source_sha256"] == body["source_sha256"]
    assert service.skill_registry.active("image_generator") is None
    assert store.db.execute("SELECT COUNT(*) FROM skills").fetchone()[0] == 0
    store.close()

    reopened = Store(tmp_path / "candidate-assessment.sqlite3")
    new_orchestrator = Orchestrator(
        reopened,
        ToolRegistry(),
        planner=Planner(StaticPlanProvider(_plan("unused", "image_generator"))),
        security_controller=SecurityController(store=reopened),
    )
    reopened_service = AgentAPIService(new_orchestrator, token=API_TOKEN)
    with TestClient(create_app(service=reopened_service), raise_server_exceptions=False) as client:
        loaded = client.get(
            f"/api/v1/capability-candidates/{candidate_id}",
            headers={"Authorization": f"Bearer {API_TOKEN}"},
        )
        duplicate = client.post(
            f"/api/v1/tasks/{task_id}/capability-candidates/assess",
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            json=_manifest(),
        )
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["candidate_id"] == candidate_id
    assert loaded.json()["capability_gap_task_id"] == task_id
    assert loaded.json()["status"] == "CANDIDATE"
    assert loaded.json()["review_status"] == "PENDING_REVIEW"
    assert duplicate.status_code == 409, duplicate.text
    assert reopened_service.skill_registry.active("image_generator") is None
    reopened.close()


def test_existing_registered_tool_still_requires_exact_security_approval(tmp_path: Path) -> None:
    store = Store(tmp_path / "registered-tool-policy.sqlite3")
    controller = SecurityController(store=store)
    network = NetworkTool("existing_network_tool")
    registry = ToolRegistry()
    registry.register(network)
    executor = GenericExecutor(registry, controller)
    step = PlanStep("network-action", "network", network.name, {"value": "safe"}, "test", [])

    blocked = executor.execute_step(
        step,
        ExecutionContext(task_id="existing-capability", security_controller=controller),
    )
    assert blocked.status == "blocked"
    assert blocked.error is not None and blocked.error.code == "permission_not_approved"
    assert network.calls == 0
    approval = store.all_approval_requests()[-1]
    controller.approval_gate.approve(approval["approval_id"])
    allowed = executor.execute_step(
        step,
        ExecutionContext(
            task_id="existing-capability",
            security_controller=controller,
            approval_ids={step.id: approval["approval_id"]},
        ),
    )
    assert allowed.status == "success"
    assert network.calls == 1
    store.close()


def test_assessment_rejects_candidate_when_requested_tool_is_now_registered(tmp_path: Path) -> None:
    orchestrator, store, tools, task_id = _make_blocked_task(tmp_path)
    service = AgentAPIService(orchestrator, token=API_TOKEN)
    now_available = CountingTool("image_generator")
    tools.register(now_available)

    response = _post_assessment(service, task_id, _manifest())

    assert response.status_code == 409, response.text
    assert response.json()["error_code"] == "capability_gap_resolved"
    assert store.db.execute("SELECT COUNT(*) FROM skill_candidates").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM skills").fetchone()[0] == 0
    assert now_available.calls == 0
    assert store.get_task_state(task_id)["status"] == "BLOCKED"
    store.close()


def test_registration_waits_until_candidate_assessment_is_durably_decided(tmp_path: Path) -> None:
    orchestrator, store, tools, task_id = _make_blocked_task(tmp_path)
    service = AgentAPIService(orchestrator, token=API_TOKEN)
    assessor = service.capability_candidate_assessor
    registry_lock = _ObservedRegistryLock()
    tools._lock = registry_lock

    persist_entered = threading.Event()
    release_persist = threading.Event()
    decision_saved = threading.Event()
    registration_completed = threading.Event()
    thread_errors: list[BaseException] = []
    registration_order: list[bool] = []

    original_persist = assessor.registry.persist_candidate

    def pause_persist(*args: Any, **kwargs: Any) -> None:
        persist_entered.set()
        if not release_persist.wait(timeout=5):
            raise AssertionError("test did not release candidate persistence")
        original_persist(*args, **kwargs)

    assessor.registry.persist_candidate = pause_persist
    original_assessment = assessor._assess_for_gap_under_registry_lock

    def mark_assessment_decided(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_assessment(*args, **kwargs)
        decision_saved.set()
        return result

    assessor._assess_for_gap_under_registry_lock = mark_assessment_decided
    response_holder: list[Any] = []

    def assess() -> None:
        try:
            response_holder.append(_post_assessment(service, task_id, _manifest()))
        except BaseException as exc:
            thread_errors.append(exc)

    def register() -> None:
        try:
            tools.register(CountingTool("image_generator"))
            registration_order.append(decision_saved.is_set())
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            registration_completed.set()

    assessment_thread = threading.Thread(target=assess, name="candidate-assessment")
    registration_thread = threading.Thread(target=register, name="candidate-registration")
    assessment_thread.start()
    try:
        assert persist_entered.wait(timeout=5), "assessment did not reach durable persistence"
        registration_thread.start()
        assert registry_lock.registration_contended.wait(timeout=5), "registration did not contend on the shared lock"
        assert not decision_saved.is_set()
        assert not registration_completed.is_set()
    finally:
        release_persist.set()
        assessment_thread.join(timeout=5)
        if registration_thread.ident is not None:
            registration_thread.join(timeout=5)

    assert not assessment_thread.is_alive()
    assert not registration_thread.is_alive()
    assert thread_errors == []
    assert len(response_holder) == 1
    assert response_holder[0].status_code == 201, response_holder[0].text
    assert response_holder[0].json()["status"] == "CANDIDATE"
    assert decision_saved.is_set()
    assert registration_order == [True]
    assert tools.get("image_generator").calls == 0
    assert store.db.execute("SELECT COUNT(*) FROM skill_candidates").fetchone()[0] == 1
    store.close()


def test_static_assessment_evidence_is_deterministic_for_same_source(tmp_path: Path) -> None:
    orchestrator, store, _tools, task_id = _make_blocked_task(tmp_path)
    service = AgentAPIService(orchestrator, token=API_TOKEN)

    first = _post_assessment(service, task_id, _manifest(version=1))
    second = _post_assessment(service, task_id, _manifest(version=2))

    assert first.status_code == second.status_code == 201
    first_body, second_body = first.json(), second.json()
    for key in ("status", "review_status", "source_sha256", "contract", "analysis"):
        assert first_body[key] == second_body[key], key
    assert first_body["execution_attempted"] is second_body["execution_attempted"] is False
    assert first_body["registration_attempted"] is second_body["registration_attempted"] is False
    assert first_body["permissions_granted"] is second_body["permissions_granted"] is False
    state = store.get_task_state(task_id)
    assert state["status"] == "BLOCKED"
    assert state["recovery_state"]["capability_gap"]["requested_tools"] == ["image_generator"]
    store.close()


def test_operator_cannot_assess_or_read_capability_candidates(tmp_path: Path) -> None:
    orchestrator, store, _tools, task_id = _make_blocked_task(tmp_path)
    service = AgentAPIService(orchestrator, token=None)
    service.auth_manager.initialize_owner("ashbm1995@gmail.com", "owner-password-for-tests-123")
    service.auth_manager.create_user("operator@example.test", "operator-password-for-tests-123", "OPERATOR")

    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        login = client.post(
            "/api/v1/auth/login",
            json={"email": "operator@example.test", "password": "operator-password-for-tests-123"},
        )
        assert login.status_code == 200, login.text
        csrf = client.cookies.get("vulnora_csrf")
        assert csrf
        post = client.post(
            f"/api/v1/tasks/{task_id}/capability-candidates/assess",
            headers={"X-CSRF-Token": str(csrf)},
            json=_manifest(),
        )
        get = client.get("/api/v1/capability-candidates/not-a-candidate")

    assert post.status_code == 403, post.text
    assert get.status_code == 403, get.text
    assert store.db.execute("SELECT COUNT(*) FROM skill_candidates").fetchone()[0] == 0
    store.close()
