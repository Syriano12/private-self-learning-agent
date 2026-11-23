from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from private_agent.auth import AuthManager
from private_agent.api import AgentAPIService, create_app
from private_agent.core.llm import LLMResponse
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import Planner
from private_agent.core.task_state import TaskState
from private_agent.storage import Store, now
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


API_TOKEN = "capability-gap-owner-token-2026"


class StaticPlanProvider:
    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.calls = 0

    def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict[str, Any]) -> LLMResponse:
        self.calls += 1
        return LLMResponse(self.plan, json.dumps(self.plan), "capability-gap-fixture")


class CountingReadTool:
    name = "registered_read"
    permission_level = "public_read"
    planned_effect = "bounded local read"
    risk_level = "low"
    required_capabilities: list[str] = []
    declared_capabilities: list[str] = []
    network_required = False

    def __init__(self) -> None:
        self.calls = 0

    def description(self) -> str:
        return "A bounded synthetic read tool for tests"

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        }

    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        }

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        self.calls += 1
        return ToolResult.success({"result": f"read:{inputs['value']}"})


def _plan(goal: str, tool_name: str) -> dict[str, Any]:
    return {
        "goal": goal,
        "rationale": "Use the proposed capability to satisfy the synthetic goal.",
        "steps": [
            {
                "id": "capability-step",
                "objective": "Perform the synthetic bounded action",
                "tool": tool_name,
                "input": {"value": "safe"},
                "reason": "The synthetic fixture proposes this tool.",
                "depends_on": [],
                "success_criteria": {"required_fields": ["result"]},
            }
        ],
    }


def test_unregistered_tool_request_blocks_before_execution_and_is_durable_and_visible_over_http(
    tmp_path: Path,
) -> None:
    goal = "Create a synthetic image using image_generator."
    registered_tool = CountingReadTool()
    registry = ToolRegistry()
    registry.register(registered_tool)
    provider = StaticPlanProvider(_plan(goal, "image_generator"))
    db_path = tmp_path / "capability-gap.sqlite3"
    store = Store(db_path)
    orchestrator = Orchestrator(store, registry, planner=Planner(provider))

    result = orchestrator.run(goal)

    assert provider.calls == 1
    assert registered_tool.calls == 0
    assert result["final_status"] == "BLOCKED"
    assert result["attempts"] == 0
    gap = result["capability_gap"]
    assert gap["code"] == "UNREGISTERED_TOOL_REQUEST"
    assert gap["source"] == "planner_registry_validation"
    assert gap["requested_tools"] == ["image_generator"]
    assert gap["unavailable_tool_count"] == 1
    assert gap["registered_tools"] == ["registered_read"]
    assert gap["execution_attempted"] is False
    assert gap["acquisition_attempted"] is False
    assert gap["permissions_granted"] is False

    task_id = result["task_id"]
    persisted = store.get_task_state(task_id)
    assert persisted is not None
    assert persisted["status"] == "BLOCKED"
    assert persisted["current_phase"] == "CAPABILITY_GAP"
    assert persisted["recovery_state"] == {"reason": "capability_unavailable", "capability_gap": gap}
    store.close()

    reopened = Store(db_path)
    reopened_state = reopened.get_task_state(task_id)
    assert reopened_state is not None
    assert reopened_state["status"] == "BLOCKED"
    assert reopened_state["recovery_state"]["capability_gap"] == gap
    service = AgentAPIService(
        Orchestrator(reopened, registry, planner=Planner(provider)),
        token=API_TOKEN,
    )
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        response = client.get(
            f"/api/v1/tasks/{task_id}",
            headers={"Authorization": f"Bearer {API_TOKEN}"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "BLOCKED"
    assert body["current_phase"] == "CAPABILITY_GAP"
    assert body["recovery_state"]["reason"] == "capability_unavailable"
    response_gap = body["recovery_state"]["capability_gap"]
    for key, value in gap.items():
        assert response_gap[key] == value
    assert response_gap["candidate_lifecycle"]["promotion_status"] == "BLOCKED"
    assert response_gap["candidate_lifecycle"]["candidates"][0]["status"] == "NOT_PROPOSED"
    assert body["recovery_state"]["capability_gap"]["execution_attempted"] is False
    assert body["recovery_state"]["capability_gap"]["acquisition_attempted"] is False
    reopened.close()


def test_registered_public_read_tool_keeps_normal_plan_and_execution_path(tmp_path: Path) -> None:
    goal = "Read the synthetic value with registered_read."
    tool = CountingReadTool()
    registry = ToolRegistry()
    registry.register(tool)
    provider = StaticPlanProvider(_plan(goal, tool.name))
    store = Store(tmp_path / "registered-tool.sqlite3")
    orchestrator = Orchestrator(store, registry, planner=Planner(provider))

    result = orchestrator.run(goal)

    assert provider.calls == 1
    assert tool.calls == 1
    assert result["final_status"] == "COMPLETED"
    assert result["verification"]["status"] == "VERIFIED"
    assert result["step_results"][0]["output"] == {"result": "read:safe"}
    assert "capability_gap" not in result
    persisted = store.get_task_state(result["task_id"])
    assert persisted is not None
    assert persisted["status"] == "COMPLETED"
    assert persisted["recovery_state"] == {}
    store.close()


def test_empty_registry_records_blocker_without_querying_provider_or_executing_tools(tmp_path: Path) -> None:
    goal = "Perform a synthetic task with no registered tools."
    store = Store(tmp_path / "empty-registry.sqlite3")
    orchestrator = Orchestrator(store, ToolRegistry(), planner=Planner())

    result = orchestrator.run(goal)

    assert result["final_status"] == "BLOCKED"
    assert result["capability_gap"]["code"] == "NO_REGISTERED_TOOLS"
    assert result["capability_gap"]["registered_tools"] == []
    assert result["capability_gap"]["execution_attempted"] is False
    assert result["capability_gap"]["acquisition_attempted"] is False
    persisted = store.get_task_state(result["task_id"])
    assert persisted is not None
    assert persisted["status"] == "BLOCKED"
    assert persisted["recovery_state"]["reason"] == "capability_unavailable"
    store.close()


def test_specialized_bug_bounty_route_blocks_when_selected_scanner_is_not_registered(tmp_path: Path) -> None:
    registered_tool = CountingReadTool()
    registry = ToolRegistry()
    registry.register(registered_tool)
    provider = StaticPlanProvider(_plan("Run bounded reconnaissance", "httpx_recon"))
    db_path = tmp_path / "bug-bounty-capability-gap.sqlite3"
    store = Store(db_path)
    orchestrator = Orchestrator(store, registry, planner=Planner(provider))

    result = orchestrator.run(
        "Run bounded reconnaissance",
        task_context={
            "task_kind": "bug_bounty",
            "bugbounty_request": {
                "scanner": "httpx",
                "scope": {"program_id": "synthetic-local-fixture"},
                "target": "https://example.invalid/health",
                "identity": "synthetic",
                "resource_id": "synthetic-resource",
            },
        },
    )

    assert provider.calls == 0
    assert registered_tool.calls == 0
    assert result["final_status"] == "BLOCKED"
    assert result["attempts"] == 0
    assert result["capability_gap"]["code"] == "UNREGISTERED_TOOL_REQUEST"
    assert result["capability_gap"]["requested_tools"] == ["httpx_recon"]
    assert result["capability_gap"]["execution_attempted"] is False
    assert result["capability_gap"]["acquisition_attempted"] is False
    persisted = store.get_task_state(result["task_id"])
    assert persisted is not None
    assert persisted["status"] == "BLOCKED"
    assert persisted["recovery_state"]["capability_gap"]["requested_tools"] == ["httpx_recon"]

    store.close()
    reopened = Store(db_path)
    persisted_after_restart = reopened.get_task_state(result["task_id"])
    assert persisted_after_restart is not None
    assert persisted_after_restart["recovery_state"] == persisted["recovery_state"]
    marker = "INERT_BUG_BOUNTY_RECOVERY_CANARY_941c"
    state = TaskState.from_dict(persisted_after_restart)
    state.recovery_state["diagnosis"] = {"details": {"raw_tool_diagnostic": marker}}
    state.recovery_state["capability_gap"]["candidate_lifecycle"] = {"internal_debug": marker}
    state.recovery_state["capability_gap"]["unrecognized_raw_field"] = marker
    state.recovery_state["unrecognized_raw_field"] = marker
    state.touch()
    reopened.save_task_state(state, "TEST_BUG_BOUNTY_RECOVERY_REDACTION")
    persisted_after_restart = reopened.get_task_state(result["task_id"])
    assert persisted_after_restart is not None
    assert marker in repr(persisted_after_restart["recovery_state"])
    service = AgentAPIService(Orchestrator(reopened, registry, planner=Planner(provider)), token=API_TOKEN)
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        general = client.get(f"/api/v1/tasks/{result['task_id']}", headers=headers)
        specialized = client.get(f"/api/v1/bug-bounty/tasks/{result['task_id']}", headers=headers)

    assert general.status_code == 200, general.text
    assert specialized.status_code == 200, specialized.text
    assert marker not in general.text
    assert marker not in specialized.text
    general_body = general.json()
    specialized_body = specialized.json()
    assert general_body["status"] == specialized_body["status"] == "BLOCKED"
    assert general_body["current_phase"] == "CAPABILITY_GAP"
    assert specialized_body["recovery_state"] == general_body["recovery_state"]
    assert specialized_body["recovery_state"]["reason"] == persisted_after_restart["recovery_state"]["reason"]
    gap = specialized_body["recovery_state"]["capability_gap"]
    assert gap["code"] == "UNREGISTERED_TOOL_REQUEST"
    assert gap["summary"] == "The planner requested a tool that is not registered; no tool was executed or installed."
    assert gap["requested_tools"] == ["httpx_recon"]
    assert gap["execution_attempted"] is False
    assert gap["acquisition_attempted"] is False
    assert gap["permissions_granted"] is False
    assert gap["candidate_lifecycle"]["promotion_status"] == "BLOCKED"
    assert gap["candidate_lifecycle"]["isolation_status"] == "UNAVAILABLE"
    assert gap["candidate_lifecycle"]["candidates"][0]["status"] == "NOT_PROPOSED"
    assert "sandbox_execution" in gap["candidate_lifecycle"]["blocking_gates"]
    assert specialized_body["run"] is None
    assert specialized_body["scan"] is None
    assert specialized_body["report"] == ""
    reopened.close()


def test_repeated_unregistered_tool_requests_remain_a_single_truthful_capability_gap(tmp_path: Path) -> None:
    goal = "Complete two synthetic actions with unavailable_tool."
    plan = _plan(goal, "unavailable_tool")
    plan["steps"].append(
        {
            "id": "capability-step-2",
            "objective": "Repeat the synthetic action",
            "tool": "unavailable_tool",
            "input": {"value": "safe"},
            "reason": "The same absent capability is proposed again.",
            "depends_on": [],
            "success_criteria": {"required_fields": ["result"]},
        }
    )
    registered_tool = CountingReadTool()
    registry = ToolRegistry()
    registry.register(registered_tool)
    provider = StaticPlanProvider(plan)
    store = Store(tmp_path / "duplicate-capability-gap.sqlite3")
    orchestrator = Orchestrator(store, registry, planner=Planner(provider))

    result = orchestrator.run(goal)

    assert provider.calls == 1
    assert registered_tool.calls == 0
    assert result["final_status"] == "BLOCKED"
    assert result["capability_gap"]["requested_tools"] == ["unavailable_tool"]
    assert result["capability_gap"]["unavailable_tool_count"] == 1
    assert result["capability_gap"]["execution_attempted"] is False
    store.close()


def test_task_detail_api_omits_raw_recovery_diagnostics_and_unknown_fields(tmp_path: Path) -> None:
    goal = "Read a synthetic value with registered_read."
    tool = CountingReadTool()
    registry = ToolRegistry()
    registry.register(tool)
    provider = StaticPlanProvider(_plan(goal, tool.name))
    store = Store(tmp_path / "recovery-redaction.sqlite3")
    orchestrator = Orchestrator(store, registry, planner=Planner(provider))
    result = orchestrator.run(goal)
    task_id = result["task_id"]

    marker = "INERT_RAW_TOOL_DIAGNOSTIC_CANARY_7f19"
    persisted = store.get_task_state(task_id)
    assert persisted is not None
    state = TaskState.from_dict(persisted)
    state.recovery_state = {
        "reason": f"persisted_plan_invalid:ValueError:{marker}",
        "decision": {
            "strategy": "RETRY",
            "reason": marker,
            "step_id": "safe-step",
            "original_tool": "registered_read",
            "modified_input": {"value": marker},
            "details": {"stderr": marker},
        },
        "diagnosis": {
            "failure_type": "TOOL_ERROR",
            "reason": marker,
            "failed_step": "safe-step",
            "tool_name": "registered_read",
            "recoverable": True,
            "suggested_recovery": "RETRY",
            "evidence": [{"type": "execution_error", "value": {"stderr": marker}}],
            "details": {"debug": marker},
        },
        "round": 1,
        "unrecognized_internal_field": marker,
    }
    state.touch()
    store.save_task_state(state, "TEST_RECOVERY_DIAGNOSTIC_REDACTION")
    assert marker in repr(store.get_task_state(task_id)["recovery_state"])

    service = AgentAPIService(orchestrator, token=API_TOKEN)
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        response = client.get(
            f"/api/v1/tasks/{task_id}",
            headers={"Authorization": f"Bearer {API_TOKEN}"},
        )

    assert response.status_code == 200, response.text
    assert marker not in response.text
    recovery = response.json()["recovery_state"]
    assert recovery == {
        "reason": "persisted_plan_invalid",
        "decision": {"strategy": "RETRY", "step_id": "safe-step", "original_tool": "registered_read"},
        "diagnosis": {
            "failure_type": "TOOL_ERROR",
            "failed_step": "safe-step",
            "tool_name": "registered_read",
            "recoverable": True,
            "suggested_recovery": "RETRY",
        },
        "round": 1,
    }
    store.close()


def test_bug_bounty_recovery_state_parity_after_sqlite_reopen_for_owner(
    tmp_path: Path, request: Any
) -> None:
    db_path = tmp_path / "bug-bounty-recovery-state.sqlite3"
    store = Store(db_path)
    request.addfinalizer(store.close)
    auth = AuthManager(store)
    owner = auth.initialize_owner("ashbm1995@gmail.com", "owner-password-for-tests-123")
    operator = auth.create_user(
        "recovery-state-operator@example.com",
        "operator-password-for-tests-123",
        "OPERATOR",
        actor_user_id=owner.user_id,
    )

    registered_tool = CountingReadTool()
    registry = ToolRegistry()
    registry.register(registered_tool)
    provider = StaticPlanProvider(_plan("Run bounded reconnaissance", "httpx_recon"))
    orchestrator = Orchestrator(store, registry, planner=Planner(provider))
    bug_bounty_context = {
        "task_kind": "bug_bounty",
        "bugbounty_request": {
            "scanner": "httpx",
            "scope": {"program_id": "synthetic-local-fixture"},
            "target": "https://example.invalid/health",
            "identity": "synthetic",
            "resource_id": "synthetic-resource",
        },
    }
    blocked = orchestrator.run("Run bounded reconnaissance", task_context=bug_bounty_context)
    blocked_task_id = blocked["task_id"]
    persisted_nonempty = store.get_task_state(blocked_task_id)
    assert persisted_nonempty is not None
    assert persisted_nonempty["recovery_state"]

    empty_task = store.create_queued_task(
        "Synthetic Bug Bounty task with no recovery state",
        initial_message="Synthetic Bug Bounty task with no recovery state",
        context=bug_bounty_context,
        owner_user_id=owner.user_id,
    )
    empty_task_id = str(empty_task["task_id"])
    # Direct Orchestrator.run() bypasses API owner assignment; add the same owner record for this access-control fixture.
    with store._db_lock, store.db:
        store.db.execute(
            "INSERT INTO task_owners (task_id, user_id, created_at) VALUES (?, ?, ?)",
            (blocked_task_id, owner.user_id, now()),
        )
    store.close()

    reopened = Store(db_path)
    request.addfinalizer(reopened.close)
    reopened_nonempty = reopened.get_task_state(blocked_task_id)
    reopened_empty = reopened.get_task_state(empty_task_id)
    assert reopened_nonempty is not None
    assert reopened_nonempty["recovery_state"] == persisted_nonempty["recovery_state"]
    assert reopened_empty is not None
    assert reopened_empty["recovery_state"] == {}

    service = AgentAPIService(
        Orchestrator(reopened, registry, planner=Planner(provider)),
        token=None,
    )
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        owner_login = client.post(
            "/api/v1/auth/login",
            json={"email": "ashbm1995@gmail.com", "password": "owner-password-for-tests-123"},
        )
        assert owner_login.status_code == 200, owner_login.text
        assert owner_login.json()["user"]["user_id"] == owner.user_id
        assert owner_login.json()["user"]["role"] == "OWNER"

        for task_id, persisted in (
            (blocked_task_id, reopened_nonempty),
            (empty_task_id, reopened_empty),
        ):
            general = client.get(f"/api/v1/tasks/{task_id}")
            specialized = client.get(f"/api/v1/bug-bounty/tasks/{task_id}")
            assert general.status_code == 200, general.text
            assert specialized.status_code == 200, specialized.text
            general_recovery_state = general.json()["recovery_state"]
            specialized_recovery_state = specialized.json()["recovery_state"]
            assert general_recovery_state == specialized_recovery_state
            if persisted["recovery_state"]:
                assert set(general_recovery_state) == set(persisted["recovery_state"])
                assert general_recovery_state["reason"] == persisted["recovery_state"]["reason"]
                stored_gap = persisted["recovery_state"]["capability_gap"]
                response_gap = general_recovery_state["capability_gap"]
                assert {
                    key: value for key, value in response_gap.items() if key != "candidate_lifecycle"
                } == stored_gap
                assert response_gap["candidate_lifecycle"]["candidates"][0]["target_tool"] == "httpx_recon"
            else:
                assert general_recovery_state == {}
                assert specialized_recovery_state == {}

        client.cookies.clear()
        operator_login = client.post(
            "/api/v1/auth/login",
            json={"email": "recovery-state-operator@example.com", "password": "operator-password-for-tests-123"},
        )
        assert operator_login.status_code == 200, operator_login.text
        assert operator_login.json()["user"]["user_id"] == operator.user_id
        for task_id in (blocked_task_id, empty_task_id):
            assert client.get(f"/api/v1/tasks/{task_id}").status_code == 404
            assert client.get(f"/api/v1/bug-bounty/tasks/{task_id}").status_code == 404
    reopened.close()
