from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from private_agent.api import AgentAPIService, create_app
from private_agent.auth import AuthManager
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.recovery import RecoveryManager, input_fingerprint
from private_agent.core.task_state import TaskState
from private_agent.observability import ObservabilityMonitor
from private_agent.security import OwnerPermissionRule, SecurityController
from private_agent.storage import Store
from private_agent.worker import BackgroundWorker
from tests.test_verified_input_bindings import BindingPlanner, BindingTool, make_registry

OWNER_EMAIL = "ashbm1995@gmail.com"
OWNER_PASSWORD = "restart-owner-password-for-tests-123"
OPERATOR_EMAIL = "restart-operator@example.com"
OPERATOR_PASSWORD = "restart-operator-password-123"
GOAL = "produce and consume a verified local derived_value"
EXPECTED_INPUT = {"derived_value": "derived:fixture"}
EXPECTED_FINGERPRINT = input_fingerprint(EXPECTED_INPUT)


def _build_components(
    database: Path,
    call_log: dict[str, list[dict[str, Any]]],
) -> tuple[Store, AgentAPIService, Orchestrator, Any]:
    store = Store(database)
    monitor = ObservabilityMonitor(store)
    security = SecurityController(store=store, observability=monitor)
    source = BindingTool("source", calls=call_log["source"])
    consumer = BindingTool("consumer", calls=call_log["consumer"])
    consumer.required_capabilities = ["NETWORK_ACCESS"]
    consumer.declared_capabilities = ["NETWORK_ACCESS"]
    consumer.risk_level = "high"
    tools = make_registry(source, consumer)
    orchestrator = Orchestrator(
        store,
        tools,
        planner=BindingPlanner(),
        security_controller=security,
        observability=monitor,
        recovery_manager=RecoveryManager(max_recovery_attempts=0, max_attempts_per_step=0),
    )
    service = AgentAPIService(orchestrator, token=None)
    return store, service, orchestrator, tools


def _login_as(client: TestClient, email: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    csrf = client.cookies.get("vulnora_csrf")
    assert csrf
    return str(csrf)


def _create_waiting_task(
    store: Store,
    service: AgentAPIService,
    orchestrator: Orchestrator,
    *,
    idempotency_key: str,
    worker_id: str,
) -> tuple[str, str, dict[str, Any]]:
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        csrf = _login_as(client, OWNER_EMAIL, OWNER_PASSWORD)
        created = client.post(
            "/api/v1/tasks",
            json={"goal": GOAL},
            headers={"X-CSRF-Token": csrf, "Idempotency-Key": idempotency_key},
        )
        assert created.status_code == 201, created.text
        task_id = str(created.json()["task_id"])
        waiting = BackgroundWorker(orchestrator, worker_id=worker_id).run_once()
    assert waiting["status"] == "WAITING_APPROVAL", waiting
    assert waiting["task_id"] == task_id
    state = store.get_task_state(task_id)
    assert state is not None and state["status"] == "WAITING_APPROVAL"
    queue = store.task_queue_record(task_id)
    assert queue is not None and queue["queue_status"] == "WAITING_APPROVAL"
    approval_id = str(state["approval_ids"]["consumer-step"])
    return task_id, approval_id, state


def _record_for_step(state: dict[str, Any], step_id: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (key, record)
        for key, record in state["action_records"].items()
        if record.get("step_id") == step_id
    ]
    assert len(matches) == 1, matches
    return matches[0]


def _new_call_log() -> dict[str, list[dict[str, Any]]]:
    # This test harness ledger models an observable local tool boundary that
    # remains available while the agent's Store, service and worker are rebuilt.
    return {"source": [], "consumer": []}


def _write_test_only_state_snapshots(store: Store, state: TaskState) -> None:
    """Simulate validly checksummed on-disk damage without using production writers."""
    state.validate()
    payload = state.to_dict()
    state_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with store._db_lock, store.db:
        checkpoint = store.db.execute(
            "SELECT checkpoint_id FROM task_checkpoints WHERE task_id = ? ORDER BY sequence DESC LIMIT 1",
            (state.task_id,),
        ).fetchone()
        assert checkpoint is not None
        updated = store.db.execute(
            "UPDATE task_states SET state_fingerprint = ?, state_json = ? WHERE task_id = ?",
            (payload["state_fingerprint"], state_json, state.task_id),
        )
        assert updated.rowcount == 1
        updated = store.db.execute(
            "UPDATE task_checkpoints SET state_fingerprint = ?, state_json = ? WHERE checkpoint_id = ?",
            (payload["state_fingerprint"], state_json, checkpoint["checkpoint_id"]),
        )
        assert updated.rowcount == 1
        for action_key, record in state.action_records.items():
            updated = store.db.execute(
                """UPDATE task_action_records
                   SET step_id = ?, tool_name = ?, status = ?, action_json = ?, updated_at = ?
                   WHERE task_id = ? AND action_key = ?""",
                (
                    record.get("step_id", ""),
                    record.get("tool_name", ""),
                    record.get("status", "NOT_STARTED"),
                    json.dumps(record, ensure_ascii=False, sort_keys=True),
                    record.get("updated_at", state.updated_at),
                    state.task_id,
                    action_key,
                ),
            )


def test_bound_approval_and_verified_dependency_survive_restart_and_approval_requeue_crash(
    tmp_path: Path,
) -> None:
    database = tmp_path / "bound-approval-restart.sqlite3"
    call_log = _new_call_log()
    store, service, orchestrator, tools = _build_components(database, call_log)
    owner = AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    AuthManager(store).create_user(
        OPERATOR_EMAIL,
        OPERATOR_PASSWORD,
        "OPERATOR",
        actor_user_id=owner.user_id,
    )

    task_id, approval_id, waiting_state = _create_waiting_task(
        store,
        service,
        orchestrator,
        idempotency_key="bound-restart-e2e",
        worker_id="bound-restart-first-worker",
    )
    waiting_consumer_key, waiting_consumer_record = _record_for_step(waiting_state, "consumer-step")
    assert waiting_consumer_key in waiting_state["pending_actions"]
    assert waiting_consumer_record["status"] == "NOT_STARTED"
    assert waiting_consumer_record["tool_name"] == ""
    assert waiting_consumer_record["approval_id"] == approval_id
    source_key, source_record = _record_for_step(waiting_state, "source-step")
    assert source_record["status"] == "VERIFIED"
    assert source_record["verification"]["status"] == "VERIFIED"
    assert source_record["verification"]["verified"] is True
    assert source_record["output"]["output"] == {"derived_value": "derived:fixture"}
    assert waiting_state["verification_results"][source_key]["verified"] is True
    assert call_log["source"] == [{"seed": "fixture"}]
    assert call_log["consumer"] == []
    assert store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL"
    store.close()

    # Reopen SQLite and reconstruct the service, orchestrator, registry and worker.
    store, service, orchestrator, tools = _build_components(database, call_log)
    reopened_state = store.get_task_state(task_id)
    assert reopened_state is not None
    assert reopened_state["status"] == "WAITING_APPROVAL"
    assert reopened_state["action_records"][source_key] == source_record
    assert store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL"
    worker = BackgroundWorker(orchestrator, worker_id="bound-restart-second-worker")

    # A pending approval remains unclaimable after restart; neither action repeats.
    assert worker.run_once()["status"] == "NO_TASK"
    assert store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL"
    assert call_log["source"] == [{"seed": "fixture"}]
    assert call_log["consumer"] == []

    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        operator_csrf = _login_as(client, OPERATOR_EMAIL, OPERATOR_PASSWORD)
        foreign_role_attempt = client.post(
            f"/api/v1/approvals/{approval_id}/approve",
            json={},
            headers={"X-CSRF-Token": operator_csrf},
        )
        assert foreign_role_attempt.status_code == 403, foreign_role_attempt.text
        assert store.get_approval_request(approval_id)["status"] == "PENDING"

        owner_csrf = _login_as(client, OWNER_EMAIL, OWNER_PASSWORD)
        approval_response = client.get(f"/api/v1/approvals/{approval_id}")
        assert approval_response.status_code == 200, approval_response.text
        approval = approval_response.json()
        assert approval["input_fingerprint"] == EXPECTED_FINGERPRINT
        assert approval["input_fingerprint"] != input_fingerprint({})
        assert approval["status"] == "PENDING"

        approved = client.post(
            f"/api/v1/approvals/{approval_id}/approve",
            json={},
            headers={"X-CSRF-Token": owner_csrf},
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["status"] == "APPROVED"
        # No worker is attached to this API process: approval is durable while
        # the queue is still waiting, representing a crash before requeue.
        assert store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL"
    store.close()

    # A second restart crosses the approval-persisted / queue-requeued boundary.
    store, service, orchestrator, tools = _build_components(database, call_log)
    assert store.get_approval_request(approval_id)["status"] == "APPROVED"
    assert store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL"
    restarted_worker = BackgroundWorker(orchestrator, worker_id="bound-restart-third-worker")
    restarted_worker._reconcile_approved_waiting_tasks()
    assert store.task_queue_record(task_id)["queue_status"] == "QUEUED"

    resumed = restarted_worker.run_once()
    assert resumed["status"] == "COMPLETED", resumed
    assert resumed["task_id"] == task_id
    assert store.task_queue_record(task_id)["queue_status"] == "COMPLETED"
    assert call_log["source"] == [{"seed": "fixture"}]
    assert call_log["consumer"] == [EXPECTED_INPUT]

    completed_state = store.get_task_state(task_id)
    assert completed_state is not None
    consumer_key, consumer_record = _record_for_step(completed_state, "consumer-step")
    assert consumer_record["status"] == "VERIFIED"
    assert consumer_record["verification"]["status"] == "VERIFIED"
    assert consumer_record["verification"]["verified"] is True
    assert completed_state["verification_results"][consumer_key]["status"] == "VERIFIED"
    decision = completed_state["security_decisions"]["consumer-step"]
    assert decision["decision"] == "ALLOW"
    assert decision["input_fingerprint"] == EXPECTED_FINGERPRINT
    assert completed_state["status"] == "COMPLETED"
    assert completed_state["result"]["final_status"] == "COMPLETED"
    store.close()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_restart_fails_closed_when_verified_dependency_result_is_missing_or_corrupt(
    tmp_path: Path,
    damage: str,
) -> None:
    database = tmp_path / f"bound-result-{damage}.sqlite3"
    call_log = _new_call_log()
    store, service, orchestrator, _ = _build_components(database, call_log)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    task_id, approval_id, waiting_state = _create_waiting_task(
        store,
        service,
        orchestrator,
        idempotency_key=f"bound-result-{damage}",
        worker_id=f"bound-result-{damage}-first-worker",
    )
    source_key, _ = _record_for_step(waiting_state, "source-step")
    assert call_log["source"] == [{"seed": "fixture"}]
    store.close()

    store, service, orchestrator, _ = _build_components(database, call_log)
    damaged = TaskState.from_dict(store.get_task_state(task_id))
    source_record = damaged.action_records[source_key]
    if damage == "missing":
        source_record.pop("output")
    else:
        source_record["output"]["output"]["derived_value"] = "tampered-after-verification"
    damaged.touch()
    damaged.validate()
    _write_test_only_state_snapshots(store, damaged)
    assert TaskState.from_dict(store.get_task_state(task_id)).state_fingerprint == damaged.state_fingerprint

    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        csrf = _login_as(client, OWNER_EMAIL, OWNER_PASSWORD)
        approved = client.post(
            f"/api/v1/approvals/{approval_id}/approve",
            json={},
            headers={"X-CSRF-Token": csrf},
        )
        assert approved.status_code == 200, approved.text
    resumed = BackgroundWorker(orchestrator, worker_id=f"bound-result-{damage}-second-worker").run_once()
    assert resumed["status"] == "BLOCKED", resumed
    assert store.get_task_state(task_id)["status"] == "BLOCKED"
    assert store.task_queue_record(task_id)["queue_status"] == "BLOCKED"
    assert call_log["source"] == [{"seed": "fixture"}]
    assert call_log["consumer"] == []
    store.close()


def test_requeue_rejects_approved_foreign_task_approval_after_restart(tmp_path: Path) -> None:
    database = tmp_path / "foreign-approval-restart.sqlite3"
    call_log = _new_call_log()
    store, service, orchestrator, _ = _build_components(database, call_log)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    first_task_id, first_approval_id, _ = _create_waiting_task(
        store,
        service,
        orchestrator,
        idempotency_key="foreign-approval-first",
        worker_id="foreign-approval-worker-first",
    )
    second_task_id, foreign_approval_id, _ = _create_waiting_task(
        store,
        service,
        orchestrator,
        idempotency_key="foreign-approval-second",
        worker_id="foreign-approval-worker-second",
    )
    assert first_approval_id != foreign_approval_id
    assert call_log["consumer"] == []
    store.close()

    store, service, orchestrator, _ = _build_components(database, call_log)
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        csrf = _login_as(client, OWNER_EMAIL, OWNER_PASSWORD)
        approved = client.post(
            f"/api/v1/approvals/{foreign_approval_id}/approve",
            json={},
            headers={"X-CSRF-Token": csrf},
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["status"] == "APPROVED"

    first_state = TaskState.from_dict(store.get_task_state(first_task_id))
    first_state.approval_ids["consumer-step"] = foreign_approval_id
    first_state.touch()
    first_state.validate()
    _write_test_only_state_snapshots(store, first_state)

    with pytest.raises(ValueError, match="approval_not_approved"):
        store.requeue_waiting_approval_task(first_task_id)
    assert store.get_approval_request(foreign_approval_id)["task_id"] == second_task_id
    assert store.task_queue_record(first_task_id)["queue_status"] == "WAITING_APPROVAL"
    assert store.task_queue_record(second_task_id)["queue_status"] == "WAITING_APPROVAL"
    assert call_log["consumer"] == []
    store.close()


def test_policy_stale_approval_after_restart_never_executes_bound_action(tmp_path: Path) -> None:
    database = tmp_path / "stale-approval-restart.sqlite3"
    call_log = _new_call_log()
    store, service, orchestrator, _ = _build_components(database, call_log)
    owner = AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    task_id, approval_id, _ = _create_waiting_task(
        store,
        service,
        orchestrator,
        idempotency_key="stale-approval-restart",
        worker_id="stale-approval-first-worker",
    )
    assert call_log["source"] == [{"seed": "fixture"}]
    store.close()

    store, service, orchestrator, _ = _build_components(database, call_log)
    store.create_owner_permission_rule(
        OwnerPermissionRule(
            "restart-stale-deny-consumer",
            "DENY",
            "TOOL",
            "consumer",
            reason="deny bound consumer after restart",
        ),
        actor=owner.user_id,
        reason="invalidate the pending approval before resume",
    )
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        csrf = _login_as(client, OWNER_EMAIL, OWNER_PASSWORD)
        approved = client.post(
            f"/api/v1/approvals/{approval_id}/approve",
            json={},
            headers={"X-CSRF-Token": csrf},
        )
        assert approved.status_code == 409, approved.text
        assert approved.json()["error_code"] == "approval_stale"
        assert store.get_approval_request(approval_id)["status"] == "PENDING"

    waiting = BackgroundWorker(orchestrator, worker_id="stale-approval-second-worker").run_once()
    assert waiting["status"] == "NO_TASK", waiting
    assert store.get_approval_request(approval_id)["status"] == "PENDING"
    assert store.get_task_state(task_id)["status"] == "WAITING_APPROVAL"
    assert store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL"
    assert call_log["source"] == [{"seed": "fixture"}]
    assert call_log["consumer"] == []
    store.close()


@pytest.mark.parametrize("provider_failure", ["missing", "error"])
def test_approval_policy_provider_failure_is_fail_closed_and_worker_survives(
    tmp_path: Path,
    provider_failure: str,
) -> None:
    database = tmp_path / f"policy-provider-{provider_failure}.sqlite3"
    call_log = _new_call_log()
    store, service, orchestrator, _ = _build_components(database, call_log)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    task_id, approval_id, _ = _create_waiting_task(
        store,
        service,
        orchestrator,
        idempotency_key=f"policy-provider-{provider_failure}",
        worker_id=f"policy-provider-{provider_failure}-first-worker",
    )
    gate = orchestrator.security.approval_gate
    if provider_failure == "missing":
        gate.set_policy_version_provider(None)
    else:
        def fail_policy_lookup() -> int:
            raise RuntimeError("local policy lookup unavailable")

        gate.set_policy_version_provider(fail_policy_lookup)

    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        csrf = _login_as(client, OWNER_EMAIL, OWNER_PASSWORD)
        response = client.post(
            f"/api/v1/approvals/{approval_id}/approve",
            json={},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 503, response.text
        assert response.json()["error_code"] == "approval_policy_unavailable"

    assert store.get_approval_request(approval_id)["status"] == "PENDING"
    assert store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL"
    worker = BackgroundWorker(orchestrator, worker_id=f"policy-provider-{provider_failure}-worker")
    assert worker.run_once()["status"] == "NO_TASK"
    assert store.get_approval_request(approval_id)["status"] == "PENDING"
    assert call_log["source"] == [{"seed": "fixture"}]
    assert call_log["consumer"] == []
    store.close()


def test_missing_policy_provider_does_not_persist_expiration_transition(tmp_path: Path) -> None:
    database = tmp_path / "missing-policy-provider-expiry.sqlite3"
    call_log = _new_call_log()
    store, service, orchestrator, _ = _build_components(database, call_log)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    _, approval_id, _ = _create_waiting_task(
        store,
        service,
        orchestrator,
        idempotency_key="missing-policy-provider-expiry",
        worker_id="missing-policy-provider-expiry-first-worker",
    )
    approval = store.get_approval_request(approval_id)
    assert approval is not None
    approval["expires_at"] = "2000-01-01T00:00:00+00:00"
    with store._db_lock:
        store.db.execute(
            "UPDATE approval_requests SET expires_at = ?, approval_json = ? WHERE approval_id = ?",
            (approval["expires_at"], json.dumps(approval, sort_keys=True), approval_id),
        )
        store.db.commit()

    gate = orchestrator.security.approval_gate
    gate.set_policy_version_provider(None)
    with pytest.raises(ValueError, match="approval_policy_unavailable"):
        gate.approve(approval_id, actor="owner")

    assert store.get_approval_request(approval_id)["status"] == "PENDING"
    assert call_log["consumer"] == []
    store.close()


def test_approval_compare_and_set_rejects_policy_change_between_check_and_commit(tmp_path: Path) -> None:
    database = tmp_path / "policy-version-race.sqlite3"
    call_log = _new_call_log()
    store, service, orchestrator, _ = _build_components(database, call_log)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    task_id, approval_id, _ = _create_waiting_task(
        store,
        service,
        orchestrator,
        idempotency_key="policy-version-race",
        worker_id="policy-version-race-first-worker",
    )
    gate = orchestrator.security.approval_gate
    original_version = orchestrator.security.policy_engine._current_policy_version()
    observed_versions = iter((original_version, original_version + 1))
    gate.set_policy_version_provider(lambda: next(observed_versions))

    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        csrf = _login_as(client, OWNER_EMAIL, OWNER_PASSWORD)
        response = client.post(
            f"/api/v1/approvals/{approval_id}/approve",
            json={},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 409, response.text
        assert response.json()["error_code"] == "approval_stale"

    assert store.get_approval_request(approval_id)["status"] == "PENDING"
    assert store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL"
    assert call_log["consumer"] == []
    store.close()


def test_previously_approved_task_is_not_requeued_after_policy_changes(tmp_path: Path) -> None:
    database = tmp_path / "approved-then-stale-requeue.sqlite3"
    call_log = _new_call_log()
    store, service, orchestrator, _ = _build_components(database, call_log)
    owner = AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    task_id, approval_id, _ = _create_waiting_task(
        store,
        service,
        orchestrator,
        idempotency_key="approved-then-stale-requeue",
        worker_id="approved-then-stale-first-worker",
    )
    approved = orchestrator.security.approval_gate.approve(approval_id, actor=owner.user_id)
    assert approved.status == "APPROVED"
    store.create_owner_permission_rule(
        OwnerPermissionRule(
            "invalidate-already-approved-consumer",
            "DENY",
            "TOOL",
            "consumer",
            reason="invalidate approval before recovery requeue",
        ),
        actor=owner.user_id,
        reason="advance policy after the approval was recorded",
    )

    worker = BackgroundWorker(orchestrator, worker_id="approved-then-stale-recovery-worker")
    assert worker.run_once()["status"] == "NO_TASK"
    assert store.get_approval_request(approval_id)["status"] == "APPROVED"
    assert store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL"

    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        csrf = _login_as(client, OWNER_EMAIL, OWNER_PASSWORD)
        response = client.post(
            f"/api/v1/tasks/{task_id}/resume",
            json={},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 409, response.text
        assert response.json()["error_code"] == "approval_stale"

    assert store.get_approval_request(approval_id)["status"] == "APPROVED"
    assert store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL"
    assert call_log["source"] == [{"seed": "fixture"}]
    assert call_log["consumer"] == []
    store.close()
