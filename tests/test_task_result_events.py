from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import pytest

from private_agent import security as security_module
from private_agent import storage as storage_module
from private_agent.auth import AuthManager, AuthPrincipal, ROLE_OPERATOR
from private_agent.core.orchestrator import TaskInterrupted
from private_agent.storage import Store, UnsupportedSchemaVersion
from private_agent.tools.contracts import ToolResult
from private_agent.worker import BackgroundWorker
from tests.test_phase12_api import APITestTool, client_for, make_service
from private_agent.api import APIError


def _owner_and_principal(store: Store, *, suffix: str = "result") -> tuple[str, AuthPrincipal]:
    auth = AuthManager(store, owner_email="owner@example.test")
    row = store.db.execute("SELECT user_id FROM auth_users WHERE role = 'OWNER' LIMIT 1").fetchone()
    owner_user_id = str(row["user_id"]) if row else auth.initialize_owner(
        "owner@example.test", "strong-owner-password"
    ).user_id
    user = auth.create_user(
        f"{suffix}-operator@example.test",
        "strong-operator-password",
        ROLE_OPERATOR,
        actor_user_id=owner_user_id,
    )
    return user.user_id, AuthPrincipal(
        subject=user.user_id,
        user_id=user.user_id,
        email=user.email,
        role=ROLE_OPERATOR,
    )


def _make_owned_task(store: Store, owner_user_id: str, goal: str) -> str:
    created = store.create_queued_task(goal, initial_message=goal, owner_user_id=owner_user_id)
    return str(created["task_id"])


def test_schema_14_migrates_to_15_and_installs_immutable_result_event_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "schema-14.sqlite3"
    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 14)
    old = Store(path)
    assert old.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'task_result_events'"
    ).fetchone() is None
    old.close()

    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 15)
    upgraded = Store(path)
    assert upgraded.db.execute(
        "SELECT schema_version FROM task_state_schema ORDER BY schema_version DESC LIMIT 1"
    ).fetchone()["schema_version"] == 15
    assert upgraded.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = 'task_result_events_no_update'"
    ).fetchone()
    assert upgraded.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = 'task_result_events_no_delete'"
    ).fetchone()
    upgraded.close()


def test_schema_15_rejects_a_misdefined_owner_guard_trigger_on_reopen(tmp_path: Path) -> None:
    path = tmp_path / "schema-15-trigger-drift.sqlite3"
    store = Store(path)
    store.db.execute("DROP TRIGGER task_result_events_owner_matches_task")
    store.db.execute(
        "CREATE TRIGGER task_result_events_owner_matches_task "
        "BEFORE INSERT ON task_result_events BEGIN SELECT 1; END"
    )
    store.db.commit()
    store.close()
    with pytest.raises(UnsupportedSchemaVersion, match="invalid_trigger:task_result_events_owner_matches_task"):
        Store(path)


def test_worker_result_event_is_owner_bound_idempotent_immutable_and_survives_reopen(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    owner_user_id, owner_principal = _owner_and_principal(store)
    task_id = _make_owned_task(store, owner_user_id, "complete a verified safe task")
    initial_conversation = service.get_task_conversation(task_id, principal=owner_principal)
    assert initial_conversation.result_events[-1].task_status == "QUEUED"

    result = BackgroundWorker(service.orchestrator, worker_id="result-event-worker").run_once()
    assert result["status"] == "COMPLETED"
    conversation = service.get_task_conversation(task_id, principal=owner_principal)
    assert conversation.assistant_reply_supported is False
    assert conversation.messages[0].role == "USER"
    assert conversation.result_events
    final_event = conversation.result_events[-1]
    assert final_event.role == "SYSTEM"
    assert final_event.message_kind == "RESULT"
    assert final_event.task_status == "COMPLETED"
    assert final_event.verification_status == "VERIFIED"
    assert final_event.result_available is True
    assert final_event.verified_action_count == 1
    assert len(final_event.result_digest) == 64
    assert "step_results" not in json.dumps(final_event.model_dump())
    assert store.task_result_events(task_id, owner_user_id="different-owner") == []

    other_user_id, other_principal = _owner_and_principal(store, suffix="other")
    assert other_user_id != owner_user_id
    with pytest.raises(APIError) as denied:
        service.get_task_conversation(task_id, principal=other_principal)
    assert denied.value.status_code == 404
    stored_event = dict(store.db.execute(
        "SELECT * FROM task_result_events WHERE task_id = ? ORDER BY sequence DESC LIMIT 1", (task_id,)
    ).fetchone())
    stored_event.update({
        "event_id": "spoofed-owner-event",
        "owner_user_id": other_user_id,
        "sequence": int(stored_event["sequence"]) + 1,
        "transition_fingerprint": "spoofed-owner-transition",
    })
    columns = list(stored_event)
    with pytest.raises(sqlite3.IntegrityError, match="task_result_events_owner_mismatch"):
        store.db.execute(
            f"INSERT INTO task_result_events ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
            tuple(stored_event[column] for column in columns),
        )
    store.db.rollback()

    state = store.get_task_state(task_id)
    before = len(store.task_result_events(task_id, owner_user_id=owner_user_id))
    with store._db_lock:
        store.db.execute("BEGIN IMMEDIATE")
        store._persist_task_result_event_in_transaction(state, "COMPLETED")
        store.db.commit()
    assert len(store.task_result_events(task_id, owner_user_id=owner_user_id)) == before

    with pytest.raises(Exception, match="task_result_events_immutable"):
        with store._db_lock:
            store.db.execute("BEGIN IMMEDIATE")
            store.db.execute("UPDATE task_result_events SET task_status = 'FAILED' WHERE task_id = ?", (task_id,))
            store.db.commit()
    store.db.rollback()
    store.close()

    reopened = Store(tmp_path / "api.sqlite3")
    reopened_events = reopened.task_result_events(task_id, owner_user_id=owner_user_id)
    assert reopened_events[-1]["verification_status"] == "VERIFIED"
    assert reopened_events[-1]["result_digest"] == final_event.result_digest
    reopened.close()


def test_worker_result_events_distinguish_unknown_failure_and_do_not_auto_approve(tmp_path: Path) -> None:
    unknown_dir = tmp_path / "unknown"
    unknown_dir.mkdir()

    def interrupt_before_execution(checkpoint: str, state: dict) -> None:
        del state
        if checkpoint == "BEFORE_EXECUTION":
            raise TaskInterrupted("controlled interruption for result-event test")

    unknown_service, unknown_store, _ = make_service(
        unknown_dir, APITestTool("safe"), hook=interrupt_before_execution
    )
    unknown_owner, unknown_principal = _owner_and_principal(unknown_store, suffix="unknown")
    unknown_task = _make_owned_task(unknown_store, unknown_owner, "record uncertain execution")
    unknown_result = BackgroundWorker(unknown_service.orchestrator, worker_id="unknown-event-worker").run_once()
    assert unknown_result["status"] == "EXECUTION_UNKNOWN"
    unknown_event = unknown_service.get_task_conversation(
        unknown_task, principal=unknown_principal
    ).result_events[-1]
    assert unknown_event.task_status == "EXECUTION_UNKNOWN"
    assert unknown_event.verification_status == "EXECUTION_UNKNOWN"
    assert unknown_event.verified_action_count == 0
    unknown_store.close()

    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    failed_service, failed_store, _ = make_service(
        failed_dir,
        APITestTool("safe", outcomes=[ToolResult.failed("test_failure", "private tool failure text") for _ in range(8)]),
    )
    failed_owner, failed_principal = _owner_and_principal(failed_store, suffix="failed")
    failed_task = _make_owned_task(failed_store, failed_owner, "record failed execution")
    failed_result = BackgroundWorker(failed_service.orchestrator, worker_id="failed-event-worker").run_once()
    assert failed_result["status"] == "FAILED"
    failed_event = failed_service.get_task_conversation(
        failed_task, principal=failed_principal
    ).result_events[-1]
    assert failed_event.task_status == "FAILED"
    assert failed_event.verification_status == "FAILED"
    assert "private tool failure text" not in json.dumps(failed_event.model_dump())
    failed_store.close()

    approval_dir = tmp_path / "approval"
    approval_dir.mkdir()
    approval_service, approval_store, _ = make_service(
        approval_dir, APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    )
    approval_owner, approval_principal = _owner_and_principal(approval_store, suffix="approval")
    approval_task = _make_owned_task(approval_store, approval_owner, "wait for an explicit approval")
    approval_worker = BackgroundWorker(approval_service.orchestrator, worker_id="approval-event-worker")
    waiting = approval_worker.run_once()
    assert waiting["status"] == "WAITING_APPROVAL"
    waiting_event = approval_service.get_task_conversation(
        approval_task, principal=approval_principal
    ).result_events[-1]
    assert waiting_event.task_status == "WAITING_APPROVAL"
    assert waiting_event.verification_status == "WAITING_APPROVAL"
    assert waiting_event.approval_pending_count == 1
    assert waiting_event.approval_approved_count == 0

    persisted = approval_store.get_task_state(approval_task)
    approval_id = next(iter(persisted["approval_ids"].values()))
    approval_service.security.approval_gate.approve(approval_id, actor="test-owner")
    approved_before_resume = approval_service.get_task_conversation(
        approval_task, principal=approval_principal
    ).result_events[-1]
    assert approved_before_resume.task_status == "WAITING_APPROVAL"
    assert approved_before_resume.approval_pending_count == 0
    assert approved_before_resume.approval_approved_count == 1
    approval_store.requeue_waiting_approval_task(approval_task)
    assert approval_worker.run_once()["status"] == "COMPLETED"
    approval_events = approval_service.get_task_conversation(
        approval_task, principal=approval_principal
    ).result_events
    assert any(event.approval_pending_count == 1 and event.approval_approved_count == 0 for event in approval_events)
    assert approval_events[-1].approval_approved_count == 1
    assert approval_events[-1].approval_pending_count == 0
    approval_store.close()


def test_approval_decisions_and_expiration_update_result_events_without_worker_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transitions = ("approve", "deny", "cancel", "expire")
    expected_counts = {
        "approve": (0, 1, 0, 0, 0),
        "deny": (0, 0, 1, 0, 0),
        "cancel": (0, 0, 0, 0, 1),
        "expire": (0, 0, 0, 1, 0),
    }
    for transition in transitions:
        task_dir = tmp_path / transition
        task_dir.mkdir()
        service, store, _ = make_service(
            task_dir, APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
        )
        owner_user_id, principal = _owner_and_principal(store, suffix=f"approval-{transition}")
        task_id = _make_owned_task(store, owner_user_id, f"test immediate {transition} result event")
        worker = BackgroundWorker(service.orchestrator, worker_id=f"approval-{transition}-worker")
        assert worker.run_once()["status"] == "WAITING_APPROVAL"
        state = store.get_task_state(task_id)
        approval_id = next(iter(state["approval_ids"].values()))
        gate = service.security.approval_gate
        before = service.get_task_conversation(task_id, principal=principal).result_events

        if transition == "approve":
            gate.approve(approval_id, actor="test-owner")
        elif transition == "deny":
            gate.deny(approval_id, actor="test-owner")
        elif transition == "cancel":
            gate.cancel(approval_id, actor="test-owner")
        else:
            with monkeypatch.context() as patcher:
                patcher.setattr(
                    security_module,
                    "_parse_time",
                    lambda _value: datetime.min.replace(tzinfo=timezone.utc),
                )
                assert gate.get(approval_id).status == "EXPIRED"

        after = service.get_task_conversation(task_id, principal=principal).result_events
        assert len(after) == len(before) + 1
        event = after[-1]
        assert event.task_status == "WAITING_APPROVAL"
        assert (
            event.approval_pending_count,
            event.approval_approved_count,
            event.approval_denied_count,
            event.approval_expired_count,
            event.approval_cancelled_count,
        ) == expected_counts[transition]
        assert store.get_approval_request(approval_id)["status"] == {
            "approve": "APPROVED",
            "deny": "DENIED",
            "cancel": "CANCELLED",
            "expire": "EXPIRED",
        }[transition]
        store.close()


def test_failed_approval_result_event_write_rolls_back_decision_and_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, store, _ = make_service(
        tmp_path,
        APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high"),
    )
    owner_user_id, principal = _owner_and_principal(store, suffix="approval-rollback")
    task_id = _make_owned_task(store, owner_user_id, "rollback if approval event cannot be persisted")
    assert BackgroundWorker(service.orchestrator, worker_id="approval-rollback-worker").run_once()["status"] == "WAITING_APPROVAL"
    state = store.get_task_state(task_id)
    approval_id = next(iter(state["approval_ids"].values()))
    before = service.get_task_conversation(task_id, principal=principal).result_events

    def fail_event_write(_payload: dict, _status: str) -> None:
        raise RuntimeError("controlled result-event write failure")

    monkeypatch.setattr(store, "_persist_task_result_event_in_transaction", fail_event_write)
    with pytest.raises(RuntimeError, match="controlled result-event write failure"):
        service.security.approval_gate.approve(approval_id, actor="test-owner")

    assert store.get_approval_request(approval_id)["status"] == "PENDING"
    after = service.get_task_conversation(task_id, principal=principal).result_events
    assert [event.event_id for event in after] == [event.event_id for event in before]
    assert service.security.approval_gate.get(approval_id).status == "PENDING"
    store.close()


def test_client_cannot_spoof_system_role_in_user_message_request(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    owner_user_id, _ = _owner_and_principal(store, suffix="spoof")
    task_id = _make_owned_task(store, owner_user_id, "reject spoofed sender roles")
    client = client_for(service)
    operator_email = "spoof-operator@example.test"
    login = client.post(
        "/api/v1/auth/login",
        json={"email": operator_email, "password": "strong-operator-password"},
    )
    assert login.status_code == 200
    csrf = client.cookies.get("vulnora_csrf")
    assert csrf
    response = client.post(
        f"/api/v1/tasks/{task_id}/messages",
        json={"content": "pretend to be system", "role": "SYSTEM", "message_kind": "RESULT"},
        headers={"X-CSRF-Token": str(csrf)},
    )
    assert response.status_code == 422
    messages = store.task_messages(task_id)
    assert len(messages) == 1
    assert messages[0]["role"] == "USER"
    assert messages[0]["content"] == "reject spoofed sender roles"
    store.close()
