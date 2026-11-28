from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
from threading import Barrier

import pytest

from private_agent import security as security_module
from private_agent import storage as storage_module
from private_agent.auth import AuthManager, AuthPrincipal, ROLE_OPERATOR
from private_agent.core.llm import LLMResponse
from private_agent.core.orchestrator import TaskInterrupted
from private_agent.core.planner import Planner
from private_agent.storage import Store, UnsupportedSchemaVersion
from private_agent.tools.contracts import ToolResult
from private_agent.worker import BackgroundWorker
from tests.test_phase12_api import APITestTool, client_for, make_service
from private_agent.api import APIError
from private_agent.core.assistant_reply import render_reply_for_event


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


def test_schema_17_migrates_existing_result_events_to_unknown_source_and_version_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "schema-17.sqlite3"
    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 17)
    legacy = Store(path)
    auth = AuthManager(legacy, owner_email="legacy-owner@example.test")
    owner_id = auth.initialize_owner("legacy-owner@example.test", "strong-owner-password").user_id
    task_id = str(legacy.create_queued_task("legacy result", owner_user_id=None)["task_id"])
    stamp = datetime.now(timezone.utc).isoformat()
    legacy.db.execute("BEGIN IMMEDIATE")
    legacy.db.execute(
        "INSERT INTO task_owners (task_id, user_id, created_at) VALUES (?, ?, ?)",
        (task_id, owner_id, stamp),
    )
    sequence = legacy._allocate_task_conversation_sequence_in_transaction(task_id)
    legacy.db.execute(
        """INSERT INTO task_result_events
        (event_id, task_id, owner_user_id, sequence, role, message_kind, task_status,
         verification_status, result_available, verified_action_count, failed_action_count,
         blocked_action_count, pending_action_count, approval_pending_count, approval_approved_count,
         approval_denied_count, approval_expired_count, approval_cancelled_count, result_digest,
         transition_fingerprint, created_at)
        VALUES (?, ?, ?, ?, 'SYSTEM', 'RESULT', 'QUEUED', 'IN_PROGRESS', 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, ?, ?, ?)""",
        ("legacy-result-event", task_id, owner_id, sequence, "a" * 64, "b" * 64, stamp),
    )
    legacy.db.commit()
    legacy.close()

    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 18)
    migrated = Store(path)
    event = migrated.task_result_events(task_id, owner_user_id=owner_id)[0]
    assert event["planning_mode"] == "UNKNOWN"
    assert event["plan_version"] == 0
    migrated.close()


def test_worker_persists_provider_validated_plan_mode_in_state_result_and_conversation_event(
    tmp_path: Path,
) -> None:
    goal = "complete a verified safe task"

    class SyntheticProvider:
        def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict):
            del system_prompt, user_prompt, response_schema
            return LLMResponse(
                {
                    "goal": goal,
                    "rationale": "Synthetic provider proposal for provenance coverage",
                    "steps": [{
                        "id": "api-step-1",
                        "objective": "complete the controlled task",
                        "tool": "safe",
                        "input": {"value": "safe"},
                        "reason": "Synthetic test plan",
                        "depends_on": [],
                        "success_criteria": {"required_fields": ["result"]},
                    }],
                },
                "{}",
                "synthetic",
            )

    service, store, _ = make_service(
        tmp_path,
        APITestTool("safe"),
        planner=Planner(SyntheticProvider()),
    )
    owner_user_id, principal = _owner_and_principal(store, suffix="provider-plan")
    task_id = _make_owned_task(store, owner_user_id, goal)

    result = BackgroundWorker(service.orchestrator, worker_id="provider-plan-worker").run_once()

    assert result["status"] == "COMPLETED", result
    state = store.get_task_state(task_id)
    assert state["plan"]["planning_mode"] == "PROVIDER_VALIDATED"
    assert state["plan_version"] == 1
    legacy_plan_row = store.db.execute("SELECT plan_json FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert json.loads(legacy_plan_row["plan_json"])["planning_mode"] == "PROVIDER_VALIDATED"
    assert state["result"]["planning_mode"] == "PROVIDER_VALIDATED"
    assert state["result"]["plan_version"] == 1
    assert state["result"]["step_results"][0]["planning_mode"] == "PROVIDER_VALIDATED"
    assert state["result"]["step_results"][0]["plan_version"] == 1
    assert state["result"]["step_outcomes"][0]["observation"]["planning_mode"] == "PROVIDER_VALIDATED"
    assert state["result"]["step_outcomes"][0]["observation"]["plan_version"] == 1
    event = service.get_task_conversation(task_id, principal=principal).result_events[-1]
    assert event.planning_mode == "PROVIDER_VALIDATED"
    assert event.plan_version == 1
    task = service.get_task(task_id, principal=principal)
    assert task.planning_mode == "PROVIDER_VALIDATED"
    assert task.plan_version == 1
    assert len(task.actions) == 1
    assert task.actions[0].planning_mode == "PROVIDER_VALIDATED"
    assert task.actions[0].plan_version == 1
    store.close()


def test_worker_persists_deterministic_fallback_mode_on_step_result_and_action(tmp_path: Path) -> None:
    class PlanningInputTool(APITestTool):
        def planning_inputs(self, goal: str) -> dict[str, str]:
            del goal
            return {"value": "safe"}

    goal = "complete a harmless action without a planning provider"
    service, store, _ = make_service(tmp_path, PlanningInputTool("safe"), planner=Planner())
    owner_user_id, principal = _owner_and_principal(store, suffix="fallback-plan")
    task_id = _make_owned_task(store, owner_user_id, goal)

    result = BackgroundWorker(service.orchestrator, worker_id="fallback-plan-worker").run_once()

    assert result["status"] == "COMPLETED", result
    state = store.get_task_state(task_id)
    assert state["plan"]["planning_mode"] == "DETERMINISTIC_FALLBACK"
    legacy_plan_row = store.db.execute("SELECT plan_json FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert json.loads(legacy_plan_row["plan_json"])["planning_mode"] == "DETERMINISTIC_FALLBACK"
    assert state["result"]["step_results"][0]["planning_mode"] == "DETERMINISTIC_FALLBACK"
    assert state["result"]["step_results"][0]["plan_version"] == 1
    task = service.get_task(task_id, principal=principal)
    assert task.planning_mode == "DETERMINISTIC_FALLBACK"
    assert task.actions[0].planning_mode == "DETERMINISTIC_FALLBACK"
    assert task.actions[0].plan_version == 1
    store.close()


def test_provider_error_before_plan_persistence_remains_unknown_not_provider_validated(tmp_path: Path) -> None:
    class FailingProvider:
        def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict):
            del system_prompt, user_prompt, response_schema
            raise RuntimeError("synthetic provider unavailable")

    service, store, _ = make_service(
        tmp_path,
        APITestTool("safe"),
        planner=Planner(FailingProvider()),
    )
    owner_user_id, principal = _owner_and_principal(store, suffix="provider-error")
    task_id = _make_owned_task(store, owner_user_id, "complete a harmless action")

    result = BackgroundWorker(service.orchestrator, worker_id="provider-error-worker").run_once()

    assert result["status"] == "FAILED", result
    task = service.get_task(task_id, principal=principal)
    assert task.status == "FAILED"
    assert task.planning_mode == "UNKNOWN"
    assert task.plan_version == 0
    event = service.get_task_conversation(task_id, principal=principal).result_events[-1]
    assert event.task_status == "FAILED"
    assert event.planning_mode == "UNKNOWN"
    assert event.plan_version == 0
    store.close()


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
    with store._db_lock:
        store.db.execute("BEGIN IMMEDIATE")
        store._allocate_task_conversation_sequence_in_transaction(task_id)
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
        APITestTool("safe", outcomes=[ToolResult.failed("test_failure", "private tool failure text", no_side_effects=True) for _ in range(8)]),
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



def _downgrade_conversation_db_to_legacy_v16(path: Path, task_id: str) -> None:
    db = sqlite3.connect(path)
    try:
        for trigger in (
            "task_messages_sequence_counter_guard", "task_messages_sequence_cross_type_guard",
            "task_messages_sequence_immutable", "task_result_events_sequence_counter_guard",
            "task_result_events_sequence_cross_type_guard", "task_conversation_sequences_monotonic_guard",
            "task_conversation_sequences_no_delete", "task_result_events_no_update",
            "task_assistant_replies_no_update",
        ):
            db.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        db.execute("DROP TABLE task_conversation_sequences")
        db.execute("DELETE FROM task_state_schema WHERE schema_version = 17")
        db.execute("UPDATE task_messages SET sequence = sequence + 10000 WHERE task_id = ?", (task_id,))
        db.execute("UPDATE task_result_events SET sequence = sequence + 10000 WHERE task_id = ?", (task_id,))
        messages = db.execute(
            "SELECT message_id FROM task_messages WHERE task_id = ? ORDER BY sequence", (task_id,)
        ).fetchall()
        events = db.execute(
            "SELECT event_id FROM task_result_events WHERE task_id = ? ORDER BY sequence", (task_id,)
        ).fetchall()
        for sequence, row in enumerate(messages, start=1):
            db.execute(
                "UPDATE task_messages SET sequence = ?, created_at = '2026-01-01T00:00:00Z' WHERE message_id = ?",
                (sequence, str(row[0])),
            )
        for sequence, row in enumerate(events, start=1):
            db.execute(
                "UPDATE task_result_events SET sequence = ?, created_at = '2026-01-01T00:00:00Z' WHERE event_id = ?",
                (sequence, str(row[0])),
            )
        db.execute(
            "UPDATE task_assistant_replies SET source_sequence = "
            "(SELECT sequence FROM task_result_events WHERE event_id = task_assistant_replies.event_id) "
            "WHERE task_id = ?", (task_id,),
        )
        db.execute("""CREATE TRIGGER task_result_events_no_update
          BEFORE UPDATE ON task_result_events BEGIN
            SELECT RAISE(ABORT, 'task_result_events_immutable');
          END""")
        db.execute("""CREATE TRIGGER task_assistant_replies_no_update
          BEFORE UPDATE ON task_assistant_replies BEGIN
            SELECT RAISE(ABORT, 'task_assistant_replies_immutable');
          END""")
        db.commit()
    finally:
        db.close()


def test_schema_17_backfills_legacy_history_with_same_timestamps_and_pages_after_reopen(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    owner_user_id, _ = _owner_and_principal(store, suffix="migration")
    task_id = _make_owned_task(store, owner_user_id, "migrate a task with saved conversation history")
    store.append_task_message(
        task_id, "second user message", owner_user_id=owner_user_id,
        idempotency_key="migration-message", request_fingerprint="migration-fingerprint",
    )
    assert BackgroundWorker(service.orchestrator, worker_id="migration-worker").run_once()["status"] == "COMPLETED"
    claim = store.claim_next_task_assistant_reply_event("migration-reply-worker", task_id=task_id)
    assert claim is not None
    reply_content = render_reply_for_event(claim, "VERIFIED_A")
    store.save_task_assistant_reply(
        str(claim["event_id"]), worker_id="migration-reply-worker", reply_code="VERIFIED_A",
        content=reply_content, generation_mode="DETERMINISTIC_FALLBACK",
    )
    path = store.path
    old_message_ids = [row["message_id"] for row in store.task_messages(task_id)]
    old_event_ids = [row["event_id"] for row in store.task_result_events(task_id)]
    assert len(old_message_ids) == 2 and len(old_event_ids) >= 2
    store.close()

    _downgrade_conversation_db_to_legacy_v16(path, task_id)
    reopened = Store(path)
    page = reopened.task_conversation_items(task_id, owner_user_id=owner_user_id, limit=2)
    assert page["has_more"] is True
    assert page["next_cursor"] == page["items"][-1]["sequence"] == 2
    all_items = list(page["items"])
    cursor = int(page["next_cursor"])
    while page["has_more"]:
        page = reopened.task_conversation_items(
            task_id, owner_user_id=owner_user_id, after_sequence=cursor, limit=2,
        )
        all_items.extend(page["items"])
        if page["has_more"]:
            assert page["next_cursor"] == page["items"][-1]["sequence"]
            cursor = int(page["next_cursor"])
    sequences = [int(item["sequence"]) for item in all_items]
    assert sequences == list(range(1, len(all_items) + 1))
    assert [item["item_type"] for item in all_items[:4]] == [
        "USER_MESSAGE", "USER_MESSAGE", "SYSTEM_RESULT", "SYSTEM_RESULT",
    ]
    assert [item["message"]["message_id"] for item in all_items if item["item_type"] == "USER_MESSAGE"] == old_message_ids
    assert [item["result_event"]["event_id"] for item in all_items if item["item_type"] == "SYSTEM_RESULT"] == old_event_ids
    assert reopened.task_conversation_items(task_id, owner_user_id="different-user")["items"] == []
    events_by_id = {event["event_id"]: event for event in reopened.task_result_events(task_id, owner_user_id=owner_user_id)}
    replies = reopened.task_assistant_replies(task_id, owner_user_id=owner_user_id)
    assert len(replies) == 1
    assert replies[0]["source_sequence"] == events_by_id[replies[0]["event_id"]]["sequence"]
    counter = reopened.db.execute(
        "SELECT last_sequence FROM task_conversation_sequences WHERE task_id = ?", (task_id,)
    ).fetchone()
    assert int(counter["last_sequence"]) == len(all_items)
    reopened.close()

    restarted = Store(path)
    assert restarted.task_conversation_items(task_id)["items"][-1]["sequence"] == len(all_items)
    with pytest.raises(sqlite3.IntegrityError, match="task_conversation_sequence_not_monotonic"):
        restarted.db.execute(
            "UPDATE task_conversation_sequences SET last_sequence = last_sequence - 1 WHERE task_id = ?",
            (task_id,),
        )
    restarted.db.rollback()
    restarted.close()


def test_follow_up_idempotency_reuses_the_original_shared_sequence(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    owner_user_id, principal = _owner_and_principal(store, suffix="retry")
    task_id = _make_owned_task(store, owner_user_id, "accept an idempotent follow-up")
    first = service.append_task_message(task_id, "same submission", idempotency_key="same-key", principal=principal)
    replay = service.append_task_message(task_id, "same submission", idempotency_key="same-key", principal=principal)
    assert first.message_id == replay.message_id
    assert first.sequence == replay.sequence
    assert first.idempotent_replay is False
    assert replay.idempotent_replay is True
    assert len([row for row in store.task_messages(task_id) if row["message_kind"] == "FOLLOW_UP"]) == 1
    with pytest.raises(APIError) as conflict:
        service.append_task_message(task_id, "different text", idempotency_key="same-key", principal=principal)
    assert conflict.value.error_code == "message_idempotency_conflict"
    store.close()


def test_cross_type_writes_from_two_sqlite_connections_allocate_distinct_sequences(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    owner_user_id, _ = _owner_and_principal(store, suffix="concurrent")
    task_id = _make_owned_task(store, owner_user_id, "serialize concurrent conversation writes")
    second = Store(store.path)
    barrier = Barrier(2)

    def append_message() -> dict:
        barrier.wait(timeout=5)
        return store.append_task_message(
            task_id, "concurrent user message", owner_user_id=owner_user_id,
            idempotency_key="concurrent-message", request_fingerprint="concurrent-fingerprint",
        )

    def append_result_event() -> None:
        state = second.get_task_state(task_id)
        barrier.wait(timeout=5)
        with second._db_lock:
            second.db.execute("BEGIN IMMEDIATE")
            try:
                second._persist_task_result_event_in_transaction(state, "PLANNING")
                second.db.commit()
            except Exception:
                second.db.rollback()
                raise

    with ThreadPoolExecutor(max_workers=2) as pool:
        message_future = pool.submit(append_message)
        event_future = pool.submit(append_result_event)
        message = message_future.result(timeout=15)
        event_future.result(timeout=15)
    items = store.task_conversation_items(task_id)["items"]
    sequences = [int(item["sequence"]) for item in items]
    assert sequences == sorted(sequences)
    assert len(sequences) == len(set(sequences))
    assert message["sequence"] in sequences
    assert any(item["item_type"] == "SYSTEM_RESULT" and item["sequence"] != 2 for item in items)
    counter = store.db.execute(
        "SELECT last_sequence FROM task_conversation_sequences WHERE task_id = ?", (task_id,)
    ).fetchone()
    assert int(counter["last_sequence"]) == max(sequences)
    with pytest.raises(sqlite3.IntegrityError, match="task_messages_sequence_not_allocated"):
        store.db.execute(
            "INSERT INTO task_messages (message_id, task_id, sequence, role, message_kind, content, status, "
            "idempotency_key, request_fingerprint, created_at, consumed_at, consumed_checkpoint) "
            "VALUES ('unallocated-message', ?, ?, 'USER', 'FOLLOW_UP', 'forbidden', 'PENDING', NULL, '', 'now', '', '')",
            (task_id, int(counter["last_sequence"]) + 1),
        )
    store.db.rollback()
    second.close()
    store.close()


def test_schema_20_migration_preserves_result_events_and_reply_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 20)
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    owner_user_id, _ = _owner_and_principal(store, suffix="schema21")
    task_id = _make_owned_task(store, owner_user_id, "preserve immutable event references")
    assert BackgroundWorker(service.orchestrator, worker_id="schema20-worker").run_once()["status"] == "COMPLETED"
    claim = store.claim_next_task_assistant_reply_event("schema20-reply-worker", task_id=task_id)
    assert claim is not None
    store.save_task_assistant_reply(
        str(claim["event_id"]), worker_id="schema20-reply-worker", reply_code="VERIFIED_A",
        content=render_reply_for_event(claim, "VERIFIED_A"), generation_mode="DETERMINISTIC_FALLBACK",
    )
    claimed_task_id = _make_owned_task(store, owner_user_id, "preserve an active event claim")
    assert BackgroundWorker(service.orchestrator, worker_id="schema20-claim-worker").run_once()["status"] == "COMPLETED"
    active_claim = store.claim_next_task_assistant_reply_event(
        "schema20-open-claim-worker", task_id=claimed_task_id
    )
    assert active_claim is not None
    path = store.path
    before_events = (
        store.task_result_events(task_id, owner_user_id=owner_user_id)
        + store.task_result_events(claimed_task_id, owner_user_id=owner_user_id)
    )
    before_replies = store.task_assistant_replies(task_id, owner_user_id=owner_user_id)
    before_claims = [dict(row) for row in store.db.execute(
        "SELECT event_id, task_id, owner_user_id FROM task_assistant_reply_claims WHERE task_id = ?",
        (claimed_task_id,),
    ).fetchall()]
    assert before_events and before_replies and before_claims
    store.close()

    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 21)
    migrated = Store(path)
    after_events = (
        migrated.task_result_events(task_id, owner_user_id=owner_user_id)
        + migrated.task_result_events(claimed_task_id, owner_user_id=owner_user_id)
    )
    assert after_events == before_events
    assert migrated.task_assistant_replies(task_id, owner_user_id=owner_user_id) == before_replies
    after_claims = [dict(row) for row in migrated.db.execute(
        "SELECT event_id, task_id, owner_user_id FROM task_assistant_reply_claims WHERE task_id = ?",
        (claimed_task_id,),
    ).fetchall()]
    assert after_claims == before_claims
    assert migrated.db.execute("PRAGMA foreign_key_check").fetchall() == []
    schema_sql = str(migrated.db.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'task_result_events'"
    ).fetchone()["sql"])
    assert "'WAITING_CLARIFICATION'" in schema_sql
    verification_constraint = schema_sql.split("verification_status TEXT NOT NULL CHECK", 1)[1].split(")", 1)[0]
    assert "'WAITING_CLARIFICATION'" not in verification_constraint
    migrated.close()
