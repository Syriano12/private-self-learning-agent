from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from private_agent import storage as storage_module
from private_agent.api import APIError
from private_agent.auth import AuthManager, AuthPrincipal, ROLE_OPERATOR
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.core.task_state import TaskState, action_fingerprint
from private_agent.storage import QueueLeaseLost, Store
from private_agent.worker import BackgroundWorker
from tests.test_phase12_api import APITestTool, auth_headers, client_for, make_service


class FollowupPlanner:
    provider = None
    max_steps = 12

    def __init__(self) -> None:
        self.calls: list[str] = []

    def create(self, goal, tools, prior_knowledge, **kwargs):
        self.calls.append(goal)
        first = PlanStep(
            id="first-step",
            objective="perform the already verified safe step",
            tool="safe",
            input={"value": "first"},
            reason="initial safe action",
            depends_on=[],
            success_criteria={"required_fields": ["result"]},
        )
        if len(self.calls) == 1:
            return TaskPlan(goal, [first], "initial plan")
        network = PlanStep(
            id="network-step",
            objective="perform the new network action",
            tool="network",
            input={"value": "followup"},
            reason="created only after the follow-up was received",
            depends_on=["first-step"],
            success_criteria={"required_fields": ["result"]},
        )
        return TaskPlan(goal, [first, network], "plan incorporating user follow-up")


class DependentSafeFollowupPlanner:
    provider = None
    max_steps = 12

    def __init__(self) -> None:
        self.calls = 0

    def create(self, goal, tools, prior_knowledge, **kwargs):
        del tools, prior_knowledge, kwargs
        self.calls += 1
        first = PlanStep(
            id="first-step",
            objective="perform the already verified safe step",
            tool="safe",
            input={"value": "first"},
            reason="initial safe action",
            depends_on=[],
            success_criteria={"required_fields": ["result"]},
        )
        if self.calls == 1:
            return TaskPlan(goal, [first], "initial plan")
        followup = PlanStep(
            id="followup-step",
            objective="perform a dependent low-risk follow-up action",
            tool="followup",
            input={"value": "followup"},
            reason="created only after the follow-up was received",
            depends_on=["first-step"],
            success_criteria={"required_fields": ["result"]},
        )
        return TaskPlan(goal, [first, followup], "plan incorporating user follow-up")


def test_schema_12_migrates_to_conversation_schema_14(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "schema-12.sqlite3"
    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 12)
    original = Store(path)
    assert original.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_messages'"
    ).fetchone() is None
    original.close()

    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 14)
    migrated = Store(path)
    assert migrated.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_messages'"
    ).fetchone() is not None
    assert migrated.db.execute(
        "SELECT schema_version FROM task_state_schema ORDER BY schema_version DESC LIMIT 1"
    ).fetchone()["schema_version"] == 14
    migrated.close()


def test_schema_13_conversation_layout_migrates_to_18_without_losing_messages(tmp_path: Path) -> None:
    path = tmp_path / "schema-13-conversation.sqlite3"
    store = Store(path)
    created = store.create_queued_task(
        "legacy conversation task",
        initial_message="existing initial message",
    )
    task_id = created["task_id"]
    assert store.task_messages(task_id)[0]["content"] == "existing initial message"

    # Model the other open branch's schema-13 layout: conversations exist,
    # but the schema-13 knowledge-attestation table has not been installed.
    store.db.execute("DELETE FROM task_state_schema WHERE schema_version = 14")
    store.db.execute("DROP TABLE knowledge_attestations")
    store.db.commit()
    store.close()

    migrated = Store(path)
    assert migrated.task_messages(task_id)[0]["content"] == "existing initial message"
    assert migrated.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_attestations'"
    ).fetchone()
    assert migrated.db.execute(
        "SELECT schema_version FROM task_state_schema ORDER BY schema_version DESC LIMIT 1"
    ).fetchone()["schema_version"] == 18
    migrated.close()


def test_initial_message_owner_queue_and_idempotency_are_atomic(tmp_path: Path) -> None:
    store = Store(tmp_path / "atomic-first-message.sqlite3")
    owner = AuthManager(store).create_user(
        "conversation-owner@example.test",
        "strong-password-for-tests",
        "OPERATOR",
    )
    store.db.execute(
        "CREATE TRIGGER reject_first_message BEFORE INSERT ON task_messages "
        "BEGIN SELECT RAISE(ABORT, 'reject first message'); END"
    )
    store.db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="reject first message"):
        store.create_queued_task(
            "atomic goal",
            initial_message="atomic goal",
            owner_user_id=owner.user_id,
            idempotency_key="atomic-key",
            request_fingerprint="atomic-fingerprint",
        )

    assert store.db.execute("SELECT COUNT(*) AS n FROM task_states").fetchone()["n"] == 0
    assert store.db.execute("SELECT COUNT(*) AS n FROM task_owners").fetchone()["n"] == 0
    assert store.db.execute("SELECT COUNT(*) AS n FROM task_queue").fetchone()["n"] == 0
    assert store.db.execute("SELECT COUNT(*) AS n FROM api_idempotency_keys").fetchone()["n"] == 0
    store.db.execute("DROP TRIGGER reject_first_message")
    store.db.commit()
    created = store.create_queued_task(
        "atomic goal",
        initial_message="atomic goal",
        owner_user_id=owner.user_id,
        idempotency_key="atomic-key",
        request_fingerprint="atomic-fingerprint",
    )
    replay = store.create_queued_task(
        "atomic goal",
        initial_message="atomic goal",
        owner_user_id=owner.user_id,
        idempotency_key="atomic-key",
        request_fingerprint="atomic-fingerprint",
    )
    messages = store.task_messages(created["task_id"], owner_user_id=owner.user_id)
    assert created["replayed"] is False
    assert replay["replayed"] is True
    assert replay["task_id"] == created["task_id"]
    assert len(messages) == 1
    assert messages[0]["content"] == "atomic goal"
    assert messages[0]["message_kind"] == "INITIAL"
    assert messages[0]["status"] == "CONSUMED"
    assert store.task_messages(created["task_id"], owner_user_id="other-user") == []
    store.close()


def test_http_messages_are_pending_idempotent_owner_checked_and_read_only_to_task_state(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    created = client.post(
        "/api/v1/tasks",
        json={"goal": "start a conversation"},
        headers=auth_headers(key="conversation-create"),
    )
    assert created.status_code == 201
    task_id = created.json()["task_id"]
    assert created.json()["conversation_id"] == task_id
    state_before = store.get_task_state(task_id)
    checkpoint_before = store.latest_task_checkpoint(task_id)

    first = client.post(
        f"/api/v1/tasks/{task_id}/messages",
        json={"content": "Please include one more safe step"},
        headers=auth_headers(key="followup-1"),
    )
    replay = client.post(
        f"/api/v1/tasks/{task_id}/messages",
        json={"content": "Please include one more safe step"},
        headers=auth_headers(key="followup-1"),
    )
    conflict = client.post(
        f"/api/v1/tasks/{task_id}/messages",
        json={"content": "different text"},
        headers=auth_headers(key="followup-1"),
    )

    assert first.status_code == 202
    assert first.json()["status"] == "PENDING"
    assert first.json()["role"] == "USER"
    assert replay.status_code == 202 and replay.json()["idempotent_replay"] is True
    assert replay.json()["message_id"] == first.json()["message_id"]
    assert conflict.status_code == 409
    assert conflict.json()["error_code"] == "message_idempotency_conflict"
    state_after = store.get_task_state(task_id)
    assert state_after["state_fingerprint"] == state_before["state_fingerprint"]
    assert store.latest_task_checkpoint(task_id)["sequence"] == checkpoint_before["sequence"]

    conversation = client.get(f"/api/v1/tasks/{task_id}/messages", headers=auth_headers())
    assert conversation.status_code == 200
    body = conversation.json()
    assert body["api_version"] == "v1"
    assert body["schema_version"] == 3
    assert body["conversation_id"] == task_id
    assert [message["message_kind"] for message in body["messages"]] == ["INITIAL", "FOLLOW_UP"]
    assert all(message["role"] == "USER" for message in body["messages"])
    assert body["messages"][1]["status"] == "PENDING"
    assert client.get(f"/api/v1/tasks/{task_id}/messages").status_code == 401
    store.close()


def test_conversation_read_and_append_require_task_owner(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    auth = AuthManager(store, owner_email="owner@example.test")
    owner = auth.initialize_owner("owner@example.test", "strong-owner-password")
    task_owner = auth.create_user(
        "task-owner@example.test",
        "strong-task-owner-password",
        ROLE_OPERATOR,
        actor_user_id=owner.user_id,
    )
    other_user = auth.create_user(
        "different-user@example.test",
        "strong-different-password",
        ROLE_OPERATOR,
        actor_user_id=owner.user_id,
    )
    task = store.create_queued_task(
        "owner-only conversation",
        initial_message="owner-only conversation",
        owner_user_id=task_owner.user_id,
    )
    other_principal = AuthPrincipal(
        subject=other_user.user_id,
        user_id=other_user.user_id,
        email=other_user.email,
        role=ROLE_OPERATOR,
    )
    owner_principal = AuthPrincipal(
        subject=task_owner.user_id,
        user_id=task_owner.user_id,
        email=task_owner.email,
        role=ROLE_OPERATOR,
    )

    with pytest.raises(APIError) as read_error:
        service.get_task_conversation(task["task_id"], principal=other_principal)
    assert read_error.value.status_code == 404
    with pytest.raises(APIError) as append_error:
        service.append_task_message(task["task_id"], "not for this owner", principal=other_principal)
    assert append_error.value.status_code == 404
    assert len(store.task_messages(task["task_id"], owner_user_id=task_owner.user_id)) == 1

    conversation = service.get_task_conversation(task["task_id"], principal=owner_principal)
    appended = service.append_task_message(task["task_id"], "authorized follow-up", principal=owner_principal)
    assert len(conversation.messages) == 1
    assert appended.status == "PENDING"
    store.close()


def test_waiting_approval_keeps_followup_pending_and_execution_unknown_rejects_it(tmp_path: Path) -> None:
    network = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, network)
    created = store.create_queued_task("request a controlled network action", initial_message="request a controlled network action")
    worker = BackgroundWorker(service.orchestrator, worker_id="worker-approval")

    waiting = worker.run_once()
    assert waiting["status"] == "WAITING_APPROVAL"
    state = TaskState.from_dict(store.get_task_state(created["task_id"]))
    before = state.to_dict()
    result = service.append_task_message(created["task_id"], "Also check one additional detail")
    assert result.status == "PENDING"
    after = TaskState.from_dict(store.get_task_state(created["task_id"]))
    assert after.status == "WAITING_APPROVAL"
    assert after.plan_fingerprint == before["plan_fingerprint"]
    assert store.task_messages(created["task_id"])[-1]["status"] == "PENDING"
    assert worker.run_once()["status"] == "NO_TASK"

    unknown = store.create_queued_task("uncertain task")
    claim = store.claim_next_task("worker-unknown")
    assert claim is not None and claim["task_id"] == unknown["task_id"]
    store.finish_task_claim(
        claim,
        "EXECUTION_UNKNOWN",
        state_status="EXECUTION_UNKNOWN",
        error_code="uncertain",
    )
    with pytest.raises(ValueError, match="conversation_execution_unknown"):
        store.append_task_message(unknown["task_id"], "must not rescue an uncertain execution")
    store.close()


def test_followup_is_consumed_after_verified_action_then_replanned_and_reauthorized(tmp_path: Path) -> None:
    safe = APITestTool("safe")
    network = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, safe)
    service.orchestrator.tools.register(network)
    planner = FollowupPlanner()
    service.orchestrator.planner = planner
    created = store.create_queued_task("complete the first safe action", initial_message="complete the first safe action")

    original_execute = safe.execute

    def inject_followup(inputs, context):
        if safe.calls == 0:
            queued = service.append_task_message(created["task_id"], "Then run the network step")
            assert queued.status == "PENDING"
        return original_execute(inputs, context)

    safe.execute = inject_followup
    worker = BackgroundWorker(service.orchestrator, worker_id="worker-conversation")
    waiting = worker.run_once()

    assert waiting["status"] == "WAITING_APPROVAL"
    assert safe.calls == 1
    assert network.calls == 0
    assert len(planner.calls) == 2
    assert "Then run the network step" in planner.calls[1]
    persisted = TaskState.from_dict(store.get_task_state(created["task_id"]))
    assert persisted.plan_version == 2
    first_step = persisted.plan["steps"][0]
    first_key = action_fingerprint(
        plan_version=persisted.plan_version,
        step_id=first_step["id"],
        tool=first_step["tool"],
        inputs=first_step["input"],
    )
    assert first_key in persisted.completed_actions
    assert persisted.action_records[first_key]["status"] == "VERIFIED"
    assert persisted.action_records[first_key]["planning_mode"] == "UNKNOWN"
    assert persisted.action_records[first_key]["plan_version"] == 1
    assert persisted.result["planning_mode"] == "UNKNOWN"
    assert persisted.result["plan_version"] == 2
    assert store.task_messages(created["task_id"])[-1]["status"] == "CONSUMED"
    assert persisted.status == "WAITING_APPROVAL"
    assert persisted.stop_reason is not None and persisted.stop_reason.code.value == "APPROVAL_REQUIRED"
    assert any(item["message_kind"] == "FOLLOW_UP" for item in store.task_messages(created["task_id"]))

    approval_id = next(iter(persisted.approval_ids.values()))
    service.security.approval_gate.approve(approval_id, actor="test-owner")
    store.requeue_waiting_approval_task(created["task_id"])
    completed = worker.run_once()
    assert completed["status"] == "COMPLETED"
    assert network.calls == 1
    assert store.task_queue_record(created["task_id"])["queue_status"] == "COMPLETED"
    final_state = TaskState.from_dict(store.get_task_state(created["task_id"]))
    step_results = final_state.result["step_results"]
    assert {(item["tool_name"], item["planning_mode"], item["plan_version"]) for item in step_results} == {
        ("safe", "UNKNOWN", 1),
        ("network", "UNKNOWN", 2),
    }
    outcome_versions = {
        item["observation"]["plan_version"]
        for item in final_state.result["step_outcomes"]
    }
    assert outcome_versions == {1, 2}
    store.close()


def test_replanned_message_commit_survives_restart_as_unknown_without_reexecution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    safe = APITestTool("safe")
    network = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, safe)
    service.orchestrator.tools.register(network)
    planner = FollowupPlanner()
    service.orchestrator.planner = planner
    created = store.create_queued_task(
        "complete the first safe action",
        initial_message="complete the first safe action",
    )
    followup_message_ids: list[str] = []
    original_execute = safe.execute

    def inject_followup(inputs, context):
        if safe.calls == 0:
            queued = service.append_task_message(created["task_id"], "Then run the network step")
            assert queued.status == "PENDING"
            followup_message_ids.append(queued.message_id)
        return original_execute(inputs, context)

    safe.execute = inject_followup
    original_save_task_state = store.save_task_state

    def crash_after_replanned_commit(
        state,
        checkpoint_type: str,
        *,
        task_status: str | None = None,
        consume_message_ids=(),
    ) -> None:
        original_save_task_state(
            state,
            checkpoint_type,
            task_status=task_status,
            consume_message_ids=consume_message_ids,
        )
        if checkpoint_type == "CONVERSATION_REPLANNED":
            raise SimulatedProcessCrash

    monkeypatch.setattr(store, "save_task_state", crash_after_replanned_commit)
    worker = BackgroundWorker(service.orchestrator, worker_id="worker-crash-at-replan")

    with pytest.raises(SimulatedProcessCrash):
        worker.run_once()

    assert safe.calls == 1
    assert network.calls == 0
    assert len(planner.calls) == 2
    assert len(followup_message_ids) == 1
    store.close()

    restarted_service, reopened_store, _ = make_service(tmp_path, safe)
    restarted_service.orchestrator.tools.register(network)
    restarted_planner = FollowupPlanner()
    restarted_service.orchestrator.planner = restarted_planner
    reopened_store.db.execute(
        "UPDATE task_queue SET lease_expires_at = ? WHERE task_id = ?",
        ("1970-01-01T00:00:00+00:00", created["task_id"]),
    )
    reopened_store.db.commit()

    restarted_worker = BackgroundWorker(
        restarted_service.orchestrator,
        worker_id="worker-after-restart",
    )
    recovered = restarted_worker.run_once()

    assert recovered["status"] == "EXECUTION_UNKNOWN"
    assert recovered["task_id"] == created["task_id"]
    assert recovered["processed"] is False
    persisted_payload = reopened_store.get_task_state(created["task_id"])
    assert persisted_payload is not None
    persisted = TaskState.from_dict(persisted_payload)
    messages = reopened_store.task_messages(created["task_id"])
    consumed_followups = [
        item for item in messages if item["message_id"] in followup_message_ids
    ]
    assert len(consumed_followups) == 1
    assert consumed_followups[0]["status"] == "CONSUMED"
    assert consumed_followups[0]["consumed_checkpoint"] == "CONVERSATION_REPLANNED"
    assert persisted.plan_version == 2
    replans = persisted.context["conversation_replans"]
    assert len(replans) == 1
    assert replans[0]["message_ids"] == followup_message_ids
    replanned_checkpoints = [
        item for item in reopened_store.task_checkpoints(created["task_id"])
        if item["checkpoint_type"] == "CONVERSATION_REPLANNED"
    ]
    assert len(replanned_checkpoints) == 1
    assert reopened_store.latest_task_checkpoint(created["task_id"])["checkpoint_type"] == "EXECUTION_UNKNOWN"
    assert persisted.status == "EXECUTION_UNKNOWN"
    assert persisted.stop_reason is not None
    assert persisted.stop_reason.code.value == "EXECUTION_UNKNOWN"
    assert reopened_store.task_queue_record(created["task_id"])["queue_status"] == "EXECUTION_UNKNOWN"

    first_step = persisted.plan["steps"][0]
    first_key = action_fingerprint(
        plan_version=persisted.plan_version,
        step_id=first_step["id"],
        tool=first_step["tool"],
        inputs=first_step["input"],
    )
    assert first_key in persisted.completed_actions
    assert persisted.action_records[first_key]["status"] == "VERIFIED"
    assert safe.calls == 1
    assert network.calls == 0
    assert restarted_planner.calls == []
    assert restarted_worker.run_once()["status"] == "NO_TASK"
    assert safe.calls == 1
    assert network.calls == 0
    assert len(restarted_planner.calls) == 0
    reopened_store.close()


def test_unchanged_followup_plan_preserves_verified_outcome(tmp_path: Path) -> None:
    class UnchangedPlanner:
        provider = None
        max_steps = 12

        def __init__(self) -> None:
            self.calls: list[str] = []

        def create(self, goal, tools, prior_knowledge, **kwargs):
            del tools, prior_knowledge, kwargs
            self.calls.append(goal)
            step = PlanStep(
                id="first-step",
                objective="perform the already verified safe step",
                tool="safe",
                input={"value": "first"},
                reason="initial safe action",
                depends_on=[],
                success_criteria={"required_fields": ["result"]},
            )
            return TaskPlan(goal, [step], "same plan after follow-up")

    safe = APITestTool("safe")
    service, store, _ = make_service(tmp_path, safe)
    planner = UnchangedPlanner()
    service.orchestrator.planner = planner
    created = store.create_queued_task("complete the first safe action", initial_message="complete the first safe action")

    original_execute = safe.execute

    def inject_followup(inputs, context):
        if safe.calls == 0:
            queued = service.append_task_message(created["task_id"], "The verified first step is sufficient; add no action")
            assert queued.status == "PENDING"
        return original_execute(inputs, context)

    safe.execute = inject_followup
    completed = BackgroundWorker(service.orchestrator, worker_id="worker-unchanged-followup").run_once()

    assert completed["status"] == "COMPLETED", completed
    assert safe.calls == 1
    assert len(planner.calls) == 2
    assert "The verified first step is sufficient; add no action" in planner.calls[1]
    result = completed["result"]
    assert result["verification"]["verified"] is True
    assert [item["verification_status"] for item in result["final_step_outcomes"]] == ["VERIFIED"]
    persisted = TaskState.from_dict(store.get_task_state(created["task_id"]))
    assert persisted.plan_version == 2
    assert persisted.status == "COMPLETED"
    assert store.task_messages(created["task_id"])[-1]["status"] == "CONSUMED"
    store.close()


def test_low_risk_followup_dependency_uses_verified_result_without_resume(tmp_path: Path) -> None:
    safe = APITestTool("safe")
    followup = APITestTool("followup")
    service, store, _ = make_service(tmp_path, safe)
    service.orchestrator.tools.register(followup)
    planner = DependentSafeFollowupPlanner()
    service.orchestrator.planner = planner
    created = store.create_queued_task(
        "complete the first safe action",
        initial_message="complete the first safe action",
    )

    original_execute = safe.execute

    def inject_followup(inputs, context):
        if safe.calls == 0:
            queued = service.append_task_message(created["task_id"], "Then perform one dependent safe action")
            assert queued.status == "PENDING"
        return original_execute(inputs, context)

    safe.execute = inject_followup
    worker = BackgroundWorker(service.orchestrator, worker_id="worker-dependent-conversation")
    completed = worker.run_once()

    assert completed["status"] == "COMPLETED", completed
    assert safe.calls == 1
    assert followup.calls == 1
    assert planner.calls == 2
    persisted = TaskState.from_dict(store.get_task_state(created["task_id"]))
    assert persisted.status == "COMPLETED"
    assert store.task_messages(created["task_id"])[-1]["status"] == "CONSUMED"
    store.close()


def test_terminal_task_cannot_accept_followup_and_worker_reads_require_lease(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    created = store.create_queued_task("finish and then reject messages")
    with pytest.raises(QueueLeaseLost, match="conversation_worker_lease_required"):
        store.pending_task_messages_for_worker(created["task_id"])
    completed = BackgroundWorker(service.orchestrator, worker_id="worker-final").run_once()
    assert completed["status"] == "COMPLETED"
    with pytest.raises(ValueError, match="conversation_terminal"):
        store.append_task_message(completed["task_id"], "too late")
    store.close()


def test_worker_cannot_read_pending_followups_before_verified_checkpoint(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    created = store.create_queued_task("must reach a verified checkpoint")
    message = service.append_task_message(created["task_id"], "wait for verified work")
    claim = store.claim_next_task("worker-before-checkpoint")
    assert claim is not None and claim["task_id"] == created["task_id"]

    with store.lease_context(created["task_id"], claim["lease_id"], claim["worker_id"]):
        with pytest.raises(ValueError, match="conversation_safe_checkpoint_required"):
            store.pending_task_messages_for_worker(created["task_id"])
    assert store.task_messages(created["task_id"])[-1]["message_id"] == message.message_id
    assert store.task_messages(created["task_id"])[-1]["status"] == "PENDING"
    store.finish_task_claim(
        claim,
        "FAILED",
        state_status="FAILED",
        error_code="test_cleanup",
        error_message="synthetic test task ended before execution",
    )
    store.close()


def test_pending_message_idempotency_replay_rejects_terminal_task(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    created = store.create_queued_task("terminal replay must not be accepted")
    message = service.append_task_message(
        created["task_id"],
        "follow-up received before terminal failure",
        idempotency_key="terminal-replay-key",
    )
    claim = store.claim_next_task("worker-terminal-replay")
    assert claim is not None and claim["task_id"] == created["task_id"]
    store.finish_task_claim(
        claim,
        "FAILED",
        state_status="FAILED",
        error_code="test_terminal",
        error_message="synthetic terminal state",
    )

    with pytest.raises(APIError) as replay_error:
        service.append_task_message(
            created["task_id"],
            "follow-up received before terminal failure",
            idempotency_key="terminal-replay-key",
        )
    assert replay_error.value.status_code == 409
    assert replay_error.value.error_code == "conversation_terminal"
    messages = store.task_messages(created["task_id"])
    assert len(messages) == 1
    assert messages[0]["message_id"] == message.message_id
    assert messages[0]["status"] == "PENDING"
    store.close()
