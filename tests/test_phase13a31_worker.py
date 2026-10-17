from __future__ import annotations

import threading
from pathlib import Path

import pytest

from private_agent.storage import KnowledgeItem, QueueLeaseLost, Store
from private_agent.worker import BackgroundWorker
from tests.test_phase12_api import APITestTool, make_service


def test_claims_one_queued_task_and_fences_finish(tmp_path: Path) -> None:
    store = Store(tmp_path / "claim.sqlite3")
    created = store.create_queued_task("claim me")

    claim = store.claim_next_task("worker-a", lease_ttl_seconds=60)

    assert claim is not None
    assert claim["kind"] == "CLAIMED"
    assert claim["task_id"] == created["task_id"]
    assert claim["attempt_count"] == 1
    queue = store.task_queue_record(created["task_id"])
    assert queue["queue_status"] == "LEASED"
    assert queue["lease_id"] == claim["lease_id"]

    finished = store.finish_task_claim(claim, "FAILED", error_code="test_failure", error_message="controlled")
    assert finished["queue_status"] == "FAILED"
    assert store.task_queue_record(created["task_id"])["lease_id"] == ""
    store.close()


def test_two_independent_sqlite_connections_cannot_claim_same_task(tmp_path: Path) -> None:
    path = tmp_path / "claim-race.sqlite3"
    seed = Store(path)
    seed.create_queued_task("claim once")
    seed.close()
    first = Store(path)
    second = Store(path)
    barrier = threading.Barrier(2)

    def claim(store: Store, worker_id: str):
        barrier.wait()
        return store.claim_next_task(worker_id, lease_ttl_seconds=60)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda args: claim(*args), ((first, "worker-a"), (second, "worker-b"))))
    claimed = [result for result in results if result is not None]
    assert len(claimed) == 1
    queue = first.task_queue_record(claimed[0]["task_id"])
    assert queue["queue_status"] == "LEASED"
    assert queue["worker_id"] == claimed[0]["worker_id"]
    first.close()
    second.close()


def test_claim_returns_none_when_queue_is_empty(tmp_path: Path) -> None:
    store = Store(tmp_path / "empty-queue.sqlite3")
    assert store.claim_next_task("worker-a") is None
    store.close()


def test_independent_memory_writes_do_not_require_a_worker_lease(tmp_path: Path) -> None:
    store = Store(tmp_path / "independent-memory.sqlite3")

    store.add_knowledge(
        KnowledgeItem(
            id="independent-knowledge",
            domain="test",
            concept="independent",
            knowledge_type="fact",
            content="safe to persist without a queue task",
            source="test",
            evidence="test",
        )
    )
    store.save_reflection({"insight_id": "independent-reflection", "pattern": "safe", "confidence": 0.9})
    store.save_learned_strategy({"strategy_id": "independent-strategy", "status": "validated", "confidence": 0.9})

    assert store.search_knowledge("independent")
    assert store.get_reflection("independent-reflection") is not None
    assert store.get_learned_strategy("independent-strategy") is not None
    store.close()


def test_run_once_starts_queued_task_without_resume_and_persists_completion(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    created = store.create_queued_task("execute queued task")
    worker = BackgroundWorker(service.orchestrator, worker_id="worker-a")
    service.orchestrator.resume_task = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resume_task must not run for QUEUED"))

    result = worker.run_once()

    assert result["status"] == "COMPLETED"
    assert result["processed"] is True
    assert result["task_id"] == created["task_id"]
    assert service.orchestrator.tools._tools["safe"].calls == 1
    assert store.get_task_state(created["task_id"])["status"] == "COMPLETED"
    assert store.task_queue_record(created["task_id"])["queue_status"] == "COMPLETED"
    store.close()


def test_pending_and_denied_approval_never_execute_and_remain_not_claimable(tmp_path: Path) -> None:
    tool = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, tool)
    created = store.create_queued_task("approval required task")
    worker = BackgroundWorker(service.orchestrator, worker_id="worker-a")

    first = worker.run_once()

    assert first["status"] == "WAITING_APPROVAL"
    assert tool.calls == 0
    state = store.get_task_state(created["task_id"])
    assert state["status"] == "WAITING_APPROVAL"
    approval_id = next(iter(state["approval_ids"].values()))
    service.orchestrator.security.approval_gate.deny(approval_id, actor="owner")
    assert store.task_queue_record(created["task_id"])["queue_status"] == "WAITING_APPROVAL"
    with pytest.raises(ValueError, match="approval_not_approved"):
        store.requeue_waiting_approval_task(created["task_id"])
    assert worker.run_once()["status"] == "NO_TASK"
    assert tool.calls == 0
    store.close()


def test_approved_waiting_task_is_explicitly_requeued_then_resumed_safely(tmp_path: Path) -> None:
    tool = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, tool)
    created = store.create_queued_task("approval then resume")
    worker = BackgroundWorker(service.orchestrator, worker_id="worker-a")

    waiting = worker.run_once()

    assert waiting["status"] == "WAITING_APPROVAL"
    state = store.get_task_state(created["task_id"])
    approval_id = next(iter(state["approval_ids"].values()))
    service.orchestrator.security.approval_gate.approve(approval_id, actor="owner")
    store.requeue_waiting_approval_task(created["task_id"])

    resumed = worker.run_once()

    assert resumed["status"] == "COMPLETED"
    assert tool.calls == 1
    assert store.get_task_state(created["task_id"])["status"] == "COMPLETED"
    store.close()


def test_expired_lease_becomes_execution_unknown_and_is_not_retried(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    created = store.create_queued_task("uncertain queued task")
    claim = store.claim_next_task("worker-old", lease_ttl_seconds=60)
    store.db.execute(
        "UPDATE task_queue SET lease_expires_at = ? WHERE task_id = ?",
        ("1970-01-01T00:00:00+00:00", created["task_id"]),
    )
    store.db.commit()

    worker = BackgroundWorker(service.orchestrator, worker_id="worker-new")
    recovered = worker.run_once()

    assert recovered["status"] == "EXECUTION_UNKNOWN"
    assert recovered["task_id"] == created["task_id"]
    assert service.orchestrator.tools._tools["safe"].calls == 0
    assert store.get_task_state(created["task_id"])["status"] == "EXECUTION_UNKNOWN"
    assert store.task_queue_record(created["task_id"])["queue_status"] == "EXECUTION_UNKNOWN"
    assert worker.run_once()["status"] == "NO_TASK"
    store.close()


def test_stale_worker_cannot_finalize_after_lease_loss(tmp_path: Path) -> None:
    store = Store(tmp_path / "fencing.sqlite3")
    created = store.create_queued_task("fenced task")
    claim = store.claim_next_task("worker-old", lease_ttl_seconds=60)
    store.db.execute(
        "UPDATE task_queue SET lease_expires_at = ? WHERE task_id = ?",
        ("1970-01-01T00:00:00+00:00", created["task_id"]),
    )
    store.db.commit()

    with pytest.raises(QueueLeaseLost, match="expired"):
        store.finish_task_claim(claim, "COMPLETED")
    store.close()


def test_failure_before_execution_is_failed_without_tool_side_effect(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))

    class FailingPlanner:
        provider = None
        max_steps = 1

        def create(self, goal, tools, prior_knowledge):
            raise RuntimeError("planner_failed_before_execution")

    service.orchestrator.planner = FailingPlanner()
    created = store.create_queued_task("planner failure")
    result = BackgroundWorker(service.orchestrator, worker_id="worker-a").run_once()

    assert result["status"] == "FAILED"
    assert result["error_code"] == "worker_failed"
    assert service.orchestrator.tools._tools["safe"].calls == 0
    assert store.get_task_state(created["task_id"])["status"] == "FAILED"
    assert store.task_queue_record(created["task_id"])["queue_status"] == "FAILED"
    store.close()


def test_interruption_after_execution_is_unknown_and_not_automatically_reexecuted(tmp_path: Path) -> None:
    tool = APITestTool("safe")

    def interrupt_after_execution(checkpoint_type, state):
        if checkpoint_type == "AFTER_EXECUTION":
            raise RuntimeError("worker_interrupted_after_external_boundary")

    service, store, _ = make_service(tmp_path, tool, hook=interrupt_after_execution)
    created = store.create_queued_task("interrupted task")
    worker = BackgroundWorker(service.orchestrator, worker_id="worker-a")

    result = worker.run_once()

    assert result["status"] == "EXECUTION_UNKNOWN"
    assert tool.calls == 1
    assert store.get_task_state(created["task_id"])["status"] == "EXECUTION_UNKNOWN"
    assert store.task_queue_record(created["task_id"])["queue_status"] == "EXECUTION_UNKNOWN"
    assert worker.run_once()["status"] == "NO_TASK"
    assert tool.calls == 1
    store.close()


def _recover_expired_claim(tmp_path: Path, filename: str = "stale-writes.sqlite3"):
    path = tmp_path / filename
    old_store = Store(path)
    created = old_store.create_queued_task("stale worker task")
    claim = old_store.claim_next_task("worker-old", lease_ttl_seconds=60)
    assert claim is not None

    recovery_store = Store(path)
    recovery_store.db.execute(
        "UPDATE task_queue SET lease_expires_at = ? WHERE task_id = ?",
        ("1970-01-01T00:00:00+00:00", created["task_id"]),
    )
    recovery_store.db.commit()
    recovered = recovery_store.claim_next_task("worker-recovery", lease_ttl_seconds=60)
    assert recovered is not None
    assert recovered["kind"] == "EXECUTION_UNKNOWN"
    return old_store, recovery_store, created["task_id"], claim


def test_stale_worker_cannot_write_state_events_observations_or_verifications_after_recovery(tmp_path: Path) -> None:
    old_store, recovery_store, task_id, claim = _recover_expired_claim(tmp_path)
    state = old_store.get_task_state(task_id)
    assert state is not None
    observation = {
        "task_id": task_id,
        "step_id": "step-1",
        "tool_name": "safe",
        "execution_status": "success",
        "observed_output": {"result": "stale"},
        "observed_error": None,
        "metadata": {},
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    verification = {"status": "VERIFIED", "verifier_type": "stale-test"}
    observability = {
        "event_id": "stale-event-1",
        "event_type": "TOOL_COMPLETED",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "task_id": task_id,
        "action_id": "step-1",
        "step_id": "step-1",
        "correlation_id": "stale-correlation",
        "component": "test",
        "severity": "INFO",
        "schema_version": 1,
        "metadata": {},
    }
    audit = {
        "event_id": "stale-audit-1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "event_type": "ACTION_EXECUTED",
        "task_id": task_id,
        "action_id": "step-1",
        "actor": "system",
        "tool_or_skill": "safe",
        "capabilities": [],
        "risk_level": "LOW",
        "policy_version": 1,
        "approval_id": "",
        "decision": "ALLOW",
        "reason": "stale",
    }

    with old_store.lease_context(task_id, str(claim["lease_id"]), str(claim["worker_id"])):
        with pytest.raises(QueueLeaseLost, match="queue_lease"):
            old_store.save_task_state(state, "STALE_CHECKPOINT")
        with pytest.raises(QueueLeaseLost, match="queue_lease"):
            old_store.save_event(task_id, "stale_event", {"source": "old-worker"})
        with pytest.raises(QueueLeaseLost, match="queue_lease"):
            old_store.save_observation(observation)
        with pytest.raises(QueueLeaseLost, match="queue_lease"):
            old_store.save_verification(task_id, "step-1", "safe", verification)
        with pytest.raises(QueueLeaseLost, match="queue_lease"):
            old_store.save_observability_event(observability)
        with pytest.raises(QueueLeaseLost, match="queue_lease"):
            old_store.save_audit_event(audit)
        with pytest.raises(QueueLeaseLost, match="queue_lease"):
            old_store.add_knowledge(
                KnowledgeItem(
                    id="stale-knowledge",
                    domain="test",
                    concept="stale",
                    knowledge_type="fact",
                    content="must not persist",
                    source="test",
                    evidence="test",
                )
            )
        with pytest.raises(QueueLeaseLost, match="queue_lease"):
            old_store.save_reflection(
                {"insight_id": "stale-reflection", "pattern": "stale", "confidence": 0.1}
            )
        with pytest.raises(QueueLeaseLost, match="queue_lease"):
            old_store.save_learned_strategy(
                {"strategy_id": "stale-strategy", "status": "uncertain", "confidence": 0.1}
            )

    assert recovery_store.get_task_state(task_id)["status"] == "EXECUTION_UNKNOWN"
    assert recovery_store.events_for_task(task_id) == []
    assert recovery_store.observations_for_task(task_id) == []
    assert recovery_store.verifications_for_task(task_id) == []
    assert recovery_store.observability_events_for_task(task_id) == []
    assert recovery_store.all_audit_events(task_id=task_id) == []
    old_store.close()
    recovery_store.close()


def test_execution_boundary_rechecks_lease_before_tool_side_effect(tmp_path: Path, monkeypatch) -> None:
    tool = APITestTool("safe")
    service, store, _ = make_service(tmp_path, tool)
    created = store.create_queued_task("lease boundary task")

    def expire_before_boundary(task_id: str) -> None:
        store.db.execute(
            "UPDATE task_queue SET lease_expires_at = ? WHERE task_id = ?",
            ("1970-01-01T00:00:00+00:00", task_id),
        )
        store.db.commit()
        raise QueueLeaseLost(f"queue_lease_expired:{task_id}")

    monkeypatch.setattr(store, "assert_active_lease", expire_before_boundary, raising=False)
    result = BackgroundWorker(
        service.orchestrator,
        worker_id="worker-a",
        lease_ttl_seconds=60,
    ).run_once()

    assert result["status"] != "COMPLETED"
    assert tool.calls == 0
    assert not any(
        event["event_type"] == "ACTION_EXECUTED"
        for event in service.security.audit.list(task_id=created["task_id"])
    )
    store.close()


def test_worker_heartbeat_renews_lease_during_long_execution(tmp_path: Path, monkeypatch) -> None:
    tool = APITestTool("safe")
    entered = threading.Event()
    renewed = threading.Event()
    release = threading.Event()
    original_execute = tool.execute

    def long_execute(inputs, context):
        entered.set()
        assert renewed.wait(timeout=5)
        assert release.wait(timeout=5)
        return original_execute(inputs, context)

    tool.execute = long_execute
    original_renew = Store.renew_task_lease

    def renew_and_signal(store_instance, task_id, lease_id, worker_id, *, lease_ttl_seconds=60):
        result = original_renew(
            store_instance,
            task_id,
            lease_id,
            worker_id,
            lease_ttl_seconds=lease_ttl_seconds,
        )
        renewed.set()
        return result

    monkeypatch.setattr(Store, "renew_task_lease", renew_and_signal)
    service, store, _ = make_service(tmp_path, tool)
    created = store.create_queued_task("long heartbeat task")
    result_holder = {}
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="worker-heartbeat",
        lease_ttl_seconds=1,
        heartbeat_interval_seconds=0.01,
    )

    thread = threading.Thread(target=lambda: result_holder.update(result=worker.run_once()))
    thread.start()
    assert entered.wait(timeout=5)
    assert renewed.wait(timeout=5)
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result_holder["result"]["status"] == "COMPLETED"
    assert store.task_queue_record(created["task_id"])["queue_status"] == "COMPLETED"
    assert tool.calls == 1
    store.close()


def test_worker_stops_accepting_results_after_heartbeat_lease_loss(tmp_path: Path, monkeypatch) -> None:
    tool = APITestTool("safe")
    entered = threading.Event()
    release = threading.Event()
    renewal_failed = threading.Event()
    original_execute = tool.execute

    def long_execute(inputs, context):
        entered.set()
        assert release.wait(timeout=5)
        return original_execute(inputs, context)

    tool.execute = long_execute
    original_renew = Store.renew_task_lease

    def signal_failed_renew(store_instance, task_id, lease_id, worker_id, *, lease_ttl_seconds=60):
        try:
            return original_renew(
                store_instance,
                task_id,
                lease_id,
                worker_id,
                lease_ttl_seconds=lease_ttl_seconds,
            )
        except QueueLeaseLost:
            renewal_failed.set()
            raise

    monkeypatch.setattr(Store, "renew_task_lease", signal_failed_renew)
    service, store, _ = make_service(tmp_path, tool)
    created = store.create_queued_task("heartbeat loss task")
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="worker-old",
        lease_ttl_seconds=1,
        heartbeat_interval_seconds=0.01,
    )
    result_holder = {}
    thread = threading.Thread(target=lambda: result_holder.update(result=worker.run_once()))
    thread.start()
    assert entered.wait(timeout=5)

    recovery_store = Store(tmp_path / "api.sqlite3")
    recovery_store.db.execute(
        "UPDATE task_queue SET lease_expires_at = ? WHERE task_id = ?",
        ("1970-01-01T00:00:00+00:00", created["task_id"]),
    )
    recovery_store.db.commit()
    recovered = recovery_store.claim_next_task("worker-new", lease_ttl_seconds=60)
    assert recovered is not None
    assert recovered["kind"] == "EXECUTION_UNKNOWN"
    assert renewal_failed.wait(timeout=5)
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result_holder["result"]["status"] == "LEASE_LOST"
    assert tool.calls == 1
    assert store.get_task_state(created["task_id"])["status"] == "EXECUTION_UNKNOWN"
    assert store.task_queue_record(created["task_id"])["queue_status"] == "EXECUTION_UNKNOWN"
    recovery_store.close()
    store.close()
