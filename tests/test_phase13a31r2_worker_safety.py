from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from private_agent.security import SecurityController
from private_agent.storage import QueueLeaseLost, Store
from private_agent.worker import BackgroundWorker
from tests.test_phase12_api import APITestTool, make_service


def _wait_until(predicate, *, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_worker_security_audit_cannot_write_after_lease_recovery(tmp_path: Path) -> None:
    path = tmp_path / "audit-window.sqlite3"
    store = Store(path)
    tool = APITestTool("safe")
    controller = SecurityController(store=store)
    created = store.create_queued_task("audit window")
    old_claim = store.claim_next_task("worker-old", lease_ttl_seconds=60)
    assert old_claim is not None

    entered_audit_window = threading.Event()
    release_audit_window = threading.Event()
    original_writer = controller.audit._write_security_claim
    calls = {"count": 0}

    def gated_writer(claim):
        calls["count"] += 1
        if calls["count"] == 1:
            entered_audit_window.set()
            assert release_audit_window.wait(timeout=5)
        return original_writer(claim)

    controller.audit._write_security_claim = gated_writer
    action = controller.action_for_tool(created["task_id"], "audit-step", tool, {})
    result: dict[str, object] = {}

    def preflight_in_worker_context() -> None:
        try:
            with store.lease_context(created["task_id"], old_claim["lease_id"], old_claim["worker_id"]):
                result["decision"] = controller.preflight(action)
        except BaseException as exc:  # the test must observe the real boundary error
            result["error"] = exc

    thread = threading.Thread(target=preflight_in_worker_context)
    thread.start()
    assert entered_audit_window.wait(timeout=5)

    recovery_store = Store(path)
    recovery_store.db.execute(
        "UPDATE task_queue SET lease_expires_at = ? WHERE task_id = ?",
        ("1970-01-01T00:00:00+00:00", created["task_id"]),
    )
    recovery_store.db.commit()
    recovered = recovery_store.claim_next_task("worker-new", lease_ttl_seconds=60)
    assert recovered is not None
    assert recovered["kind"] == "EXECUTION_UNKNOWN"

    release_audit_window.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert isinstance(result.get("error"), QueueLeaseLost)
    assert recovery_store.all_audit_events(task_id=created["task_id"]) == []
    assert recovery_store.task_queue_record(created["task_id"])["queue_status"] == "EXECUTION_UNKNOWN"
    recovery_store.close()
    store.close()


def test_stuck_heartbeat_is_reported_and_does_not_block_next_task(tmp_path: Path, monkeypatch) -> None:
    tool = APITestTool("safe")
    tool_entered = threading.Event()
    renewal_entered = threading.Event()
    release_tool = threading.Event()
    release_renewal = threading.Event()
    original_execute = tool.execute
    original_renew = Store.renew_task_lease
    first_task_id: dict[str, str] = {}

    def controlled_execute(inputs, context):
        tool_entered.set()
        assert renewal_entered.wait(timeout=5)
        assert release_tool.wait(timeout=5)
        return original_execute(inputs, context)

    def controlled_renew(store_instance, task_id, lease_id, worker_id, *, lease_ttl_seconds=60):
        if task_id == first_task_id.get("task_id"):
            renewal_entered.set()
            assert release_renewal.wait(timeout=5)
        return original_renew(
            store_instance,
            task_id,
            lease_id,
            worker_id,
            lease_ttl_seconds=lease_ttl_seconds,
        )

    tool.execute = controlled_execute
    monkeypatch.setattr(Store, "renew_task_lease", controlled_renew)
    service, store, _ = make_service(tmp_path, tool)
    created = store.create_queued_task("stuck heartbeat task")
    first_task_id["task_id"] = created["task_id"]
    result_holder: dict[str, object] = {}
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="worker-stuck-heartbeat",
        lease_ttl_seconds=1,
        heartbeat_interval_seconds=0.01,
        max_execution_age_seconds=30,
    )

    run_thread = threading.Thread(target=lambda: result_holder.update(result=worker.run_once()))
    run_thread.start()
    try:
        assert tool_entered.wait(timeout=5)
        assert renewal_entered.wait(timeout=5)
        release_tool.set()
        run_thread.join(timeout=5)
        assert not run_thread.is_alive()
        result = result_holder["result"]
        assert isinstance(result, dict)
        assert result.get("heartbeat_cleanup") == "stuck"
        assert result.get("heartbeat_stuck") is True

        next_created = store.create_queued_task("next task")
        next_result = BackgroundWorker(
            service.orchestrator,
            worker_id="worker-next",
            lease_ttl_seconds=1,
            heartbeat_interval_seconds=0.01,
            max_execution_age_seconds=30,
        ).run_once()
        assert next_result["task_id"] == next_created["task_id"]
        assert next_result["status"] == "COMPLETED"
        assert store.task_queue_record(next_created["task_id"])["queue_status"] == "COMPLETED"
    finally:
        release_tool.set()
        release_renewal.set()
        run_thread.join(timeout=5)
        store.close()


def test_heartbeat_failure_fences_task_before_result_is_accepted(tmp_path: Path, monkeypatch) -> None:
    tool = APITestTool("safe")
    entered = threading.Event()
    renewal_failed = threading.Event()
    release = threading.Event()
    original_execute = tool.execute
    original_renew = Store.renew_task_lease

    def controlled_execute(inputs, context):
        entered.set()
        assert renewal_failed.wait(timeout=5)
        assert release.wait(timeout=5)
        return original_execute(inputs, context)

    def failed_renew(store_instance, task_id, lease_id, worker_id, *, lease_ttl_seconds=60):
        renewal_failed.set()
        raise QueueLeaseLost(f"queue_lease_lost:{task_id}")

    tool.execute = controlled_execute
    monkeypatch.setattr(Store, "renew_task_lease", failed_renew)
    service, store, _ = make_service(tmp_path, tool)
    created = store.create_queued_task("heartbeat failure")
    result_holder: dict[str, object] = {}
    run_thread = threading.Thread(
        target=lambda: result_holder.update(
            result=BackgroundWorker(
                service.orchestrator,
                worker_id="worker-failure",
                lease_ttl_seconds=60,
                heartbeat_interval_seconds=0.01,
                max_execution_age_seconds=30,
            ).run_once()
        )
    )
    run_thread.start()
    try:
        assert entered.wait(timeout=5)
        assert renewal_failed.wait(timeout=5)
        release.set()
        run_thread.join(timeout=5)
        assert not run_thread.is_alive()
        assert result_holder["result"]["status"] == "LEASE_LOST"
        # The worker must not report success after the heartbeat failure.  If
        # Orchestrator already committed a terminal state before that late
        # failure, the durable terminal state is retained rather than being
        # rewritten into a contradictory EXECUTION_UNKNOWN state.
        assert store.get_task_state(created["task_id"])["status"] == "COMPLETED"
        assert store.task_queue_record(created["task_id"])["queue_status"] == "COMPLETED"
        assert tool.calls == 1
    finally:
        release.set()
        run_thread.join(timeout=5)
        monkeypatch.setattr(Store, "renew_task_lease", original_renew)
        store.close()


def test_max_execution_age_stops_renewal_and_marks_unknown(tmp_path: Path) -> None:
    tool = APITestTool("safe")
    entered = threading.Event()
    release = threading.Event()
    original_execute = tool.execute

    def blocking_execute(inputs, context):
        entered.set()
        assert release.wait(timeout=5)
        return original_execute(inputs, context)

    tool.execute = blocking_execute
    service, store, _ = make_service(tmp_path, tool)
    created = store.create_queued_task("hung execution")
    result_holder: dict[str, object] = {}
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="worker-watchdog",
        lease_ttl_seconds=60,
        heartbeat_interval_seconds=0.01,
        # The watchdog measures claim age, including planning and security
        # preflight before the controlled tool call begins. 80ms made this
        # regression timing-dependent under a full-suite load.
        max_execution_age_seconds=1.0,
    )
    run_thread = threading.Thread(target=lambda: result_holder.update(result=worker.run_once()))
    run_thread.start()
    try:
        assert entered.wait(timeout=5)
        assert _wait_until(
            lambda: store.task_queue_record(created["task_id"])["queue_status"] == "EXECUTION_UNKNOWN",
            timeout=2,
        )
        assert store.get_task_state(created["task_id"])["status"] == "EXECUTION_UNKNOWN"
        release.set()
        run_thread.join(timeout=5)
        assert not run_thread.is_alive()
        assert result_holder["result"]["status"] == "LEASE_LOST"
        assert store.task_queue_record(created["task_id"])["queue_status"] == "EXECUTION_UNKNOWN"
        assert BackgroundWorker(service.orchestrator, worker_id="worker-recovery").run_once()["status"] == "NO_TASK"
        assert tool.calls == 1
    finally:
        release.set()
        run_thread.join(timeout=5)
        store.close()
