from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from private_agent.worker import BackgroundWorker
from tests.test_phase12_api import APITestTool, auth_headers, client_for, make_service


def _wait_until(predicate, *, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


class EventSink:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, event: dict) -> None:
        with self.lock:
            self.events.append(dict(event))

    def values(self, event_type: str) -> list[dict]:
        with self.lock:
            return [event for event in self.events if event.get("event_type") == event_type]


def test_run_loop_processes_multiple_api_tasks_and_persists_final_state(tmp_path: Path) -> None:
    tool = APITestTool("safe")
    service, store, _ = make_service(tmp_path, tool)
    client = client_for(service)
    created_ids: list[str] = []
    for index in range(3):
        response = client.post(
            "/api/v1/tasks",
            json={"goal": f"complete API task {index}"},
            headers=auth_headers(key=f"worker-loop-{index}"),
        )
        assert response.status_code == 201, response.text
        assert response.json()["status"] == "QUEUED"
        created_ids.append(response.json()["task_id"])

    sink = EventSink()
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="worker-loop-e2e",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.02,
        event_logger=sink,
    )
    thread = worker.start()
    try:
        assert _wait_until(lambda: len(sink.values("worker_result")) >= 3, timeout=5)
        assert [event["status"] for event in sink.values("worker_result")[:3]] == [
            "COMPLETED",
            "COMPLETED",
            "COMPLETED",
        ]
    finally:
        assert worker.stop(timeout=2.0)
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert tool.calls == 3
    assert worker.last_report is not None
    assert worker.last_report["status"] == "STOPPED"
    assert worker.last_report["processed"] == 3
    assert len(sink.values("worker_result")) >= 3
    for task_id in created_ids:
        response = client.get(f"/api/v1/tasks/{task_id}", headers=auth_headers())
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "COMPLETED"
        assert body["verification_status"] == "VERIFIED"
        assert store.task_queue_record(task_id)["queue_status"] == "COMPLETED"
    store.close()


def test_idle_polling_is_bounded_and_shutdown_is_graceful(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    sink = EventSink()
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="worker-idle",
        idle_poll_interval_seconds=0.05,
        poll_interval_seconds=0.01,
        event_logger=sink,
    )
    started_at = time.monotonic()
    thread = worker.start()
    try:
        assert _wait_until(lambda: len(sink.values("worker_idle")) >= 2, timeout=3)
        assert worker.stop(timeout=2.0)
        thread.join(timeout=2.0)
    finally:
        worker.stop(timeout=2.0)
        thread.join(timeout=2.0)

    elapsed = time.monotonic() - started_at
    assert not thread.is_alive()
    assert elapsed >= 0.04
    assert worker.last_report is not None
    assert worker.last_report["status"] == "STOPPED"
    assert worker.last_report["idle_polls"] <= 4
    store.close()


def test_recoverable_queue_error_is_backed_off_and_loop_continues(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    sink = EventSink()
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="worker-recoverable",
        error_backoff_seconds=0.05,
        idle_poll_interval_seconds=0.01,
        event_logger=sink,
    )
    calls = {"count": 0}

    def flaky_run_once() -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            raise sqlite3.OperationalError("database is locked")
        worker.stop()
        return {"status": "NO_TASK", "processed": False, "task_id": ""}

    worker.run_once = flaky_run_once  # type: ignore[method-assign]
    started_at = time.monotonic()
    report = worker.run_forever()

    assert report["status"] == "STOPPED"
    assert report["recoverable_errors"] == 1
    assert calls["count"] == 2
    assert sink.values("worker_recoverable_error")
    assert time.monotonic() - started_at >= 0.04
    store.close()


def test_fatal_worker_error_is_not_swallowed(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    sink = EventSink()
    worker = BackgroundWorker(service.orchestrator, worker_id="worker-fatal", event_logger=sink)

    def fatal_run_once() -> dict:
        raise ValueError("programming failure")

    worker.run_once = fatal_run_once  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="programming failure"):
        worker.run_forever()

    assert worker.last_report is not None
    assert worker.last_report["status"] == "FATAL"
    assert sink.values("worker_fatal_error")
    store.close()


def test_multiple_loops_are_rejected_and_shutdown_does_not_fake_completion(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    tool = APITestTool("safe")
    original_execute = tool.execute

    def blocking_execute(inputs, context):
        entered.set()
        assert release.wait(timeout=5)
        return original_execute(inputs, context)

    tool.execute = blocking_execute
    service, store, _ = make_service(tmp_path, tool)
    created = store.create_queued_task("controlled shutdown task")
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="worker-shutdown",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.01,
    )
    thread = worker.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(RuntimeError, match="worker_already_running"):
            worker.run_forever()
        assert worker.stop(timeout=0.05) is False
        assert thread.is_alive()
        assert store.get_task_state(created["task_id"])["status"] != "COMPLETED"
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert store.get_task_state(created["task_id"])["status"] == "COMPLETED"
        assert tool.calls == 1
    finally:
        release.set()
        worker.stop(timeout=2.0)
        thread.join(timeout=2.0)
        store.close()


def test_approval_required_task_waits_then_executes_after_owner_approval(tmp_path: Path) -> None:
    tool = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, tool)
    client = client_for(service)
    response = client.post(
        "/api/v1/tasks",
        json={"goal": "approval-gated API task"},
        headers=auth_headers(key="worker-approval"),
    )
    assert response.status_code == 201, response.text
    task_id = response.json()["task_id"]
    sink = EventSink()
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="worker-approval",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.02,
        event_logger=sink,
    )
    first_thread = worker.start()
    try:
        assert _wait_until(lambda: bool(sink.values("worker_result")), timeout=5)
        assert sink.values("worker_result")[-1]["status"] == "WAITING_APPROVAL"
        waiting = client.get(f"/api/v1/tasks/{task_id}", headers=auth_headers())
        assert waiting.status_code == 200
        assert waiting.json()["status"] == "WAITING_APPROVAL"
        assert tool.calls == 0
        approval_id = next(iter(store.get_task_state(task_id)["approval_ids"].values()))
        approved = client.post(f"/api/v1/approvals/{approval_id}/approve", json={}, headers=auth_headers())
        assert approved.status_code == 200, approved.text
        store.requeue_waiting_approval_task(task_id)
    finally:
        assert worker.stop(timeout=2.0)
        first_thread.join(timeout=2.0)

    sink2 = EventSink()
    worker.event_logger = sink2
    second_thread = worker.start()
    try:
        assert _wait_until(
            lambda: any(event.get("status") == "COMPLETED" for event in sink2.values("worker_result")),
            timeout=5,
        )
    finally:
        assert worker.stop(timeout=2.0)
        second_thread.join(timeout=2.0)

    assert tool.calls == 1
    assert store.get_task_state(task_id)["status"] == "COMPLETED"
    assert store.task_queue_record(task_id)["queue_status"] == "COMPLETED"
    store.close()


def test_execution_unknown_is_not_retried_by_controlled_loop(tmp_path: Path) -> None:
    tool = APITestTool("safe")

    def interrupt_after_execution(checkpoint_type, state):
        if checkpoint_type == "AFTER_EXECUTION":
            raise RuntimeError("controlled interruption after side effect")

    service, store, _ = make_service(tmp_path, tool, hook=interrupt_after_execution)
    client = client_for(service)
    response = client.post(
        "/api/v1/tasks",
        json={"goal": "uncertain controlled task"},
        headers=auth_headers(key="worker-unknown"),
    )
    assert response.status_code == 201, response.text
    task_id = response.json()["task_id"]
    sink = EventSink()
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="worker-unknown",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.02,
        event_logger=sink,
    )
    thread = worker.start()
    try:
        assert _wait_until(lambda: bool(sink.values("worker_result")), timeout=5)
        assert sink.values("worker_result")[-1]["status"] == "EXECUTION_UNKNOWN"
    finally:
        assert worker.stop(timeout=2.0)
        thread.join(timeout=2.0)

    assert tool.calls == 1
    assert client.get(f"/api/v1/tasks/{task_id}", headers=auth_headers()).json()["status"] == "EXECUTION_UNKNOWN"
    assert worker.run_once()["status"] == "NO_TASK"
    assert client.get(f"/api/v1/tasks/{task_id}", headers=auth_headers()).json()["status"] == "EXECUTION_UNKNOWN"
    assert store.task_queue_record(task_id)["queue_status"] == "EXECUTION_UNKNOWN"
    store.close()
