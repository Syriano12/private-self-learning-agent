from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from private_agent.api import create_app
from private_agent.worker import BackgroundWorker
from tests.test_phase12_api import APITestTool, auth_headers, make_service


def _wait_until(predicate, *, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def test_api_approval_requeues_task_without_direct_store_call(tmp_path: Path) -> None:
    tool = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, tool)
    first_worker = BackgroundWorker(
        service.orchestrator,
        worker_id="phase17a-approval-first",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.02,
    )
    client = TestClient(create_app(service=service, worker=first_worker), raise_server_exceptions=False)
    first_thread = first_worker.start()
    try:
        response = client.post(
            "/api/v1/tasks",
            json={"goal": "phase17 approval task"},
            headers=auth_headers(key="phase17-approval"),
        )
        assert response.status_code == 201, response.text
        task_id = response.json()["task_id"]
        assert _wait_until(lambda: store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL")
        approval_id = next(iter(store.get_task_state(task_id)["approval_ids"].values()))
    finally:
        assert first_worker.stop(timeout=2.0)
        first_thread.join(timeout=2.0)

    approved = client.post(
        f"/api/v1/approvals/{approval_id}/approve",
        json={},
        headers=auth_headers(),
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "APPROVED"
    assert store.task_queue_record(task_id)["queue_status"] == "QUEUED"

    second_worker = BackgroundWorker(
        service.orchestrator,
        worker_id="phase17a-approval-second",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.02,
    )
    second_thread = second_worker.start()
    try:
        assert _wait_until(lambda: store.get_task_state(task_id)["status"] == "COMPLETED")
    finally:
        assert second_worker.stop(timeout=2.0)
        second_thread.join(timeout=2.0)

    assert tool.calls == 1
    store.close()


def test_queue_managed_resume_never_executes_without_worker_lease(tmp_path: Path) -> None:
    tool = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, tool)
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="phase17a-resume-fence",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.02,
    )
    client = TestClient(create_app(service=service, worker=worker), raise_server_exceptions=False)
    thread = worker.start()
    try:
        created = client.post(
            "/api/v1/tasks",
            json={"goal": "phase17 queue resume fence"},
            headers=auth_headers(key="phase17-resume-fence"),
        )
        assert created.status_code == 201, created.text
        task_id = created.json()["task_id"]
        assert _wait_until(lambda: store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL")
        approval_id = next(iter(store.get_task_state(task_id)["approval_ids"].values()))
        assert worker.stop(timeout=2.0)
        thread.join(timeout=2.0)

        approved = client.post(f"/api/v1/approvals/{approval_id}/approve", json={}, headers=auth_headers())
        assert approved.status_code == 200, approved.text
        resumed = client.post(f"/api/v1/tasks/{task_id}/resume", json={}, headers=auth_headers())
        assert resumed.status_code == 409, resumed.text
        assert resumed.json()["error_code"] == "task_queue_managed"
        assert tool.calls == 0
    finally:
        if worker.running:
            worker.stop(timeout=2.0)
        thread.join(timeout=2.0)
    store.close()


def test_app_lifespan_starts_worker_and_readiness_exposes_worker_state(tmp_path: Path) -> None:
    tool = APITestTool("safe")
    service, store, _ = make_service(tmp_path, tool)
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="phase17a-lifespan",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.02,
    )
    application = create_app(service=service, worker=worker)

    with TestClient(application, raise_server_exceptions=False) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200, ready.text
        assert ready.json()["checks"]["worker"] == "running"
        created = client.post(
            "/api/v1/tasks",
            json={"goal": "phase17 worker lifecycle task"},
            headers=auth_headers(key="phase17-lifecycle"),
        )
        assert created.status_code == 201, created.text
        task_id = created.json()["task_id"]
        assert _wait_until(
            lambda: client.get(f"/api/v1/tasks/{task_id}", headers=auth_headers()).json().get("status") == "COMPLETED"
        )

    assert worker.running is False
    assert worker.last_report is not None
    assert worker.last_report["status"] == "STOPPED"
    assert tool.calls == 1
    store.close()


def test_reusing_service_without_worker_disables_auto_requeue(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    worker = BackgroundWorker(service.orchestrator, worker_id="phase17a-reuse")
    create_app(service=service, worker=worker)
    assert service.auto_requeue_approvals is True
    create_app(service=service)
    assert service.auto_requeue_approvals is False
    store.close()


def test_worker_reconciliation_keeps_shared_sqlite_reads_usable(tmp_path: Path) -> None:
    tool = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, tool)
    created = service.create_task("phase17 sqlite read coordination", idempotency_key="phase17-read-lock")
    task_id = created.task_id
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="phase17a-read-lock",
        poll_interval_seconds=0.001,
        idle_poll_interval_seconds=0.001,
    )
    thread = worker.start()
    try:
        assert _wait_until(lambda: store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL")
        for _ in range(200):
            assert store.get_task_state(task_id)["task_id"] == task_id
            assert store.task_queue_record(task_id)["task_id"] == task_id
    finally:
        assert worker.stop(timeout=2.0)
        thread.join(timeout=2.0)
    assert worker.fatal_error is None
    store.close()
