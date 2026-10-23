from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from private_agent.api import AgentAPIService, build_api_service, create_app
from private_agent.worker import BackgroundWorker
from tests.test_phase12_api import API_TOKEN, auth_headers, make_service


def _scope() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "program_id": "local-vulnora-fixture",
        "authorization_reference": "owner-written-local-authorization",
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(days=1)).isoformat(),
        "assets": [{
            "asset_id": "fixture-api",
            "host": "fixture.local",
            "path_prefix": "/api/v1",
            "methods": ["GET"],
            "active_allowed": True,
        }],
        "permitted_identities": ["alice", "bob", "admin"],
        "permitted_resources": ["order-alice", "order-bob", "comment-bob", "report-admin"],
        "permitted_methods": ["GET"],
        "prohibited_methods": ["DELETE", "PATCH", "PUT", "POST"],
        "active_testing": True,
    }


def _wait_for(predicate, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def test_required_worker_makes_readiness_and_task_creation_fail_closed(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, __import__("tests.test_phase12_api", fromlist=["APITestTool"]).APITestTool("safe"))
    application = create_app(service=service, require_worker=True)
    client = TestClient(application, raise_server_exceptions=False)

    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["error_code"] == "not_ready"
    assert ready.json()["checks"]["worker"] == "missing"

    rejected = client.post("/api/v1/tasks", json={"goal": "must not queue without worker"}, headers=auth_headers())
    assert rejected.status_code == 503
    assert rejected.json()["error_code"] == "worker_not_ready"
    legacy_rejected = client.post("/tasks/run", json={"goal": "legacy must not queue without worker"}, headers=auth_headers())
    assert legacy_rejected.status_code == 503
    assert legacy_rejected.json()["error_code"] == "worker_not_ready"
    assert store.db.execute("SELECT COUNT(*) AS count FROM task_queue").fetchone()["count"] == 0
    store.close()


def test_bug_bounty_endpoints_reject_generic_task_ids(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, __import__("tests.test_phase12_api", fromlist=["APITestTool"]).APITestTool("safe"))
    application = create_app(service=service)
    with TestClient(application, raise_server_exceptions=False) as client:
        created = client.post("/api/v1/tasks", json={"goal": "generic task"}, headers=auth_headers())
        assert created.status_code == 201
        task_id = created.json()["task_id"]
        assert client.get(f"/api/v1/bug-bounty/tasks/{task_id}", headers=auth_headers()).status_code == 404
        assert client.get(f"/api/v1/bug-bounty/tasks/{task_id}/findings", headers=auth_headers()).status_code == 404
    store.close()


def test_required_stopped_worker_rejects_creation_before_lifespan(tmp_path: Path) -> None:
    tool = __import__("tests.test_phase12_api", fromlist=["APITestTool"]).APITestTool("safe")
    service, store, _ = make_service(tmp_path, tool)
    worker = BackgroundWorker(service.orchestrator, worker_id="phase18-not-started")
    client = TestClient(create_app(service=service, worker=worker), raise_server_exceptions=False)
    response = client.post("/api/v1/tasks", json={"goal": "must not queue"}, headers=auth_headers())
    assert response.status_code == 503
    assert response.json()["error_code"] == "worker_not_ready"
    assert store.db.execute("SELECT COUNT(*) AS count FROM task_queue").fetchone()["count"] == 0
    store.close()


def test_vulnora_bug_bounty_task_uses_real_queue_worker_and_report(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "bugbounty.sqlite3"
    monkeypatch.setenv("AGENT_DB_PATH", str(db_path))
    monkeypatch.setenv("AGENT_API_TOKEN", API_TOKEN)
    service = build_api_service(token=API_TOKEN)
    worker = BackgroundWorker(service.orchestrator, worker_id="phase18-bugbounty", poll_interval_seconds=0.01, idle_poll_interval_seconds=0.02)
    application = create_app(service=service, worker=worker)

    with TestClient(application, raise_server_exceptions=False) as client:
        assert client.get("/ready").status_code == 200
        created = client.post(
            "/api/v1/bug-bounty/tasks",
            json={"scope": _scope(), "fixture": "idor_horizontal"},
            headers=auth_headers(key="phase18-bugbounty-task"),
        )
        assert created.status_code == 201, created.text
        task_id = created.json()["task_id"]
        assert service.store.task_queue_record(task_id)["task_id"] == task_id

        assert _wait_for(lambda: client.get(f"/api/v1/bug-bounty/tasks/{task_id}", headers=auth_headers()).json()["status"] in {"COMPLETED", "FAILED"})
        result = client.get(f"/api/v1/bug-bounty/tasks/{task_id}", headers=auth_headers())
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["status"] == "COMPLETED"
        assert body["verification_status"] == "VERIFIED"
        assert body["run"]["inventory"]["endpoints"]
        assert body["run"]["findings"]
        assert "External targets: `none`" in body["report"]

        inventory = client.get(f"/api/v1/bug-bounty/tasks/{task_id}/inventory", headers=auth_headers())
        findings = client.get(f"/api/v1/bug-bounty/tasks/{task_id}/findings", headers=auth_headers())
        evidence = client.get(f"/api/v1/bug-bounty/tasks/{task_id}/evidence", headers=auth_headers())
        report = client.get(f"/api/v1/bug-bounty/tasks/{task_id}/report", headers=auth_headers())
        assert inventory.status_code == findings.status_code == evidence.status_code == report.status_code == 200
        assert inventory.json()["data"]["endpoints"]
        assert findings.json()["data"]
        assert evidence.json()["data"]["finding_evidence"]
        assert report.json()["data"]["report"]

        history = client.get("/api/v1/tasks", headers=auth_headers())
        assert history.status_code == 200
        assert any(item["task_id"] == task_id for item in history.json()["items"])

        unauthorized = client.get(f"/api/v1/bug-bounty/tasks/{task_id}", headers={"Authorization": "Bearer wrong-token"})
        assert unauthorized.status_code == 401

    service.store.close()


def test_explicit_cors_allowlist_never_allows_wildcard(tmp_path: Path, monkeypatch) -> None:
    service, store, _ = make_service(tmp_path, __import__("tests.test_phase12_api", fromlist=["APITestTool"]).APITestTool("safe"))
    monkeypatch.setenv("AGENT_ALLOWED_ORIGINS", "https://vulnora.example")
    application = create_app(service=service)
    client = TestClient(application, raise_server_exceptions=False)
    response = client.options(
        "/api/v1/tasks",
        headers={"Origin": "https://vulnora.example", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://vulnora.example"
    store.close()
