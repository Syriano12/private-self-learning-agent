from __future__ import annotations

import json
import stat
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from private_agent.api import build_api_service, create_app
from private_agent.worker import BackgroundWorker

TOKEN = "phase25-api-test-token"


def auth_headers(key: str = "phase25-key") -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}", "Idempotency-Key": key}


def write_fake_httpx(path: Path, marker: Path | None = None) -> Path:
    marker_line = ""
    if marker is not None:
        marker_line = f"pathlib.Path({str(marker)!r}).write_text('called')\n"
    script = textwrap.dedent(
        '''
        import pathlib
        __MARKER__
        import json
        import sys
        if '-version' in sys.argv:
            print('httpx version v1.7.0')
        elif '-h' in sys.argv:
            print('-u -json -silent -follow-redirects -rate-limit -threads -timeout -retries')
        else:
            print(json.dumps({'url': 'http://127.0.0.1:8765/health', 'status-code': 200, 'title': 'fixture'}))
        '''
    ).replace('__MARKER__', marker_line.rstrip())
    path.write_text(
        "#!/usr/bin/env python3\n" + script,
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def scanner_scope() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "program_id": "phase25-api-program",
        "authorization_reference": "phase25-written-authorization",
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=30)).isoformat(),
        "assets": [{
            "asset_id": "local-httpx",
            "host": "127.0.0.1",
            "path_prefix": "/",
            "methods": ["GET"],
            "ports": [8765],
            "active_allowed": True,
        }],
        "permitted_identities": ["alice"],
        "permitted_resources": ["resource-1"],
        "permitted_methods": ["GET"],
        "prohibited_methods": ["POST", "PUT", "PATCH", "DELETE"],
        "active_testing": True,
    }


def wait_for(fetch, predicate, timeout: float = 4.0):
    deadline = time.monotonic() + timeout
    value = fetch()
    while time.monotonic() < deadline:
        if predicate(value):
            return value
        time.sleep(0.02)
        value = fetch()
    return value


def test_httpx_api_task_requires_approval_then_fails_closed_without_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "phase25-api.sqlite3"))
    monkeypatch.setenv("AGENT_API_TOKEN", TOKEN)
    marker = tmp_path / "httpx-called"
    fake = write_fake_httpx(tmp_path / "httpx", marker)
    monkeypatch.setenv("AGENT_HTTPX_BINARY", str(fake))
    service = build_api_service(token=TOKEN)
    worker = BackgroundWorker(service.orchestrator, worker_id="phase25-api-worker", poll_interval_seconds=0.01, idle_poll_interval_seconds=0.02)
    app = create_app(service=service, worker=worker)

    with TestClient(app, raise_server_exceptions=False) as client:
        created = client.post(
            "/api/v1/bug-bounty/tasks",
            json={
                "scope": scanner_scope(),
                "scanner": "httpx",
                "target": "http://127.0.0.1:8765/health",
                "identity": "alice",
                "resource_id": "resource-1",
            },
            headers=auth_headers("phase25-httpx-task"),
        )
        assert created.status_code == 201, created.text
        task_id = created.json()["task_id"]

        approvals = wait_for(
            lambda: client.get("/api/v1/approvals?pending_only=true", headers=auth_headers()).json(),
            lambda value: isinstance(value, list) and bool(value),
        )
        assert approvals and approvals[0]["tool_or_skill"] == "httpx_recon"
        approval_id = approvals[0]["approval_id"]
        approved = client.post(f"/api/v1/approvals/{approval_id}/approve", json={}, headers=auth_headers())
        assert approved.status_code == 200, approved.text

        result = wait_for(
            lambda: client.get(f"/api/v1/bug-bounty/tasks/{task_id}", headers=auth_headers()).json(),
            lambda value: value.get("status") in {"COMPLETED", "FAILED", "EXECUTION_UNKNOWN"},
        )
        assert result["status"] == "FAILED", json.dumps(result, indent=2)
        assert result["scan"] is None
        assert result["run"] is None
        assert result["report"] == ""
        assert not marker.exists()

    service.store.close()
