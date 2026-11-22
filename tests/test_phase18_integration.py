from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from private_agent.api import AgentAPIService, build_api_service, create_app
from private_agent.auth import AuthManager
from private_agent.core.knowledge import KnowledgeAttestor, KnowledgeValidationError
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


def test_queue_worker_accepts_valid_goal_longer_than_knowledge_query_limit(tmp_path: Path) -> None:
    tool = __import__("tests.test_phase12_api", fromlist=["APITestTool"]).APITestTool("safe")
    service, store, _ = make_service(tmp_path, tool)
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="phase18-long-goal",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.01,
    )
    application = create_app(service=service, worker=worker)
    goal = "authorized local assessment " + ("detail " * 180)
    try:
        with TestClient(application, raise_server_exceptions=False) as client:
            created = client.post("/api/v1/tasks", json={"goal": goal}, headers=auth_headers(key="phase18-long-goal"))
            assert created.status_code == 201, created.text
            task_id = created.json()["task_id"]
            assert _wait_for(lambda: store.get_task_state(task_id)["status"] in {"COMPLETED", "FAILED"})
            state = store.get_task_state(task_id)
            assert len(goal) > 1024
            assert state["status"] == "COMPLETED"
    finally:
        store.close()


def test_task_goal_boundary_rejects_wrong_type_and_oversized_value(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, __import__("tests.test_phase12_api", fromlist=["APITestTool"]).APITestTool("safe"))
    try:
        with pytest.raises(ValueError, match="goal_must_be_string"):
            store.create_queued_task(123)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="goal_too_long"):
            store.create_queued_task("x" * 4001)
        assert store.db.execute("SELECT COUNT(*) AS count FROM task_queue").fetchone()["count"] == 0
    finally:
        store.close()


def test_knowledge_query_keeps_bounded_input_validation(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, __import__("tests.test_phase12_api", fromlist=["APITestTool"]).APITestTool("safe"))
    try:
        assert store.search_knowledge("x" * 4000) == []
        with pytest.raises(KnowledgeValidationError, match="knowledge_query_invalid"):
            store.search_knowledge("x" * 4001)
    finally:
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


def test_owner_finding_review_requires_evidence_gate_and_is_audited(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "finding-review.sqlite3"
    monkeypatch.setenv("AGENT_DB_PATH", str(db_path))
    monkeypatch.setenv("AGENT_API_TOKEN", API_TOKEN)
    service = build_api_service(token=API_TOKEN)
    worker = BackgroundWorker(service.orchestrator, worker_id="phase18-finding-review", poll_interval_seconds=0.01, idle_poll_interval_seconds=0.02)
    application = create_app(service=service, worker=worker)

    with TestClient(application, raise_server_exceptions=False) as client:
        created = client.post(
            "/api/v1/bug-bounty/tasks", json={"scope": _scope(), "fixture": "idor_horizontal"},
            headers=auth_headers(key="phase18-finding-review"),
        )
        assert created.status_code == 201, created.text
        task_id = created.json()["task_id"]
        assert _wait_for(lambda: client.get(f"/api/v1/bug-bounty/tasks/{task_id}", headers=auth_headers()).json().get("status") in {"COMPLETED", "FAILED"})
        body = client.get(f"/api/v1/bug-bounty/tasks/{task_id}", headers=auth_headers()).json()
        finding_id = body["run"]["findings"][0]["finding_id"]
        confirmed = client.post(
            f"/api/v1/bug-bounty/tasks/{task_id}/findings/{finding_id}/review",
            json={"decision": "CONFIRM", "note": "owner review of validated local fixture"},
            headers=auth_headers(key="phase18-finding-review"),
        )
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["evidence_validated"] is True
        confirmed_review = confirmed.json()
        learned = [
            item for item in service.orchestrator.learning_memory.list()
            if item.metadata.get("review_id") == confirmed_review["review_id"]
        ]
        assert len(learned) == 1
        assert learned[0].metadata["verified_outcome"] is True
        assert learned[0].preferred_action["type"] == "require_evidence_review"
        assert learned[0].avoided_action["type"] == "auto_confirm_finding"
        assert learned[0].metadata["non_authoritative"] is True
        reviews = client.get(f"/api/v1/bug-bounty/tasks/{task_id}/reviews", headers=auth_headers())
        assert reviews.status_code == 200 and reviews.json()[0]["decision"] == "CONFIRM"
        assert any(event.get("event_type") == "BUGBOUNTY_FINDING_REVIEW" for event in service.store.all_audit_events(task_id=task_id))

        second = client.post(
            "/api/v1/bug-bounty/tasks", json={"scope": _scope(), "fixture": "idor_horizontal"},
            headers=auth_headers(key="phase18-finding-review-second"),
        )
        assert second.status_code == 201, second.text
        second_task_id = second.json()["task_id"]
        assert _wait_for(
            lambda: client.get(f"/api/v1/bug-bounty/tasks/{second_task_id}", headers=auth_headers()).json().get("status") in {"COMPLETED", "FAILED"}
        )
        second_state = service.store.get_task_state(second_task_id)
        assert second_state is not None
        reused = second_state["context"].get("learned_strategies_used", [])
        assert any(item.get("metadata", {}).get("review_id") == confirmed_review["review_id"] for item in reused)
    service.store.close()


def test_knowledge_attestation_binds_authenticated_review_actor_and_is_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "reviewer-attestation.sqlite3"
    attestation_key = "review-attestation-test-key-32-bytes-minimum"
    monkeypatch.setenv("AGENT_KNOWLEDGE_ATTESTATION_KEY", attestation_key)
    monkeypatch.setenv("AGENT_API_TOKEN", API_TOKEN)
    service = build_api_service(token=None, database_path=str(database))
    owner = AuthManager(service.store).initialize_owner(
        "ashbm1995@gmail.com", "owner-password-for-reviewer-attestation-123"
    )
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="reviewer-attestation-integration",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.02,
    )

    with TestClient(create_app(service=service, worker=worker), raise_server_exceptions=False) as client:
        login = client.post(
            "/api/v1/auth/login",
            json={"email": "ashbm1995@gmail.com", "password": "owner-password-for-reviewer-attestation-123"},
        )
        assert login.status_code == 200, login.text
        csrf = str(client.cookies.get("vulnora_csrf"))
        headers = {"X-CSRF-Token": csrf}
        created = client.post(
            "/api/v1/bug-bounty/tasks",
            json={"scope": _scope(), "fixture": "idor_horizontal"},
            headers=headers,
        )
        assert created.status_code == 201, created.text
        task_id = created.json()["task_id"]
        assert _wait_for(
            lambda: client.get(f"/api/v1/bug-bounty/tasks/{task_id}").json().get("status")
            in {"COMPLETED", "FAILED"}
        )
        task = client.get(f"/api/v1/bug-bounty/tasks/{task_id}")
        assert task.status_code == 200, task.text
        finding_id = task.json()["run"]["findings"][0]["finding_id"]
        review = client.post(
            f"/api/v1/bug-bounty/tasks/{task_id}/findings/{finding_id}/review",
            json={"decision": "CONFIRM", "note": "authenticated owner reviewed validated local evidence"},
            headers=headers,
        )
        assert review.status_code == 200, review.text
        review_body = review.json()
        assert review_body["actor_user_id"] == owner.user_id
        assert review_body["evidence_validated"] is True

        route = (
            f"/api/v1/bug-bounty/tasks/{task_id}/reviews/"
            f"{review_body['review_id']}/knowledge-attestation"
        )
        original_finding_for_review = service._finding_for_review

        def changed_finding_evidence(target_task_id, target_finding_id, principal):
            state, raw, finding = original_finding_for_review(target_task_id, target_finding_id, principal)
            finding.evidence[0].observed["post_review_mutation"] = True
            return state, raw, finding

        monkeypatch.setattr(service, "_finding_for_review", changed_finding_evidence)
        stale = client.post(route, headers=headers)
        assert stale.status_code == 409, stale.text
        assert stale.json()["error_code"] == "review_evidence_stale"
        monkeypatch.setattr(service, "_finding_for_review", original_finding_for_review)

        issued = client.post(
            route,
            json={"verified_by": "attacker-selected-reviewer", "claim": "caller-controlled"},
            headers=headers,
        )
        assert issued.status_code == 200, issued.text
        attestation = issued.json()["attestation"]
        assert attestation["verified_by"] == owner.user_id
        assert attestation["attestation_id"] == f"attestation-{review_body['review_id']}"
        assert attestation["claim_snapshot"]["metadata"]["review_id"] == review_body["review_id"]
        assert attestation["evidence_snapshots"][0]["metadata"]["evidence_sha256"] == review_body["evidence_sha256"]
        assert KnowledgeAttestor(attestation_key).verify(attestation)

        repeated = client.post(route, headers=headers)
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["attestation"] == attestation
        stored = service.store.db.execute(
            "SELECT COUNT(*) AS count FROM knowledge_attestations WHERE attestation_id = ?",
            (attestation["attestation_id"],),
        ).fetchone()
        assert int(stored["count"]) == 1

        legacy = client.post(
            route,
            headers={"Authorization": f"Bearer {API_TOKEN}", "X-CSRF-Token": csrf},
        )
        assert legacy.status_code == 401
        assert legacy.json()["error_code"] == "legacy_bearer_disabled"

        unconfirmed_task = client.post(
            "/api/v1/bug-bounty/tasks",
            json={"scope": _scope(), "fixture": "idor_horizontal"},
            headers=headers,
        )
        assert unconfirmed_task.status_code == 201, unconfirmed_task.text
        unconfirmed_task_id = unconfirmed_task.json()["task_id"]
        assert _wait_for(
            lambda: client.get(f"/api/v1/bug-bounty/tasks/{unconfirmed_task_id}").json().get("status")
            in {"COMPLETED", "FAILED"}
        )
        unconfirmed_body = client.get(f"/api/v1/bug-bounty/tasks/{unconfirmed_task_id}").json()
        unconfirmed_finding_id = unconfirmed_body["run"]["findings"][0]["finding_id"]
        unconfirmed_review = client.post(
            f"/api/v1/bug-bounty/tasks/{unconfirmed_task_id}/findings/{unconfirmed_finding_id}/review",
            json={"decision": "INCONCLUSIVE", "note": "not confirmed"},
            headers=headers,
        )
        assert unconfirmed_review.status_code == 200, unconfirmed_review.text
        unconfirmed_route = (
            f"/api/v1/bug-bounty/tasks/{unconfirmed_task_id}/reviews/"
            f"{unconfirmed_review.json()['review_id']}/knowledge-attestation"
        )
        rejected = client.post(unconfirmed_route, headers=headers)
        assert rejected.status_code == 409, rejected.text
        assert rejected.json()["error_code"] == "review_not_confirmed"
        assert service.store.db.execute(
            "SELECT 1 FROM knowledge_attestations WHERE attestation_id = ?",
            (f"attestation-{unconfirmed_review.json()['review_id']}",),
        ).fetchone() is None

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
