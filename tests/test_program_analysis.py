from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import httpx
from fastapi.testclient import TestClient

from private_agent.api import create_app
from private_agent.auth import AuthManager
from private_agent.bugbounty.passive_research import PassiveResearchAdapter
from private_agent.bugbounty.programs import ProgramAnalysisRepository, PublicPolicyRetriever, analyze_program, canonical_program_url
from private_agent.storage import Store
from private_agent.worker import BackgroundWorker
from tests.test_phase12_api import APITestTool, make_service
from tests.test_phase_auth_rbac import OWNER_EMAIL, OWNER_PASSWORD, _login


def scope_payload() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "program_id": "local-program",
        "authorization_reference": "written-authorization-local-only",
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "assets": [{"asset_id": "fixture-api", "host": "fixture.local", "path_prefix": "/api/v1", "methods": ["GET"], "active_allowed": False}],
        "permitted_identities": ["alice"],
        "permitted_resources": ["order-alice"],
        "permitted_methods": ["GET"],
        "prohibited_methods": ["DELETE", "PATCH", "PUT", "POST"],
        "active_testing": False,
    }


def test_program_url_is_canonicalized_without_fetching_or_userinfo() -> None:
    assert canonical_program_url("HTTPS://Example.COM:443/policy/") == "https://example.com/policy"
    with pytest.raises(ValueError, match="userinfo"):
        canonical_program_url("https://user:password@example.com/policy")


def test_analysis_hashes_policy_and_never_promotes_caller_text_to_authorization() -> None:
    raw_reference = "written-authorization-local-only"
    analysis = analyze_program("https://Example.COM/policy", json.dumps(scope_payload()))
    assert analysis.status == "READY_FOR_SCOPE_REVIEW"
    assert analysis.source.content_sha256
    public = analysis.to_dict()
    assert raw_reference not in json.dumps(public)
    assert str(public["scope_proposal"]["authorization_reference"]).startswith("sha256:")
    assert "program_url_not_fetched_automatically" in analysis.limitations


def test_missing_or_expired_policy_is_review_required_and_not_executable() -> None:
    empty = analyze_program("https://example.com/policy", "")
    assert empty.status == "REVIEW_REQUIRED"
    assert empty.scope_proposal is None
    expired = scope_payload()
    expired["issued_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    expired["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    result = analyze_program("https://example.com/policy", json.dumps(expired))
    assert result.status == "REVIEW_REQUIRED"
    assert result.scope_proposal is None
    assert any("scope_expired" in item for item in result.ambiguities)


def test_future_issued_scope_is_not_active_before_its_start_time() -> None:
    from private_agent.bugbounty.models import ScopeDecision, ScopeDefinition

    future = scope_payload()
    future["issued_at"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    future["expires_at"] = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    with pytest.raises(ValueError, match="scope_not_yet_active"):
        ScopeDefinition.from_dict(future)
    analysis = analyze_program("https://example.com/policy", json.dumps(future))
    assert analysis.status == "REVIEW_REQUIRED"
    assert analysis.scope_proposal is None
    assert any("scope_not_yet_active" in item for item in analysis.ambiguities)
    future["issued_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    future["expires_at"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    active = ScopeDefinition.from_dict(future)
    decision, reason = active.authorize("https://fixture.local/api/v1", now=datetime.now(timezone.utc))
    assert decision is ScopeDecision.ALLOWED
    assert reason == "explicit_scope_match"


def test_line_policy_parser_requires_explicit_fields() -> None:
    result = analyze_program("https://example.com/policy", "program-id: local\nin-scope: fixture.local /api/v1 GET")
    assert result.status == "REVIEW_REQUIRED"
    assert "authorization_reference_missing" in result.ambiguities
    assert "permitted_identities_missing" in result.ambiguities


def test_repository_persists_and_scopes_analysis_by_owner(tmp_path: Path) -> None:
    store = Store(tmp_path / "programs.sqlite3")
    repository = ProgramAnalysisRepository(store)
    analysis = repository.save(analyze_program("https://example.com/policy", json.dumps(scope_payload())), owner_user_id="user-a")
    assert analysis.analysis_id.startswith("PROGRAM-")
    assert repository.get(analysis.analysis_id, owner_user_id="user-a") is not None
    assert repository.get(analysis.analysis_id, owner_user_id="user-b") is None
    store.close()
    reopened = Store(tmp_path / "programs.sqlite3")
    assert ProgramAnalysisRepository(reopened).get(analysis.analysis_id, owner_user_id="user-a") is not None
    reopened.close()


def test_api_analysis_requires_csrf_and_returns_owner_scoped_review(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    assert client.post("/api/v1/bug-bounty/programs/analyze", json={"program_url": "https://example.com/policy", "policy_text": "{}"}).status_code == 401
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    response = client.post(
        "/api/v1/bug-bounty/programs/analyze",
        json={"program_url": "https://example.com/policy", "policy_text": json.dumps(scope_payload())},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "READY_FOR_SCOPE_REVIEW"
    assert payload["source"]["content_sha256"]
    assert "written-authorization-local-only" not in response.text
    analysis_id = payload["analysis_id"]
    assert client.get(f"/api/v1/bug-bounty/programs/{analysis_id}").status_code == 200
    store.close()


def test_explicit_allowlisted_public_policy_retrieval_is_bounded_and_untrusted(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    service.public_policy_retriever = PublicPolicyRetriever(
        PassiveResearchAdapter(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                    text="<html><body>In scope: api.example.com; out of scope: admin.example.com; token: SECRET_PUBLIC_TEXT</body></html>",
                )
            ),
            resolver=lambda _host: ("93.184.216.34",),
        )
    )
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    response = client.post(
        "/api/v1/bug-bounty/programs/analyze",
        json={
            "program_url": "https://hackerone.com/example-public",
            "policy_text": "",
            "source_type": "hackerone_program",
            "fetch_public_policy": True,
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "REVIEW_REQUIRED"
    assert payload["source"]["source_type"] == "hackerone_program"
    assert payload["source"]["content_sha256"]
    assert payload["scope_proposal"] is None
    assert payload["observations"]["authorization_derived"] is False
    assert "api.example.com" in payload["observations"]["candidate_hosts"]
    assert "admin.example.com" in payload["observations"]["candidate_hosts"]
    assert "public_policy_not_authorization" in payload["limitations"]
    assert "program_url_not_fetched_automatically" not in payload["limitations"]
    assert "SECRET_PUBLIC_TEXT" not in response.text
    store.close()


def test_public_policy_retrieval_rejects_unsupported_source_before_network(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    response = client.post(
        "/api/v1/bug-bounty/programs/analyze",
        json={
            "program_url": "https://example.com/policy",
            "policy_text": "",
            "source_type": "user_provided_policy",
            "fetch_public_policy": True,
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "validation_error"
    store.close()


def test_scope_wildcard_is_subdomain_only_and_exclusions_win() -> None:
    from private_agent.bugbounty.models import ScopeDecision, ScopeDefinition

    payload = scope_payload()
    payload["assets"][0]["host"] = "*.example.com"
    payload["excluded_hosts"] = ["blocked.example.com"]
    scope = ScopeDefinition.from_dict(payload)
    assert scope.authorize("https://api.example.com/api/v1", now=datetime.now(timezone.utc))[0] == ScopeDecision.ALLOWED
    assert scope.authorize("https://example.com/api/v1", now=datetime.now(timezone.utc))[0] == ScopeDecision.DENIED
    assert scope.authorize("https://blocked.example.com/api/v1", now=datetime.now(timezone.utc))[1] == "host_excluded"
    with pytest.raises(ValueError, match="host_wildcard_invalid"):
        ScopeDefinition.from_dict({**payload, "assets": [{**payload["assets"][0], "host": "example.*.com"}]})


def test_research_stage_lifecycle_is_bounded_and_persists_after_reopen(tmp_path: Path) -> None:
    from private_agent.bugbounty.research import ResearchStageRepository, ResearchStageStatus, ResearchStageType, StageBudget

    path = tmp_path / "research-stages.sqlite3"
    store = Store(path)
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())), owner_user_id="user-a"
    )
    stages = ResearchStageRepository(store)
    queued = stages.create(analysis.analysis_id, ResearchStageType.INVENTORY, owner_user_id="user-a", budget=StageBudget(max_items=5, timeout_seconds=30, max_output_bytes=1000))
    assert queued.stage_status == ResearchStageStatus.QUEUED.value
    running = stages.start(queued.stage_id, owner_user_id="user-a")
    assert running.stage_status == ResearchStageStatus.RUNNING.value
    finished = stages.finish(running.stage_id, ResearchStageStatus.PARTIAL, owner_user_id="user-a", partial_result={"items": ["fixture.local"]}, error_code="sandbox_unavailable")
    assert finished.stage_status == ResearchStageStatus.PARTIAL.value
    assert finished.error_code == "sandbox_unavailable"
    with pytest.raises(ValueError, match="not_startable"):
        stages.start(queued.stage_id, owner_user_id="user-a")
    store.close()
    reopened = Store(path)
    persisted = ResearchStageRepository(reopened).get(queued.stage_id, owner_user_id="user-a")
    assert persisted is not None and persisted.stage_status == ResearchStageStatus.PARTIAL.value
    assert ResearchStageRepository(reopened).get(queued.stage_id, owner_user_id="user-b") is None
    reopened.close()


def test_api_research_stage_is_review_only_and_cancel_is_owner_scoped(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    analysis_response = client.post(
        "/api/v1/bug-bounty/programs/analyze",
        json={"program_url": "https://example.com/policy", "policy_text": json.dumps(scope_payload())},
        headers={"X-CSRF-Token": csrf},
    )
    assert analysis_response.status_code == 200, analysis_response.text
    analysis_id = analysis_response.json()["analysis_id"]
    created = client.post(
        f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
        json={"stage_type": "INVENTORY", "budget": {"max_items": 3, "timeout_seconds": 10}},
        headers={"X-CSRF-Token": csrf},
    )
    assert created.status_code == 201, created.text
    stage_id = created.json()["stage_id"]
    assert created.json()["stage_status"] == "QUEUED"
    assert client.get(f"/api/v1/bug-bounty/programs/{analysis_id}/stages").json()[0]["stage_id"] == stage_id
    cancelled = client.post(f"/api/v1/bug-bounty/research-stages/{stage_id}/cancel", headers={"X-CSRF-Token": csrf})
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["stage_status"] == "CANCELLED"
    store.close()


def test_api_stage_creation_reaches_local_worker_and_persists_result(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    analysis_response = client.post(
        "/api/v1/bug-bounty/programs/analyze",
        json={"program_url": "https://example.com/policy", "policy_text": json.dumps(scope_payload())},
        headers={"X-CSRF-Token": csrf},
    )
    assert analysis_response.status_code == 200, analysis_response.text
    analysis_id = analysis_response.json()["analysis_id"]
    created = client.post(
        f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
        json={"stage_type": "PROGRAM_ANALYSIS", "budget": {"max_items": 3, "timeout_seconds": 10}},
        headers={"X-CSRF-Token": csrf},
    )
    assert created.status_code == 201, created.text
    stage_id = created.json()["stage_id"]
    worker = BackgroundWorker(service.orchestrator, worker_id="api-stage-worker", idle_poll_interval_seconds=0.01)
    result = worker.run_once()
    assert result["status"] == "STAGE_COMPLETED"
    fetched = client.get(f"/api/v1/bug-bounty/research-stages/{stage_id}")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["stage_status"] == "COMPLETED"
    assert fetched.json()["result"]["scope_status"] == "READY_FOR_SCOPE_REVIEW"
    store.close()


def test_api_inventory_stage_uses_local_artifact_and_never_exposes_raw_secret(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    analysis_response = client.post(
        "/api/v1/bug-bounty/programs/analyze",
        json={"program_url": "https://example.com/policy", "policy_text": json.dumps(scope_payload())},
        headers={"X-CSRF-Token": csrf},
    )
    assert analysis_response.status_code == 200, analysis_response.text
    analysis_id = analysis_response.json()["analysis_id"]
    secret = "api-token-not-for-output"
    url_password = "url-password-not-for-output"
    query_secret = "query-secret-not-for-output"
    created = client.post(
        f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
        json={
            "stage_type": "INVENTORY",
            "budget": {"max_items": 10, "timeout_seconds": 10},
            "input_payload": {
                "artifacts": [
                    {
                        "name": "routes.json",
                        "content_type": "application/json",
                        "content": json.dumps({"paths": {"/orders": {"get": {}}}}),
                    },
                    {"name": "bundle.js", "content_type": "text/javascript", "content": f"const token = '{secret}'; fetch('/api/orders'); https://alice:{url_password}@example.com/api?token={query_secret}&safe=ok"},
                ]
            },
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert created.status_code == 201, created.text
    assert secret not in created.text
    assert url_password not in created.text
    assert query_secret not in created.text
    stage_id = created.json()["stage_id"]
    assert "input_payload" not in created.json()
    worker_result = BackgroundWorker(service.orchestrator, worker_id="api-inventory-worker").run_once()
    assert worker_result["status"] == "STAGE_COMPLETED"
    fetched = client.get(f"/api/v1/bug-bounty/research-stages/{stage_id}")
    assert fetched.status_code == 200, fetched.text
    assert secret not in fetched.text
    assert url_password not in fetched.text
    assert query_secret not in fetched.text
    assert "/orders" in fetched.json()["result"]["endpoints"]
    stored = store.db.execute("SELECT input_json FROM bugbounty_research_stages WHERE stage_id = ?", (stage_id,)).fetchone()
    assert stored is not None
    assert all(value not in str(stored["input_json"]) for value in (secret, url_password, query_secret))
    store.close()


def test_api_passive_source_is_allowlisted_and_input_is_not_exposed(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    analysis_id = client.post(
        "/api/v1/bug-bounty/programs/analyze",
        json={"program_url": "https://example.com/policy", "policy_text": json.dumps(scope_payload())},
        headers={"X-CSRF-Token": csrf},
    ).json()["analysis_id"]
    valid = client.post(
        f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
        json={
            "stage_type": "PASSIVE_RESEARCH",
            "budget": {"max_items": 2, "timeout_seconds": 10},
            "input_payload": {"sources": [{"source_type": "hackerone_security", "url": "https://hackerone.com/security"}]},
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert valid.status_code == 201, valid.text
    assert "input_payload" not in valid.json()
    stage_id = valid.json()["stage_id"]
    stored = store.db.execute("SELECT input_json FROM bugbounty_research_stages WHERE stage_id = ?", (stage_id,)).fetchone()
    assert stored is not None and "https://hackerone.com/security" in stored["input_json"]
    for source in (
        {"source_type": "hackerone_security", "url": "https://example.com/security"},
        {"source_type": "hackerone_security", "url": "https://hackerone.com/security?next=https://example.com"},
        {"source_type": "hackerone_security", "url": "https://user:pass@hackerone.com/security"},
    ):
        rejected = client.post(
            f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
            json={"stage_type": "PASSIVE_RESEARCH", "input_payload": {"sources": [source]}},
            headers={"X-CSRF-Token": csrf},
        )
        assert rejected.status_code == 422, rejected.text
    store.close()


def test_api_redacts_quoted_json_secrets_before_persistence(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    analysis_id = client.post(
        "/api/v1/bug-bounty/programs/analyze",
        json={"program_url": "https://example.com/policy", "policy_text": json.dumps(scope_payload())},
        headers={"X-CSRF-Token": csrf},
    ).json()["analysis_id"]
    secrets_in_json = {"token": "JSON_TOKEN_SECRET", "nested": {"password": "JSON_PASSWORD_SECRET"}, "Authorization": "Bearer JSON_AUTH_SECRET"}
    created = client.post(
        f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
        json={"stage_type": "INVENTORY", "input_payload": {"artifacts": [{"name": "token-secret.json", "content_type": "application/json", "content": json.dumps(secrets_in_json)}]}},
        headers={"X-CSRF-Token": csrf},
    )
    assert created.status_code == 201, created.text
    stage_id = created.json()["stage_id"]
    stored = store.db.execute("SELECT input_json FROM bugbounty_research_stages WHERE stage_id = ?", (stage_id,)).fetchone()
    assert stored is not None
    persisted = str(stored["input_json"])
    assert all(secret not in persisted for secret in ("JSON_TOKEN_SECRET", "JSON_PASSWORD_SECRET", "JSON_AUTH_SECRET"))
    assert "token-secret.json" not in created.text
    assert "input_payload" not in created.json()
    store.close()


def test_api_rejects_oversized_or_unbounded_local_stage_artifacts(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    analysis_response = client.post(
        "/api/v1/bug-bounty/programs/analyze",
        json={"program_url": "https://example.com/policy", "policy_text": json.dumps(scope_payload())},
        headers={"X-CSRF-Token": csrf},
    )
    analysis_id = analysis_response.json()["analysis_id"]
    oversized = {"name": "too-large.txt", "content_type": "text/plain", "content": "x" * 64_001}
    assert client.post(
        f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
        json={"stage_type": "INVENTORY", "input_payload": {"artifacts": [oversized]}},
        headers={"X-CSRF-Token": csrf},
    ).status_code == 422
    too_many = [
        {"name": f"file-{index}.txt", "content_type": "text/plain", "content": "fixture"}
        for index in range(9)
    ]
    assert client.post(
        f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
        json={"stage_type": "INVENTORY", "input_payload": {"artifacts": too_many}},
        headers={"X-CSRF-Token": csrf},
    ).status_code == 422
    assert client.post(
        f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
        json={"stage_type": "INVENTORY", "input_payload": {"unexpected": True}},
        headers={"X-CSRF-Token": csrf},
    ).status_code == 422
    store.close()


def test_owner_created_stage_executes_analysis_owned_by_another_principal(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="operator-user",
    )
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    created = client.post(
        f"/api/v1/bug-bounty/programs/{analysis.analysis_id}/stages",
        json={
            "stage_type": "INVENTORY",
            "input_payload": {"artifacts": [{"name": "routes.json", "content_type": "application/json", "content": "{\"paths\":{\"/health\":{\"get\":{}}}}"}]},
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert created.status_code == 201, created.text
    stage_id = created.json()["stage_id"]
    assert BackgroundWorker(service.orchestrator, worker_id="owner-analysis-worker").run_once()["status"] == "STAGE_COMPLETED"
    fetched = client.get(f"/api/v1/bug-bounty/research-stages/{stage_id}")
    assert fetched.status_code == 200 and fetched.json()["stage_status"] == "COMPLETED"
    store.close()


def test_report_stage_rejects_client_artifacts_and_incomplete_sources(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    analysis_id = client.post(
        "/api/v1/bug-bounty/programs/analyze",
        json={"program_url": "https://example.com/policy", "policy_text": json.dumps(scope_payload())},
        headers={"X-CSRF-Token": csrf},
    ).json()["analysis_id"]
    pending = client.post(
        f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
        json={"stage_type": "INVENTORY", "input_payload": {"artifacts": [{"name": "routes.json", "content_type": "application/json", "content": "{\"paths\":{}}"}]}},
        headers={"X-CSRF-Token": csrf},
    )
    assert pending.status_code == 201, pending.text
    for payload in (
        {"artifacts": [{"name": "copied.json", "content_type": "application/json", "content": "{\"kind\":\"LOCAL_INVENTORY\"}"}]},
        {"source_stage_ids": [pending.json()["stage_id"]]},
    ):
        rejected = client.post(
            f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
            json={"stage_type": "REPORT_PREPARATION", "input_payload": payload},
            headers={"X-CSRF-Token": csrf},
        )
        assert rejected.status_code == 422, rejected.text
    store.close()


def test_report_stage_binds_only_completed_same_owner_same_analysis_reportable_results(tmp_path: Path) -> None:
    from private_agent.bugbounty.research import ResearchStageRepository, ResearchStageStatus, ResearchStageType

    store = Store(tmp_path / "report-lineage-boundaries.sqlite3")
    programs = ProgramAnalysisRepository(store)
    analysis_a = programs.save(analyze_program("https://example.com/a", json.dumps(scope_payload())), owner_user_id="owner-a")
    analysis_b = programs.save(analyze_program("https://example.com/b", json.dumps(scope_payload())), owner_user_id="owner-a")
    stages = ResearchStageRepository(store)

    def complete(analysis_id: str, stage_type: ResearchStageType, owner: str, result: dict):
        source = stages.create(analysis_id, stage_type, owner_user_id=owner)
        stages.start(source.stage_id, owner_user_id=owner)
        return stages.finish(source.stage_id, ResearchStageStatus.COMPLETED, owner_user_id=owner, result=result)

    queued = stages.create(analysis_a.analysis_id, ResearchStageType.INVENTORY, owner_user_id="owner-a")
    with pytest.raises(ValueError, match="report_source_stage_not_completed"):
        stages.create(analysis_a.analysis_id, ResearchStageType.REPORT_PREPARATION, owner_user_id="owner-a", input_payload={"source_stage_ids": [queued.stage_id]})

    same_analysis_other_owner = complete(analysis_a.analysis_id, ResearchStageType.INVENTORY, "owner-b", {"kind": "LOCAL_INVENTORY"})
    with pytest.raises(ValueError, match="report_source_stage_boundary_mismatch"):
        stages.create(analysis_a.analysis_id, ResearchStageType.REPORT_PREPARATION, owner_user_id="owner-a", input_payload={"source_stage_ids": [same_analysis_other_owner.stage_id]})

    other_analysis = complete(analysis_b.analysis_id, ResearchStageType.INVENTORY, "owner-a", {"kind": "LOCAL_INVENTORY"})
    with pytest.raises(ValueError, match="report_source_stage_boundary_mismatch"):
        stages.create(analysis_a.analysis_id, ResearchStageType.REPORT_PREPARATION, owner_user_id="owner-a", input_payload={"source_stage_ids": [other_analysis.stage_id]})

    unsupported = complete(analysis_a.analysis_id, ResearchStageType.PROGRAM_ANALYSIS, "owner-a", {"kind": "PROGRAM_ANALYSIS"})
    with pytest.raises(ValueError, match="report_source_stage_type_mismatch"):
        stages.create(analysis_a.analysis_id, ResearchStageType.REPORT_PREPARATION, owner_user_id="owner-a", input_payload={"source_stage_ids": [unsupported.stage_id]})

    invalid_result_kind = complete(analysis_a.analysis_id, ResearchStageType.INVENTORY, "owner-a", {"kind": "FOREIGN_RESULT"})
    with pytest.raises(ValueError, match="report_source_result_kind_mismatch"):
        stages.create(analysis_a.analysis_id, ResearchStageType.REPORT_PREPARATION, owner_user_id="owner-a", input_payload={"source_stage_ids": [invalid_result_kind.stage_id]})
    store.close()


def test_report_lineage_survives_database_reopen_and_rejects_result_digest_tampering(tmp_path: Path) -> None:
    from private_agent.bugbounty.research import ResearchStageRepository, ResearchStageStatus, ResearchStageType

    path = tmp_path / "report-lineage-reopen.sqlite3"
    store = Store(path)
    analysis = ProgramAnalysisRepository(store).save(analyze_program("https://example.com/policy", json.dumps(scope_payload())), owner_user_id="owner-a")
    stages = ResearchStageRepository(store)
    source = stages.create(analysis.analysis_id, ResearchStageType.INVENTORY, owner_user_id="owner-a")
    stages.start(source.stage_id, owner_user_id="owner-a")
    source = stages.finish(source.stage_id, ResearchStageStatus.COMPLETED, owner_user_id="owner-a", result={"kind": "LOCAL_INVENTORY", "endpoints": ["GET /orders/{id}"]})
    report = stages.create(analysis.analysis_id, ResearchStageType.REPORT_PREPARATION, owner_user_id="owner-a", input_payload={"source_stage_ids": [source.stage_id]})
    stored = store.db.execute("SELECT input_json FROM bugbounty_research_stages WHERE stage_id = ?", (report.stage_id,)).fetchone()
    assert stored is not None
    persisted_input = json.loads(stored["input_json"])
    assert set(persisted_input) == {"source_stage_refs"}
    assert "artifacts" not in stored["input_json"]
    store.close()

    reopened = Store(path)
    reopened_stages = ResearchStageRepository(reopened)
    persisted_report = reopened_stages.get(report.stage_id, owner_user_id="owner-a")
    assert persisted_report is not None
    assert reopened_stages.resolve_report_artifacts(persisted_report)[0]["content"] == json.dumps({"kind": "LOCAL_INVENTORY", "endpoints": ["GET /orders/{id}"]}, sort_keys=True)
    tampered = json.dumps({"kind": "LOCAL_INVENTORY", "endpoints": ["GET /admin"]}, sort_keys=True)
    with reopened._db_lock, reopened.db:
        reopened.db.execute("UPDATE bugbounty_research_stages SET result_json = ? WHERE stage_id = ?", (tampered, source.stage_id))
    with pytest.raises(ValueError, match="report_source_result_digest_mismatch"):
        reopened_stages.resolve_report_artifacts(persisted_report)
    reopened.close()


def test_api_to_worker_local_research_pipeline_prepares_observations_hypotheses_verification_and_report(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    analysis_response = client.post(
        "/api/v1/bug-bounty/programs/analyze",
        json={"program_url": "https://example.com/policy", "policy_text": json.dumps(scope_payload())},
        headers={"X-CSRF-Token": csrf},
    )
    assert analysis_response.status_code == 200, analysis_response.text
    analysis_id = analysis_response.json()["analysis_id"]
    evidence = {
        "artifacts": [
            {
                "name": "openapi.json",
                "content_type": "application/json",
                "content": json.dumps({"openapi": "3.0.0", "paths": {"/orders/{id}": {"get": {}}, "/admin/users": {"get": {}}}}),
            }
        ]
    }
    worker = BackgroundWorker(service.orchestrator, worker_id="api-local-pipeline-worker")
    stage_ids: dict[str, str] = {}

    def create_and_run(stage_type: str, input_payload: dict) -> dict:
        created = client.post(
            f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
            json={"stage_type": stage_type, "budget": {"max_items": 20, "timeout_seconds": 10}, "input_payload": input_payload},
            headers={"X-CSRF-Token": csrf},
        )
        assert created.status_code == 201, created.text
        stage_ids[stage_type] = created.json()["stage_id"]
        assert worker.run_once()["status"] == "STAGE_COMPLETED"
        fetched = client.get(f"/api/v1/bug-bounty/research-stages/{created.json()['stage_id']}")
        assert fetched.status_code == 200, fetched.text
        return fetched.json()["result"]

    passive = create_and_run("PASSIVE_RESEARCH", evidence)
    hypotheses = create_and_run("HYPOTHESIS_GENERATION", evidence)
    fixture = {"fixture_type": "local_verification", "cases": [{"case_id": "owner-read", "expected": {"status": "allowed"}, "observed": {"status": "allowed"}}]}
    verification = create_and_run("VERIFICATION", {"artifacts": [{"name": "fixture.json", "content_type": "application/json", "content": json.dumps(fixture)}]})
    report = create_and_run(
        "REPORT_PREPARATION",
        {"source_stage_ids": [stage_ids["PASSIVE_RESEARCH"], stage_ids["HYPOTHESIS_GENERATION"], stage_ids["VERIFICATION"]]},
    )
    assert passive["network_accessed"] is False
    assert hypotheses["hypotheses"]
    assert verification["finding_confirmation_allowed"] is False
    assert report["kind"] == "LOCAL_REPORT"
    assert report["report_status"] == "LOCAL_ONLY_UNVERIFIED"
    report_row = store.db.execute("SELECT input_json FROM bugbounty_research_stages WHERE stage_id = ?", (stage_ids["REPORT_PREPARATION"],)).fetchone()
    assert report_row is not None
    stored_lineage = json.loads(report_row["input_json"])["source_stage_refs"]
    assert {item["stage_id"] for item in stored_lineage} == {stage_ids["PASSIVE_RESEARCH"], stage_ids["HYPOTHESIS_GENERATION"], stage_ids["VERIFICATION"]}
    assert all(len(item["result_sha256"]) == 64 for item in stored_lineage)
    report_response = client.get(f"/api/v1/bug-bounty/research-stages/{stage_ids['REPORT_PREPARATION']}").json()
    assert report_response["source_stage_refs"] == stored_lineage
    assert "input_payload" not in report_response
    assert client.get(f"/api/v1/bug-bounty/programs/{analysis_id}/stages").status_code == 200
    store.close()


def test_research_stage_cancellation_fences_late_finish_and_snapshots_are_copied(tmp_path: Path) -> None:
    from private_agent.bugbounty.research import ResearchStageRepository, ResearchStageStatus, ResearchStageType

    store = Store(tmp_path / "research-stage-fence.sqlite3")
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())), owner_user_id="user-a"
    )
    stages = ResearchStageRepository(store)
    queued = stages.create(analysis.analysis_id, ResearchStageType.INVENTORY, owner_user_id="user-a")
    running = stages.start(queued.stage_id, owner_user_id="user-a")
    cancelled = stages.cancel(running.stage_id, owner_user_id="user-a")
    assert cancelled.stage_status == ResearchStageStatus.CANCELLED.value
    with pytest.raises(ValueError, match="not_running"):
        stages.finish(running.stage_id, ResearchStageStatus.COMPLETED, owner_user_id="user-a", result={"items": ["late"]})
    snapshot = stages.get(queued.stage_id, owner_user_id="user-a")
    assert snapshot is not None
    rendered = snapshot.to_dict()
    rendered["partial_result"]["mutated"] = True
    reread = stages.get(queued.stage_id, owner_user_id="user-a")
    assert reread is not None and "mutated" not in reread.partial_result
    store.close()


def test_finding_review_contract_is_fail_closed_for_unvalidated_confirmation_and_duplicate_self() -> None:
    from private_agent.bugbounty.reviews import FindingReview, FindingReviewDecision

    with pytest.raises(ValueError, match="finding_review_confirm_requires_validated_evidence"):
        FindingReview.create("task-1", "finding-1", "owner-1", FindingReviewDecision.CONFIRM)
    with pytest.raises(ValueError, match="finding_review_duplicate_self"):
        FindingReview.create("task-1", "finding-1", "owner-1", FindingReviewDecision.DUPLICATE, duplicate_of="finding-1")
    inconclusive = FindingReview.create("task-1", "finding-1", "owner-1", FindingReviewDecision.INCONCLUSIVE)
    assert inconclusive.evidence_validated is False
