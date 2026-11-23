from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from private_agent.api import AgentAPIService, create_app
from private_agent.auth import AuthManager, ROLE_OPERATOR
from private_agent.bugbounty.models import public_reference
from private_agent.storage import Store
from tests.test_phase12_api import APITestTool, make_service
from tests.test_phase_auth_rbac import OWNER_EMAIL, OWNER_PASSWORD, _login


def scope_payload() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "program_id": "handoff-local-program",
        "authorization_reference": "policy-reference-that-must-not-be-persisted-raw",
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "assets": [{"asset_id": "fixture-api", "host": "fixture.local", "path_prefix": "/api/v1", "methods": ["GET"], "active_allowed": True}],
        "excluded_hosts": ["outside.fixture.local"],
        "permitted_identities": ["alice"],
        "permitted_resources": ["order-alice"],
        "permitted_methods": ["GET"],
        "prohibited_methods": ["DELETE", "PATCH", "PUT", "POST"],
        "active_testing": True,
    }


def create_analysis(client: TestClient, csrf: str, *, source_type: str = "user_provided_policy", include_policy: bool = True) -> dict:
    response = client.post(
        "/api/v1/bug-bounty/programs/analyze",
        json={
            "program_url": "https://example.test/program-policy",
            "policy_text": json.dumps(scope_payload()) if include_policy else "",
            "source_type": source_type,
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200, response.text
    return response.json()


def create_owner_client(tmp_path: Path) -> tuple[AgentAPIService, Store, TestClient, str]:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    return service, store, client, csrf


def handoff_body(analysis: dict, *, reference: str = "written-authorization-ref-2026", **extra: object) -> dict:
    return {
        "proposal_digest": analysis["proposal_digest"],
        "authorization_reference": reference,
        "authorization_attested": True,
        **extra,
    }


def test_handoff_uses_only_current_server_proposal_and_persists_redacted_provenance(tmp_path: Path) -> None:
    service, store, client, csrf = create_owner_client(tmp_path)
    analysis = create_analysis(client, csrf)
    proposal = analysis["scope_proposal"]
    assert analysis["status"] == "READY_FOR_SCOPE_REVIEW"
    assert len(analysis["proposal_digest"]) == 64
    assert proposal["authorization_reference"].startswith("sha256:")
    assert "policy-reference-that-must-not-be-persisted-raw" not in json.dumps(analysis)

    reference = "written-authorization-ref-2026"
    headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "analysis-handoff-1"}
    response = client.post(
        f"/api/v1/bug-bounty/programs/{analysis['analysis_id']}/task",
        json=handoff_body(analysis, reference=reference),
        headers=headers,
    )
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["conversation_id"] == created["task_id"]

    state = service._state(created["task_id"])
    context = state.context
    request = context["bugbounty_request"]
    submitted_scope = request["scope"]
    assert request["scanner"] == "deterministic"
    assert request["target"] == ""
    assert submitted_scope["program_id"] == proposal["program_id"]
    assert submitted_scope["assets"] == proposal["assets"]
    assert submitted_scope["excluded_hosts"] == proposal["excluded_hosts"]
    assert submitted_scope["scope_reference"] == public_reference(reference)
    assert "authorization_reference" not in submitted_scope
    assert "written-authorization-ref-2026" not in json.dumps(context)
    provenance = context["program_analysis_provenance"]
    owner_user_id = AuthManager(store).list_users()[0].user_id
    assert provenance == {
        "analysis_id": analysis["analysis_id"],
        "analysis_status": "READY_FOR_SCOPE_REVIEW",
        "source_content_sha256": analysis["source"]["content_sha256"],
        "proposal_digest": analysis["proposal_digest"],
        "scope_reference_digest": public_reference(reference),
        "scope_attestation": True,
        "scope_attested_by_user_id": owner_user_id,
        "scope_attested_at": provenance["scope_attested_at"],
        "execution_mode": "deterministic_local",
    }
    assert provenance["scope_attested_by_user_id"] == owner_user_id
    assert provenance["scope_attested_at"]

    replay = client.post(
        f"/api/v1/bug-bounty/programs/{analysis['analysis_id']}/task",
        json=handoff_body(analysis, reference=reference),
        headers=headers,
    )
    assert replay.status_code == 201, replay.text
    assert replay.json()["task_id"] == created["task_id"]
    assert replay.json()["idempotent_replay"] is True

    detail = client.get(f"/api/v1/bug-bounty/tasks/{created['task_id']}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["analysis_provenance"]["analysis_id"] == analysis["analysis_id"]
    assert detail.json()["analysis_provenance"]["source_content_sha256"] == analysis["source"]["content_sha256"]
    assert detail.json()["analysis_provenance"]["proposal_digest"] == analysis["proposal_digest"]
    assert detail.json()["analysis_provenance"]["scope_reference_digest"] == public_reference(reference)
    assert reference not in detail.text
    store.close()


def test_handoff_rejects_stale_unattested_or_client_modified_proposals_without_queue_side_effects(tmp_path: Path) -> None:
    _service, store, client, csrf = create_owner_client(tmp_path)
    analysis = create_analysis(client, csrf)
    path = f"/api/v1/bug-bounty/programs/{analysis['analysis_id']}/task"
    headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "analysis-handoff-invalid"}

    stale = client.post(path, json={**handoff_body(analysis), "proposal_digest": "0" * 64}, headers=headers)
    assert stale.status_code == 409
    assert stale.json()["error_code"] == "program_analysis_stale"

    unattested = client.post(path, json={**handoff_body(analysis), "authorization_attested": False}, headers=headers)
    assert unattested.status_code == 422
    assert unattested.json()["error_code"] == "authorization_attestation_required"

    modified = client.post(
        path,
        json={**handoff_body(analysis), "scope": {"program_id": "client-injected"}},
        headers=headers,
    )
    assert modified.status_code == 422
    assert client.get("/api/v1/tasks").json()["items"] == []

    public_analysis = create_analysis(client, csrf, source_type="hackerone_program")
    assert public_analysis["status"] == "READY_FOR_SCOPE_REVIEW"
    assert public_analysis["scope_proposal"] is not None
    public_response = client.post(
        f"/api/v1/bug-bounty/programs/{public_analysis['analysis_id']}/task",
        json=handoff_body(public_analysis),
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": "analysis-handoff-public"},
    )
    assert public_response.status_code == 409
    assert public_response.json()["error_code"] == "public_policy_is_not_authorization"
    assert client.get("/api/v1/tasks").json()["items"] == []
    store.close()


def test_handoff_rejects_review_required_analyses_and_foreign_operator_ids(tmp_path: Path) -> None:
    service, store, owner_client, owner_csrf = create_owner_client(tmp_path)
    incomplete = create_analysis(owner_client, owner_csrf, include_policy=False)
    assert incomplete["status"] == "REVIEW_REQUIRED"
    blocked = owner_client.post(
        f"/api/v1/bug-bounty/programs/{incomplete['analysis_id']}/task",
        json={"proposal_digest": "0" * 64, "authorization_reference": "written-ref", "authorization_attested": True},
        headers={"X-CSRF-Token": owner_csrf, "Idempotency-Key": "analysis-handoff-review-required"},
    )
    assert blocked.status_code == 409
    assert blocked.json()["error_code"] == "program_analysis_not_eligible"

    owner = AuthManager(store).list_users()[0]
    AuthManager(store).create_user("analysis-operator@example.test", "operator-password-for-tests-123", ROLE_OPERATOR, actor_user_id=owner.user_id)
    operator = TestClient(create_app(service=service), raise_server_exceptions=False)
    operator_csrf = _login(operator, "analysis-operator@example.test", "operator-password-for-tests-123")
    owned_by_owner = create_analysis(owner_client, owner_csrf)
    owned_by_operator = create_analysis(operator, operator_csrf)
    denied_self_handoff = operator.post(
        f"/api/v1/bug-bounty/programs/{owned_by_operator['analysis_id']}/task",
        json=handoff_body(owned_by_operator),
        headers={"X-CSRF-Token": operator_csrf, "Idempotency-Key": "operator-self-handoff"},
    )
    assert denied_self_handoff.status_code == 403
    assert denied_self_handoff.json()["error_code"] == "forbidden"

    denied = operator.post(
        f"/api/v1/bug-bounty/programs/{owned_by_owner['analysis_id']}/task",
        json=handoff_body(owned_by_owner),
        headers={"X-CSRF-Token": operator_csrf, "Idempotency-Key": "foreign-analysis-handoff"},
    )
    assert denied.status_code == 403
    assert denied.json()["error_code"] == "forbidden"
    assert owner_client.get("/api/v1/tasks").json()["items"] == []
    store.close()
