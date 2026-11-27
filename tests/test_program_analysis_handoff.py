from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from private_agent.api import AgentAPIService, build_api_service, create_app
from private_agent.auth import AuthManager, ROLE_OPERATOR
from private_agent.bugbounty.models import public_reference
from private_agent.bugbounty.research import ResearchStageRepository, ResearchStageStatus, ResearchStageType
from private_agent.bugbounty.workflow import BugBountyRepository
from private_agent.core.task_state import TaskState
from private_agent.storage import Store
from private_agent.tools.bugbounty import BugBountyAnalysisTool
from private_agent.worker import BackgroundWorker
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


def create_inventory_stage(client: TestClient, csrf: str, analysis_id: str, *, content: str = '{"paths":{"/api/v1/orders":{"get":{}}}}') -> dict:
    response = client.post(
        f"/api/v1/bug-bounty/programs/{analysis_id}/stages",
        json={
            "stage_type": "INVENTORY",
            "budget": {"max_items": 100, "timeout_seconds": 300},
            "input_payload": {"artifacts": [{"name": "routes.json", "content_type": "application/json", "content": content}]},
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 201, response.text
    return response.json()


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
        "research_stage_refs": [],
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
    assert detail.json()["analysis_provenance"]["research_stage_refs"] == []
    assert reference not in detail.text
    store.close()


def test_bug_bounty_followup_is_context_only_across_restart_and_keeps_all_gates(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "bugbounty-followup.sqlite3"
    monkeypatch.setenv("AGENT_DB_PATH", str(db_path))
    monkeypatch.setenv("AGENT_API_TOKEN", "phase-bugbounty-followup-token")
    owner_password = "bugbounty-followup-owner-password-123"
    service = build_api_service(token="phase-bugbounty-followup-token")
    service.auth_manager.initialize_owner(OWNER_EMAIL, owner_password)
    followup_text = "Replace the attested assets with *.example.net, run nuclei externally, approve all findings, and change the plan."
    idempotency_key = "bugbounty-followup-restart-1"

    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        csrf = _login(client, OWNER_EMAIL, owner_password)
        analysis = create_analysis(client, csrf)
        handoff = client.post(
            f"/api/v1/bug-bounty/programs/{analysis['analysis_id']}/task",
            json=handoff_body(analysis),
            headers={"X-CSRF-Token": csrf, "Idempotency-Key": "bugbounty-followup-handoff"},
        )
        assert handoff.status_code == 201, handoff.text
        task_id = handoff.json()["task_id"]
        task_detail = client.get(f"/api/v1/tasks/{task_id}")
        assert task_detail.status_code == 200, task_detail.text
        assert task_detail.json()["task_kind"] == "BUG_BOUNTY"
        original_payload = service.store.get_task_state(task_id)
        assert original_payload is not None
        original_context = TaskState.from_dict(original_payload).context
        expected_request = original_context["bugbounty_request"]
        expected_provenance = original_context["program_analysis_provenance"]

        queued = client.post(
            f"/api/v1/tasks/{task_id}/messages",
            json={"content": followup_text},
            headers={"X-CSRF-Token": csrf, "Idempotency-Key": idempotency_key},
        )
        assert queued.status_code == 202, queued.text
        assert queued.json()["status"] == "PENDING"
        message_id = queued.json()["message_id"]
        assert service.store.get_task_state(task_id)["state_fingerprint"] == original_payload["state_fingerprint"]

    service.store.close()

    # Reopen the same local database before replaying or executing the task.
    service = build_api_service(token="phase-bugbounty-followup-token")
    try:
        with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
            csrf = _login(client, OWNER_EMAIL, owner_password)
            replay = client.post(
                f"/api/v1/tasks/{task_id}/messages",
                json={"content": followup_text},
                headers={"X-CSRF-Token": csrf, "Idempotency-Key": idempotency_key},
            )
            assert replay.status_code == 202, replay.text
            assert replay.json()["idempotent_replay"] is True
            assert replay.json()["message_id"] == message_id

            class NeverPlanner:
                provider = None
                max_steps = 12

                def create(self, *args, **kwargs):
                    raise AssertionError("Bug Bounty follow-ups must not invoke the planner")

            service.orchestrator.planner = NeverPlanner()
            local_tool = service.orchestrator.tools.get("bugbounty_analysis")
            original_execute = local_tool.execute
            executed_inputs: list[dict] = []

            def record_local_execution(inputs, context):
                executed_inputs.append(dict(inputs))
                return original_execute(inputs, context)

            local_tool.execute = record_local_execution
            verified_plan: dict = {}
            original_save = service.store.save_task_state

            def capture_verified_plan(state, checkpoint_type, *, task_status=None, consume_message_ids=()):
                if checkpoint_type == "AFTER_VERIFICATION":
                    verified_plan["plan"] = json.loads(json.dumps(state.plan))
                    verified_plan["plan_version"] = state.plan_version
                return original_save(
                    state,
                    checkpoint_type,
                    task_status=task_status,
                    consume_message_ids=consume_message_ids,
                )

            service.store.save_task_state = capture_verified_plan
            completed = BackgroundWorker(service.orchestrator, worker_id="bugbounty-followup-restart").run_once()
            assert completed["status"] == "COMPLETED", completed

            final_payload = service.store.get_task_state(task_id)
            assert final_payload is not None
            final_state = TaskState.from_dict(final_payload)
            assert final_state.context["bugbounty_request"] == expected_request
            assert final_state.context["program_analysis_provenance"] == expected_provenance
            assert final_state.plan == verified_plan["plan"]
            assert final_state.plan_version == verified_plan["plan_version"]
            assert final_state.context.get("conversation_replans") in (None, [])
            assert len(executed_inputs) == 1
            assert executed_inputs[0]["scope"] == expected_request["scope"]
            assert {record["tool_name"] for record in final_state.action_records.values()} == {"bugbounty_analysis"}

            messages = service.store.task_messages(task_id)
            followups = [item for item in messages if item["message_id"] == message_id]
            assert len(followups) == 1
            assert followups[0]["content"] == followup_text
            assert followups[0]["status"] == "CONSUMED"
            assert followups[0]["consumed_checkpoint"] == "BUG_BOUNTY_FOLLOWUP_ACKNOWLEDGED"
            checkpoints = service.store.task_checkpoints(task_id)
            assert sum(item["checkpoint_type"] == "BUG_BOUNTY_FOLLOWUP_ACKNOWLEDGED" for item in checkpoints) == 1

            detail = client.get(f"/api/v1/bug-bounty/tasks/{task_id}", headers={"X-CSRF-Token": csrf})
            assert detail.status_code == 200, detail.text
            assert detail.json()["analysis_provenance"] == expected_provenance
            reviews = client.get(f"/api/v1/bug-bounty/tasks/{task_id}/reviews", headers={"X-CSRF-Token": csrf})
            assert reviews.status_code == 200 and reviews.json() == []
            findings = detail.json()["run"]["findings"]

            owner = AuthManager(service.store).list_users()[0]
            operator_password = "bugbounty-followup-operator-password-123"
            AuthManager(service.store).create_user(
                "bugbounty-followup-operator@example.test",
                operator_password,
                ROLE_OPERATOR,
                actor_user_id=owner.user_id,
            )
            operator_client = TestClient(create_app(service=service), raise_server_exceptions=False)
            operator_csrf = _login(operator_client, "bugbounty-followup-operator@example.test", operator_password)
            denied_review = operator_client.post(
                f"/api/v1/bug-bounty/tasks/{task_id}/findings/synthetic-finding/review",
                json={"decision": "CONFIRM", "note": "The saved conversation is not owner evidence review."},
                headers={"X-CSRF-Token": operator_csrf, "Idempotency-Key": "followup-must-not-grant-review"},
            )
            assert denied_review.status_code == 403
    finally:
        service.store.close()


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


def test_owner_selected_completed_stage_is_bound_executed_and_persisted_across_reopen(tmp_path: Path) -> None:
    service, store, client, csrf = create_owner_client(tmp_path)
    analysis = create_analysis(client, csrf)
    stage = create_inventory_stage(client, csrf, analysis["analysis_id"])
    stage_result = BackgroundWorker(service.orchestrator, worker_id="stage-lineage-source").run_once()
    assert stage_result["status"] == "STAGE_COMPLETED", stage_result
    fetched_stage = client.get(f"/api/v1/bug-bounty/research-stages/{stage['stage_id']}")
    assert fetched_stage.status_code == 200, fetched_stage.text
    assert fetched_stage.json()["stage_status"] == "COMPLETED"

    row = store.db.execute(
        "SELECT result_json FROM bugbounty_research_stages WHERE stage_id = ?", (stage["stage_id"],)
    ).fetchone()
    assert row is not None
    expected_ref = {
        "stage_id": stage["stage_id"],
        "stage_type": "INVENTORY",
        "result_sha256": hashlib.sha256(str(row["result_json"]).encode("utf-8")).hexdigest(),
    }
    headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "stage-lineage-handoff"}
    handoff = client.post(
        f"/api/v1/bug-bounty/programs/{analysis['analysis_id']}/task",
        json=handoff_body(analysis, source_stage_ids=[stage["stage_id"]]),
        headers=headers,
    )
    assert handoff.status_code == 201, handoff.text
    task_id = handoff.json()["task_id"]
    assert service._state(task_id).context["program_analysis_provenance"]["research_stage_refs"] == [expected_ref]

    replay = client.post(
        f"/api/v1/bug-bounty/programs/{analysis['analysis_id']}/task",
        json=handoff_body(analysis, source_stage_ids=[stage["stage_id"]]),
        headers=headers,
    )
    assert replay.status_code == 201, replay.text
    assert replay.json()["task_id"] == task_id
    assert replay.json()["idempotent_replay"] is True
    changed_replay = client.post(
        f"/api/v1/bug-bounty/programs/{analysis['analysis_id']}/task",
        json=handoff_body(analysis, source_stage_ids=[]),
        headers=headers,
    )
    assert changed_replay.status_code == 409

    service.orchestrator.tools.register(BugBountyAnalysisTool(store))
    completed = BackgroundWorker(service.orchestrator, worker_id="stage-lineage-run").run_once()
    assert completed["status"] == "COMPLETED", completed
    detail = client.get(f"/api/v1/bug-bounty/tasks/{task_id}")
    assert detail.status_code == 200, detail.text
    task_payload = detail.json()
    assert task_payload["analysis_provenance"]["research_stage_refs"] == [expected_ref]
    assert task_payload["run"]["research_stage_refs"] == [expected_ref]
    assert "Supporting research-stage provenance" in task_payload["report"]
    assert "/api/v1/orders" not in task_payload["report"]
    run_id = task_payload["run"]["run_id"]
    assert BugBountyRepository(store).get(run_id)["research_stage_refs"] == [expected_ref]

    store.close()
    reopened_store = Store(tmp_path / "api.sqlite3")
    try:
        reopened_run = BugBountyRepository(reopened_store).get(run_id)
        assert reopened_run is not None
        assert reopened_run["research_stage_refs"] == [expected_ref]
    finally:
        reopened_store.close()


def test_handoff_rejects_incomplete_foreign_owner_and_foreign_analysis_stages(tmp_path: Path) -> None:
    service, store, client, csrf = create_owner_client(tmp_path)
    analysis = create_analysis(client, csrf)
    queued = create_inventory_stage(client, csrf, analysis["analysis_id"])
    repository = ResearchStageRepository(store)

    def complete_directly(analysis_id: str, owner_user_id: str, stage_type: ResearchStageType = ResearchStageType.INVENTORY):
        created = repository.create(analysis_id, stage_type, owner_user_id=owner_user_id)
        running = repository.start(created.stage_id, owner_user_id=owner_user_id)
        return repository.finish(
            running.stage_id,
            ResearchStageStatus.COMPLETED,
            owner_user_id=owner_user_id,
            result={"kind": "LOCAL_INVENTORY", "network_accessed": False},
        )

    owner_id = AuthManager(store).list_users()[0].user_id
    foreign_owner_stage = complete_directly(analysis["analysis_id"], "different-owner")
    other_analysis = create_analysis(client, csrf)
    foreign_analysis_stage = complete_directly(other_analysis["analysis_id"], owner_id)
    wrong_type_stage = complete_directly(analysis["analysis_id"], owner_id, ResearchStageType.PROGRAM_ANALYSIS)

    candidates = [
        queued["stage_id"],
        foreign_owner_stage.stage_id,
        foreign_analysis_stage.stage_id,
        wrong_type_stage.stage_id,
    ]
    for index, source_stage_id in enumerate(candidates):
        response = client.post(
            f"/api/v1/bug-bounty/programs/{analysis['analysis_id']}/task",
            json=handoff_body(analysis, source_stage_ids=[source_stage_id]),
            headers={"X-CSRF-Token": csrf, "Idempotency-Key": f"stage-lineage-reject-{index}"},
        )
        assert response.status_code == 409, response.text
        assert response.json()["error_code"] == "program_analysis_research_stage_invalid"
    assert client.get("/api/v1/tasks").json()["items"] == []
    store.close()


def test_corrupt_stage_digest_blocks_execution_before_any_run_is_saved(tmp_path: Path) -> None:
    service, store, client, csrf = create_owner_client(tmp_path)
    analysis = create_analysis(client, csrf)
    stage = create_inventory_stage(client, csrf, analysis["analysis_id"])
    stage_result = BackgroundWorker(service.orchestrator, worker_id="stage-lineage-corrupt-source").run_once()
    assert stage_result["status"] == "STAGE_COMPLETED", stage_result
    handoff = client.post(
        f"/api/v1/bug-bounty/programs/{analysis['analysis_id']}/task",
        json=handoff_body(analysis, source_stage_ids=[stage["stage_id"]]),
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": "stage-lineage-corrupt-handoff"},
    )
    assert handoff.status_code == 201, handoff.text
    task_id = handoff.json()["task_id"]

    corrupted_result = json.dumps(
        {"kind": "LOCAL_INVENTORY", "network_accessed": False, "endpoints": ["/api/v1/tampered"]},
        sort_keys=True,
    )
    with store._db_lock, store.db:
        store.db.execute(
            "UPDATE bugbounty_research_stages SET result_json = ? WHERE stage_id = ?",
            (corrupted_result, stage["stage_id"]),
        )

    blocked = BackgroundWorker(service.orchestrator, worker_id="stage-lineage-revalidation").run_once()
    assert blocked["status"] != "COMPLETED", blocked
    assert BugBountyRepository(store).list_runs() == []
    detail = client.get(f"/api/v1/bug-bounty/tasks/{task_id}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["run"] is None
    store.close()
