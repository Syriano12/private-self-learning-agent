from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from private_agent.api import build_api_service, create_app
from private_agent.auth import AuthManager, ROLE_OPERATOR
from private_agent.bugbounty.learning import BugBountyLearningService
from private_agent.bugbounty.reviews import FindingReview, FindingReviewRepository
from private_agent.bugbounty.workflow import BugBountyAgent, BugBountyRepository
from private_agent.storage import Store
from private_agent.tools.bugbounty import BugBountyAnalysisTool
from private_agent.worker import BackgroundWorker
from tests.test_phase_auth_rbac import OWNER_EMAIL, OWNER_PASSWORD, _login


OWNER_GOAL = "Bug Bounty analysis for authorized program local-learning-fixture"


def _scope() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "program_id": "local-learning-fixture",
        "authorization_reference": "owner-authorized-synthetic-local-fixture",
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


def _create_and_run_local_fixture(client: TestClient, worker: BackgroundWorker, csrf: str) -> tuple[str, dict]:
    created = client.post(
        "/api/v1/bug-bounty/tasks",
        json={"scope": _scope(), "fixture": "idor_horizontal"},
        headers={"X-CSRF-Token": csrf},
    )
    assert created.status_code == 201, created.text
    task_id = created.json()["task_id"]
    result = worker.run_once()
    assert result["status"] == "COMPLETED", result
    detail = client.get(f"/api/v1/bug-bounty/tasks/{task_id}")
    assert detail.status_code == 200, detail.text
    return task_id, detail.json()


def test_owner_review_memory_changes_future_bugbounty_plan_after_sqlite_reopen_without_promotion(
    tmp_path: Path,
) -> None:
    database = tmp_path / "bugbounty-owner-learning.sqlite3"
    service = build_api_service(token=None, database_path=str(database))
    AuthManager(service.store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    worker = BackgroundWorker(service.orchestrator, worker_id="bugbounty-learning-source")
    try:
        with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
            csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
            source_task_id, source = _create_and_run_local_fixture(client, worker, csrf)
            source_findings = source["run"]["findings"]
            assert len(source_findings) == 1
            finding = source_findings[0]
            assert finding["vulnerability_class"] == "IDOR/BOLA"
            assert finding["state"] == "REPORT_READY"
            assert not any(step.get("action") == "prioritize_owner_review_before_confirmation" for step in source["run"]["plan"])

            reviewed = client.post(
                f"/api/v1/bug-bounty/tasks/{source_task_id}/findings/{finding['finding_id']}/review",
                json={"decision": "CONFIRM", "note": "Owner confirmed the validated synthetic local fixture."},
                headers={"X-CSRF-Token": csrf},
            )
            assert reviewed.status_code == 200, reviewed.text
            assert reviewed.json()["evidence_validated"] is True
            assert reviewed.json()["actor_user_id"]
            assert BugBountyLearningService(service.store).planning_advisories(
                task_id=source_task_id,
                goal=OWNER_GOAL,
            )
    finally:
        service.store.close()

    # The next task runs through a newly constructed service after reopening the same SQLite file.
    reopened_service = build_api_service(token=None, database_path=str(database))
    try:
        reopened_worker = BackgroundWorker(reopened_service.orchestrator, worker_id="bugbounty-learning-future")
        with TestClient(create_app(service=reopened_service), raise_server_exceptions=False) as client:
            csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
            target_task_id, target = _create_and_run_local_fixture(client, reopened_worker, csrf)

            advice = [
                step for step in target["run"]["plan"]
                if step.get("action") == "prioritize_owner_review_before_confirmation"
            ]
            assert len(advice) == 1
            assert advice[0]["vulnerability_class"] == "IDOR/BOLA"
            assert advice[0]["active"] is False
            assert advice[0]["external_access"] is False
            assert advice[0]["non_executable"] is True
            assert advice[0]["non_authoritative"] is True
            assert advice[0]["provenance"]["review_id"] == reviewed.json()["review_id"]
            assert advice[0]["provenance"]["evidence_sha256"] == reviewed.json()["evidence_sha256"]

            # Memory changed only an advisory plan step. The normal local Evidence Gate,
            # explicit scope, and finding lifecycle still determine the outcome.
            for scope_key in ("program_id", "assets", "permitted_methods", "active_testing"):
                assert target["run"]["scope"][scope_key] == source["run"]["scope"][scope_key]
            assert target["run"]["findings"][0]["state"] == "REPORT_READY"
            assert target["run"]["findings"][0]["validation"]["status"] == "VALIDATED"
            target_reviews = client.get(f"/api/v1/bug-bounty/tasks/{target_task_id}/reviews")
            assert target_reviews.status_code == 200, target_reviews.text
            assert target_reviews.json() == []
            assert target_task_id != source_task_id
            saved_run = BugBountyRepository(reopened_service.store).get(target["run"]["run_id"])
            assert saved_run is not None
            assert any(
                step.get("action") == "prioritize_owner_review_before_confirmation"
                for step in saved_run["plan"]
            )

            # A fresh contrary owner review suppresses the positive hint for later tasks.
            target_finding_id = target["run"]["findings"][0]["finding_id"]
            refuted = client.post(
                f"/api/v1/bug-bounty/tasks/{target_task_id}/findings/{target_finding_id}/review",
                json={"decision": "REFUTE", "note": "Negative control: the owner rejects this candidate."},
                headers={"X-CSRF-Token": csrf},
            )
            assert refuted.status_code == 200, refuted.text
            assert not BugBountyLearningService(reopened_service.store).planning_advisories(
                task_id=target_task_id,
                goal=OWNER_GOAL,
            )
    finally:
        reopened_service.store.close()


def test_bug_bounty_advisory_is_owner_bound_fresh_and_conflict_sensitive(tmp_path: Path) -> None:
    database = tmp_path / "bugbounty-learning-gates.sqlite3"
    store = Store(database)
    auth = AuthManager(store)
    owner = auth.initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    other = auth.create_user(
        "bugbounty-learning-other@example.test",
        "bugbounty-learning-other-password-123",
        ROLE_OPERATOR,
        actor_user_id=owner.user_id,
    )
    source_task_id = store.create_queued_task(OWNER_GOAL, owner_user_id=owner.user_id)["task_id"]
    run = BugBountyAgent(store=store).run(_scope(), fixture_variant="idor_horizontal", persist=True)
    finding = run.findings[0]
    positive = FindingReview.create(
        source_task_id,
        finding.finding_id,
        owner.user_id,
        "CONFIRM",
        evidence_sha256=finding.validated_evidence_sha256,
        evidence_validated=True,
        note="Verified local evidence for the test fixture.",
    )
    FindingReviewRepository(store).save(positive)
    BugBountyLearningService(store).apply_review(positive, finding)
    store.close()

    reopened = Store(database)
    try:
        owner_task_id = reopened.create_queued_task(OWNER_GOAL, owner_user_id=owner.user_id)["task_id"]
        other_task_id = reopened.create_queued_task(OWNER_GOAL, owner_user_id=other.user_id)["task_id"]
        unowned_task_id = reopened.create_queued_task(OWNER_GOAL)["task_id"]
        learning = BugBountyLearningService(reopened)
        assert len(learning.planning_advisories(task_id=owner_task_id, goal=OWNER_GOAL)) == 1
        assert learning.planning_advisories(task_id=other_task_id, goal=OWNER_GOAL) == []
        assert learning.planning_advisories(task_id=unowned_task_id, goal=OWNER_GOAL) == []

        # Stale source reviews fail closed even if the cached strategy remains active.
        stale_at = (datetime.now(timezone.utc) - timedelta(days=181)).isoformat()
        with reopened._db_lock, reopened.db:
            reopened.db.execute(
                "UPDATE bugbounty_finding_reviews SET created_at = ? WHERE review_id = ?",
                (stale_at, positive.review_id),
            )
        assert learning.planning_advisories(task_id=owner_task_id, goal=OWNER_GOAL) == []

        # Restore a fresh positive review, then add a current same-class refutation.
        with reopened._db_lock, reopened.db:
            reopened.db.execute(
                "UPDATE bugbounty_finding_reviews SET created_at = ? WHERE review_id = ?",
                (datetime.now(timezone.utc).isoformat(), positive.review_id),
            )
        negative = FindingReview.create(
            source_task_id,
            finding.finding_id,
            owner.user_id,
            "REFUTE",
            note="Contrary owner review for the same vulnerability class.",
        )
        FindingReviewRepository(reopened).save(negative)
        BugBountyLearningService(reopened).apply_review(negative, finding)
        assert learning.planning_advisories(task_id=owner_task_id, goal=OWNER_GOAL) == []
    finally:
        reopened.close()


def test_untrusted_advisory_cannot_be_injected_as_a_tool_input_or_change_scope() -> None:
    untrusted = {
        "strategy_id": "strategy-client-forged",
        "vulnerability_class": "IDOR/BOLA",
        "review_id": "REVIEW-client-forged",
        "finding_id": "F-client-forged",
        "evidence_sha256": "a" * 64,
        "confidence": 0.99,
        "requirement": "prioritize_owner_review_before_confirmation",
        "non_authoritative": False,
        "active": True,
        "external_access": True,
    }
    scope = _scope()
    scope["active_testing"] = False
    scope["assets"][0]["active_allowed"] = False
    run = BugBountyAgent().run(
        scope,
        learned_review_advisories=[untrusted],
        persist=False,
    )

    assert not any(step.get("action") == "prioritize_owner_review_before_confirmation" for step in run.plan)
    assert run.findings == []
    assert run.scope.active_testing is False
    assert run.scope.assets[0].active_allowed is False
    assert "learned_review_advisories" not in BugBountyAnalysisTool(None).input_schema()["properties"]
