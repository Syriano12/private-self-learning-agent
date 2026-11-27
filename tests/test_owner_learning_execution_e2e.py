from __future__ import annotations

import ast
import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from private_agent.api import AgentAPIService, build_api_service, create_app
from private_agent.auth import AuthManager, ROLE_OPERATOR
from private_agent.core.knowledge import KnowledgeClaim, KnowledgeConflict
from private_agent.core.observation import Observation
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import Planner
from private_agent.core.verification import VerificationResult, VerifierRegistry
from private_agent.security import SecurityController
from private_agent.storage import Store
from private_agent.tools.bug_investigation import BugInvestigationTool
from private_agent.tools.repository import RepositoryInspector, RepositoryTool
from private_agent.tools.research import ToolRegistry
from private_agent.worker import BackgroundWorker


ATTESTATION_KEY = "owner-learning-e2e-key-with-at-least-32-bytes"
OWNER_EMAIL = "owner-learning-e2e@example.test"
OWNER_PASSWORD = "owner-learning-e2e-password-123"
FUTURE_GOAL = (
    "Review the IDOR and BOLA authorization boundary in repository sample/orders, "
    "focusing on the tenant_id argument passed to load_order."
)


def _scope() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "program_id": "local-owner-learning-fixture",
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


def _wait_for_task(store: Store, task_id: str, timeout: float = 8.0) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = store.get_task_state(task_id)
        if state and state.get("status") in {"COMPLETED", "FAILED", "BLOCKED", "WAITING_APPROVAL"}:
            return state
        time.sleep(0.02)
    return store.get_task_state(task_id)


def _submit_goal_via_api(
    client: TestClient,
    store: Store,
    *,
    idempotency_key: str,
) -> tuple[str, dict[str, Any] | None]:
    headers = {
        "X-CSRF-Token": str(client.cookies.get("vulnora_csrf")),
        "Idempotency-Key": idempotency_key,
    }
    created = client.post("/api/v1/tasks", json={"goal": FUTURE_GOAL}, headers=headers)
    assert created.status_code == 201, created.text
    task_id = created.json()["task_id"]
    return task_id, _wait_for_task(store, task_id)


class FixtureASTVerifier:
    """Test-only checker for a static fixture mismatch, not an IDOR impact verifier."""

    name = "fixture_static_regression"

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root

    def verify(self, observation: Observation, criteria: dict[str, Any]) -> VerificationResult:
        if observation.execution_status != "success" or not isinstance(observation.observed_output, dict):
            return VerificationResult(
                "FAILED", "The local investigation did not return an object", verifier_type=self.name
            )

        output = observation.observed_output
        source_path = self.repository_root / "service.py"
        try:
            raw = source_path.read_bytes()
            tree = ast.parse(raw.decode("utf-8"), filename="service.py")
        except (OSError, UnicodeDecodeError, SyntaxError):
            return VerificationResult(
                "FAILED", "The independent fixture source check could not parse service.py", verifier_type=self.name
            )

        definitions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        observed_bad_call = False
        expected_call_line = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "load_order":
                continue
            definition = definitions.get("load_order")
            if definition is None:
                continue
            parameters = {
                arg.arg
                for arg in (*definition.args.posonlyargs, *definition.args.args, *definition.args.kwonlyargs)
            }
            if any(keyword.arg == "tenant_id" and keyword.arg not in parameters for keyword in node.keywords):
                observed_bad_call = True
                expected_call_line = int(node.lineno)
                break

        raw_findings = output.get("findings")
        findings: list[Any] = raw_findings if isinstance(raw_findings, list) else []
        matching_findings = [
            item for item in findings
            if isinstance(item, dict)
            and item.get("kind") == "static_call_signature_mismatch"
            and item.get("path") == "service.py"
            and item.get("line_start") == expected_call_line
        ]
        digest = hashlib.sha256(raw).hexdigest()
        has_bound_source_ref = any(
            isinstance(ref, dict)
            and ref.get("path") == "service.py"
            and ref.get("sha256") == digest
            and ref.get("line_start") == expected_call_line
            for finding in matching_findings
            for ref in (
                finding["evidence_refs"]
                if isinstance(finding.get("evidence_refs"), list)
                else []
            )
        )
        passed = (
            observed_bad_call
            and bool(matching_findings)
            and has_bound_source_ref
            and isinstance(output.get("snapshot"), dict)
            and output.get("root_cause", {}).get("status") == "ESTABLISHED"
            and criteria.get("finding_kind") == "static_call_signature_mismatch"
        )
        if not passed:
            return VerificationResult(
                "FAILED",
                "The reported static mismatch did not match the fixture AST and source digest",
                verifier_type=self.name,
                details={
                    "observed_bad_call": observed_bad_call,
                    "matching_findings": len(matching_findings),
                    "source_reference_bound": has_bound_source_ref,
                },
            )
        return VerificationResult(
            "VERIFIED",
            "Independent AST verification confirmed only a static call-signature mismatch; no IDOR/BOLA impact was tested",
            evidence=[{
                "type": "independent_fixture_static_mismatch",
                "path": "service.py",
                "line": expected_call_line,
                "sha256": digest,
            }],
            verifier_type=self.name,
            details={
                "finding_kind": "static_call_signature_mismatch",
                "source_checked_independently": True,
                "repository_code_executed": False,
                "verification_scope": "static_call_signature_mismatch_only",
                "security_impact_verified": False,
            },
        )


class FixtureVerificationPlanner(Planner):
    """Keep the production plan choice; only name the fixture's independent verifier."""

    def create(
        self,
        goal: str,
        tools: ToolRegistry,
        prior_knowledge: list[dict[str, Any]],
        *,
        constraints: dict[str, Any] | None = None,
        permissions: dict[str, str] | None = None,
        retrieved_experiences: list[dict[str, Any]] | None = None,
        reflection_insights: list[dict[str, Any]] | None = None,
        learned_strategies: list[dict[str, Any]] | None = None,
        task_id: str = "",
    ):
        plan = super().create(
            goal,
            tools,
            prior_knowledge,
            constraints=constraints,
            permissions=permissions,
            retrieved_experiences=retrieved_experiences,
            reflection_insights=reflection_insights,
            learned_strategies=learned_strategies,
            task_id=task_id,
        )
        if plan.steps and plan.steps[0].tool == "investigate_repository_bug":
            plan.steps[0].verifier = FixtureASTVerifier.name
            plan.steps[0].success_criteria = {"finding_kind": "static_call_signature_mismatch"}
        return plan


def _local_registry(repository_root: Path, repository_name: str = "sample/orders") -> ToolRegistry:
    inspector = RepositoryInspector(local_repositories={repository_name: repository_root})
    registry = ToolRegistry()
    # Structure is first to make the no-memory fallback's alternative explicit.
    registry.register(RepositoryTool("inspect_repository_structure", inspector))
    registry.register(BugInvestigationTool(inspector))
    return registry


def _assert_invalid_memory_does_not_select_investigator(
    planner: Planner,
    registry: ToolRegistry,
    claim: dict[str, Any] | list[dict[str, Any]],
    permissions: dict[str, str],
) -> None:
    claims = [claim] if isinstance(claim, dict) else claim
    plan = planner.create(FUTURE_GOAL, registry, claims, permissions=permissions)
    assert plan.steps[0].tool == "inspect_repository_structure"


def test_owner_reviewed_knowledge_changes_future_local_plan_after_database_reopen(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "owner-learning.sqlite3"
    repository_root = tmp_path / "local-orders"
    repository_root.mkdir()
    (repository_root / "service.py").write_text(
        "def fetch_order(order_id):\n"
        "    return load_order(order_id=order_id, tenant_id=\"tenant-a\")\n"
        "\n"
        "def load_order(order_id):\n"
        "    return {\"id\": order_id}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_KNOWLEDGE_ATTESTATION_KEY", ATTESTATION_KEY)
    monkeypatch.setenv("AGENT_OWNER_EMAIL", OWNER_EMAIL)

    # Stage 1: real authenticated owner review through the existing API and worker.
    first_service = build_api_service(token=None, database_path=str(database))
    owner_user = AuthManager(first_service.store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    other_user = AuthManager(first_service.store).create_user(
        "owner-learning-other@example.test",
        "owner-learning-other-password-123",
        ROLE_OPERATOR,
        actor_user_id=owner_user.user_id,
    )
    first_worker = BackgroundWorker(
        first_service.orchestrator,
        worker_id="owner-learning-source",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.02,
    )
    try:
        with TestClient(create_app(service=first_service, worker=first_worker), raise_server_exceptions=False) as client:
            login = client.post(
                "/api/v1/auth/login",
                json={"email": OWNER_EMAIL, "password": OWNER_PASSWORD},
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
            source_task_id = created.json()["task_id"]
            source_state = _wait_for_task(first_service.store, source_task_id)
            assert source_state and source_state["status"] == "COMPLETED"

            task_response = client.get(f"/api/v1/bug-bounty/tasks/{source_task_id}")
            assert task_response.status_code == 200, task_response.text
            findings = task_response.json()["run"]["findings"]
            assert findings and findings[0]["vulnerability_class"] == "IDOR/BOLA"
            finding_id = findings[0]["finding_id"]
            review = client.post(
                f"/api/v1/bug-bounty/tasks/{source_task_id}/findings/{finding_id}/review",
                json={"decision": "CONFIRM", "note": "owner confirmed the validated synthetic fixture"},
                headers=headers,
            )
            assert review.status_code == 200, review.text
            review_body = review.json()
            assert review_body["actor_user_id"]
            assert review_body["evidence_validated"] is True

            attested = client.post(
                f"/api/v1/bug-bounty/tasks/{source_task_id}/reviews/"
                f"{review_body['review_id']}/knowledge-attestation",
                headers=headers,
            )
            assert attested.status_code == 200, attested.text
            claim_snapshot = attested.json()["attestation"]["claim_snapshot"]
            assert claim_snapshot["knowledge_type"] == "reviewed_security_finding"
            assert claim_snapshot["predicate"] == "owner_confirmed_after_evidence_gate"
            assert "IDOR/BOLA" in claim_snapshot["object_text"]
    finally:
        first_service.store.close()

    # Stage 2: reopen the same SQLite database and use actual local repository tools.
    reopened = Store(database, knowledge_attestation_key=ATTESTATION_KEY.encode("utf-8"))
    try:
        active_claims = reopened.search_knowledge_claims(FUTURE_GOAL)
        assert len(active_claims) == 1
        claim = active_claims[0]
        assert claim["is_attested"] and claim["is_active"] and claim["is_fresh"]
        assert claim["is_verified"] and claim["status"] == "SUPPORTED"
        assert claim["metadata"]["review_id"] == review_body["review_id"]

        registry = _local_registry(repository_root)
        permissions = {
            "inspect_repository_structure": "public_read",
            "investigate_repository_bug": "public_read",
        }
        baseline = Planner().create(FUTURE_GOAL, registry, [], permissions=permissions)
        memory_plan = Planner().create(FUTURE_GOAL, registry, active_claims, permissions=permissions)
        assert baseline.steps[0].tool == "inspect_repository_structure"
        assert memory_plan.steps[0].tool == "investigate_repository_bug"
        assert memory_plan.steps[0].input["repository"] == "sample/orders"
        assert "does not establish a finding" in memory_plan.rationale

        # Negative planner controls: bad trust state, changed class, conflict, and
        # absent/disallowed tools cannot turn this claim into an investigator route.
        invalid_claims = [
            {**claim, "is_attested": False},
            {**claim, "is_fresh": False},
            {**claim, "is_active": False},
            {**claim, "is_verified": False},
            {**claim, "status": "UNSUPPORTED"},
            {**claim, "predicate": "user_asserted"},
            {**claim, "conflict_count": 1},
            {**claim, "conflict_suppression_ids": ["conflict-1"]},
            {**claim, "object_text": "Owner-confirmed Cross-Site Scripting finding"},
        ]
        for invalid_claim in invalid_claims:
            _assert_invalid_memory_does_not_select_investigator(
                Planner(), registry, invalid_claim, permissions
            )
        restricted = {"inspect_repository_structure": "public_read"}
        _assert_invalid_memory_does_not_select_investigator(Planner(), registry, claim, restricted)
        registry_without_investigator = ToolRegistry()
        registry_without_investigator.register(registry.get("inspect_repository_structure"))
        _assert_invalid_memory_does_not_select_investigator(
            Planner(), registry_without_investigator, claim, restricted
        )
        registry_with_other_repository = _local_registry(repository_root, "other/orders")
        other_permissions = {
            "inspect_repository_structure": "public_read",
            "investigate_repository_bug": "public_read",
        }
        _assert_invalid_memory_does_not_select_investigator(
            Planner(), registry_with_other_repository, claim, other_permissions
        )

        # Stage 3: API queue/worker, production Planner route, real GenericExecutor,
        # SecurityController and ToolRegistry; only the fixture verifier is test-only.
        security_controller = SecurityController(store=reopened)
        orchestrator = Orchestrator(
            reopened,
            registry,
            planner=FixtureVerificationPlanner(),
            verifiers=VerifierRegistry([FixtureASTVerifier(repository_root)]),
            approved_permissions={"public_read"},
            security_controller=security_controller,
        )
        future_service = AgentAPIService(orchestrator, token=None)
        future_worker = BackgroundWorker(
            orchestrator,
            worker_id="owner-learning-future",
            poll_interval_seconds=0.01,
            idle_poll_interval_seconds=0.02,
        )
        guarded_tasks: list[tuple[str, str, str]] = []
        with TestClient(
            create_app(service=future_service, worker=future_worker),
            raise_server_exceptions=False,
        ) as client:
            login = client.post(
                "/api/v1/auth/login",
                json={"email": OWNER_EMAIL, "password": OWNER_PASSWORD},
            )
            assert login.status_code == 200, login.text
            future_task_id, future_state = _submit_goal_via_api(
                client, reopened, idempotency_key="owner-learning-future-task"
            )
            task_response = client.get(f"/api/v1/tasks/{future_task_id}")
            assert task_response.status_code == 200, task_response.text
            assert task_response.json()["status"] == "COMPLETED"
            assert task_response.json()["verification_status"] == "INSUFFICIENT"
            task_list_response = client.get("/api/v1/tasks")
            assert task_list_response.status_code == 200, task_list_response.text
            listed_task = next(item for item in task_list_response.json()["items"] if item["task_id"] == future_task_id)
            assert listed_task["verification_status"] == "INSUFFICIENT"

            # A stronger deny rule blocks the same memory-selected tool.
            deny_policy = security_controller.policy_engine.policy
            deny_policy.policy_version += 1
            deny_policy.denied_capabilities = ["READ_WORKSPACE"]
            security_controller.update_policy(deny_policy, actor="test-owner")
            denied_task_id, denied_state = _submit_goal_via_api(
                client, reopened, idempotency_key="owner-learning-denied-task"
            )
            assert denied_state and denied_state["status"] == "BLOCKED"
            guarded_tasks.append((denied_task_id, "BLOCKED", "DENY"))

            # Removing the deny does not grant execution: LOW-risk reading still
            # waits for a separately bound owner approval under this test policy.
            approval_policy = security_controller.policy_engine.policy
            approval_policy.policy_version += 1
            approval_policy.denied_capabilities = []
            approval_policy.approval_required_risks = ["LOW"]
            security_controller.update_policy(approval_policy, actor="test-owner")
            approval_task_id, approval_state = _submit_goal_via_api(
                client, reopened, idempotency_key="owner-learning-approval-task"
            )
            assert approval_state and approval_state["status"] == "WAITING_APPROVAL"
            guarded_tasks.append((approval_task_id, "WAITING_APPROVAL", "REQUIRE_APPROVAL"))

        assert future_state and future_state["status"] == "COMPLETED", future_state
        persisted = reopened.get_task_state(future_task_id)
        assert persisted is not None
        assert persisted["plan"]["steps"][0]["tool"] == "investigate_repository_bug"
        assert persisted["context"]["prior_knowledge_used"][0]["claim_id"] == claim["claim_id"]
        result = persisted["result"]
        assert result["final_status"] == "COMPLETED"
        task_verification = result["verification"]
        assert task_verification["status"] == "INSUFFICIENT"
        assert task_verification["verified"] is False
        assert task_verification["verification_scope"] == "STATIC_ONLY"
        assert task_verification["security_impact_verified"] is False
        assert task_verification["details"]["code"] == "security_impact_not_verified"
        assert result["step_results"][0]["tool_name"] == "investigate_repository_bug"
        assert result["step_results"][0]["status"] == "success"
        assert result["step_results"][0]["metadata"]["target_code_executed"] is False
        verification = result["final_step_outcomes"][0]["verification"]
        assert verification["status"] == "VERIFIED"
        assert verification["verifier_type"] == FixtureASTVerifier.name
        assert verification["details"]["source_checked_independently"] is True
        assert verification["details"]["verification_scope"] == "static_call_signature_mismatch_only"
        assert verification["details"]["security_impact_verified"] is False
        assert result["experience"]["verification_status"] == "INSUFFICIENT"
        assert result["experience"]["verification_scope"] == "STATIC_ONLY"
        assert result["experience"]["security_impact_verified"] is False
        assert result["experience"]["identity"]["verification_status"] == "INSUFFICIENT"
        assert result["experience"]["identity"]["verification_scope"] == "STATIC_ONLY"
        assert result["experience"]["identity"]["security_impact_verified"] is False
        report = result["step_results"][0]["output"]
        assert report["root_cause"]["status"] == "ESTABLISHED"
        assert any(item["kind"] == "static_call_signature_mismatch" for item in report["findings"])
        assert not any(item.get("vulnerability_class") for item in report["findings"])
        assert all(item["decision"] == "ALLOW" for item in result["security_decisions"])

        # The attested hint never invokes the tool when policy denies or requests approval.
        for guarded_task_id, expected_status, expected_decision in guarded_tasks:
            guarded_persisted = reopened.get_task_state(guarded_task_id)
            assert guarded_persisted is not None
            assert guarded_persisted["status"] == expected_status
            assert guarded_persisted["plan"]["steps"][0]["tool"] == "investigate_repository_bug"
            guarded_result = guarded_persisted["result"]
            assert guarded_result["final_status"] == "BLOCKED"
            assert guarded_result["step_results"] == []
            assert guarded_result["security_decisions"][0]["decision"] == expected_decision
            guarded_audit = security_controller.audit.list(task_id=guarded_task_id)
            expected_event = "PERMISSION_DENIED" if expected_decision == "DENY" else "APPROVAL_REQUESTED"
            assert any(event["event_type"] == expected_event for event in guarded_audit)
            assert not any(event["event_type"] == "ACTION_EXECUTED" for event in guarded_audit)
            if expected_decision == "REQUIRE_APPROVAL":
                assert guarded_result["approval_requests"]
                assert guarded_result["approval_requests"][0]["status"] == "PENDING"

        # Read-path negative controls: the real SQLite retrieval boundary must
        # keep owner-bound, stale, and conflicted claims from steering planning.
        ownerless_task_id = reopened.create_queued_task(FUTURE_GOAL)["task_id"]
        other_owner_task_id = reopened.create_queued_task(
            FUTURE_GOAL, owner_user_id=other_user.user_id
        )["task_id"]
        for task_id in (ownerless_task_id, other_owner_task_id):
            rejected_claims = reopened.search_knowledge_claims(FUTURE_GOAL, task_id=task_id)
            assert rejected_claims == []
            _assert_invalid_memory_does_not_select_investigator(
                Planner(), registry, rejected_claims, permissions
            )

        stale_claims = reopened.search_knowledge_claims(
            FUTURE_GOAL,
            task_id=future_task_id,
            as_of=(datetime.now(timezone.utc) + timedelta(days=31)).isoformat(),
        )
        assert stale_claims == []
        _assert_invalid_memory_does_not_select_investigator(
            Planner(), registry, stale_claims, permissions
        )

        evidence_rows = reopened.list_knowledge_evidence(claim["claim_id"])
        assert evidence_rows
        contradictory_claim = reopened.save_knowledge_claim(KnowledgeClaim(
            claim_id="claim-owner-learning-e2e-contradiction",
            domain="cybersecurity.bugbounty",
            subject="owner-refuted IDOR/BOLA fixture",
            predicate="contradicts",
            object_text="A local owner review rejected the IDOR/BOLA claim",
            knowledge_type="reviewed_security_finding",
        ))
        conflict_id = "conflict-owner-learning-e2e"
        reopened.save_knowledge_conflict(KnowledgeConflict(
            conflict_id=conflict_id,
            claim_id=claim["claim_id"],
            conflicting_claim_id=contradictory_claim.claim_id,
            conflict_type="owner_refutation",
            evidence_ids=[evidence_rows[0]["evidence_id"]],
        ))
        saved_conflicts = reopened.list_knowledge_conflicts(claim["claim_id"])
        assert any(
            item["conflict_id"] == conflict_id and item["status"] == "OPEN"
            for item in saved_conflicts
        )
        inactive_claims = reopened.search_knowledge_claims(
            FUTURE_GOAL, task_id=future_task_id, include_unverified=True
        )
        inactive_claim = next(
            (item for item in inactive_claims if item["claim_id"] == claim["claim_id"]),
            None,
        )
        assert inactive_claim is not None
        assert inactive_claim["is_active"] is False
        assert conflict_id in inactive_claim["conflict_suppression_ids"]

        # Conflict persistence and its signed suppression survive another reopen.
        reopened.close()
        reopened = Store(database, knowledge_attestation_key=ATTESTATION_KEY.encode("utf-8"))
        reopened_conflicts = reopened.list_knowledge_conflicts(claim["claim_id"])
        assert any(
            item["conflict_id"] == conflict_id and item["status"] == "OPEN"
            for item in reopened_conflicts
        )
        reopened_inactive_claims = reopened.search_knowledge_claims(
            FUTURE_GOAL, task_id=future_task_id, include_unverified=True
        )
        reopened_inactive_claim = next(
            (item for item in reopened_inactive_claims if item["claim_id"] == claim["claim_id"]),
            None,
        )
        assert reopened_inactive_claim is not None
        assert reopened_inactive_claim["is_active"] is False
        assert conflict_id in reopened_inactive_claim["conflict_suppression_ids"]
        conflicted_claims = reopened.search_knowledge_claims(
            FUTURE_GOAL, task_id=future_task_id
        )
        assert conflicted_claims == []
        _assert_invalid_memory_does_not_select_investigator(
            Planner(), registry, conflicted_claims, permissions
        )
    finally:
        reopened.close()
