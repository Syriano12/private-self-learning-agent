from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from private_agent.bugbounty import (
    BugBountyAgent,
    BugBountyRepository,
    BugBountyRun,
    Finding,
    FindingState,
    Severity,
    evaluate_cross_application_corpus,
    markdown_report,
)
from private_agent.bugbounty.models import Evidence
from private_agent.api import AgentAPIService
from private_agent.core.experience import sanitize_value
from private_agent.storage import Store


def scope_payload(*, active: bool = True) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "program_id": "phase24",
        "authorization_reference": "phase24-written-authorization",
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "assets": [{
            "asset_id": "fixture-api",
            "host": "fixture.local",
            "path_prefix": "/api/v1",
            "methods": ["GET"],
            "active_allowed": active,
        }],
        "permitted_identities": ["alice", "bob", "admin"],
        "permitted_resources": ["order-alice", "order-bob", "comment-bob", "report-admin"],
        "permitted_methods": ["GET"],
        "prohibited_methods": ["DELETE", "PATCH", "PUT", "POST"],
        "active_testing": active,
    }


def valid_run() -> BugBountyRun:
    run = BugBountyAgent().run(scope_payload(), fixture_variant="idor_horizontal", persist=False)
    assert run.findings
    assert run.findings[0].state is FindingState.REPORT_READY
    return run


def candidate_from_valid(mutator) -> Finding:
    finding = valid_run().findings[0]
    evidence = finding.evidence[0]
    changed = mutator(evidence)
    return replace(finding, state=FindingState.VALIDATION_PENDING, evidence=[changed])


def transition_to_confirmed(finding: Finding) -> None:
    finding.transition(FindingState.CONFIRMED)


def test_direct_confirmed_construction_fails_closed_without_semantic_validation():
    with pytest.raises(ValueError, match="finding_confirmation"):
        Finding(
            "F-direct",
            "IDOR/BOLA",
            "unsupported direct confirmation",
            "fixture.local",
            "/orders/{id}",
            FindingState.CONFIRMED,
            Severity.HIGH,
            0.9,
            "scope",
        )


def test_transition_rejects_missing_policy_decision():
    def mutate(evidence: Evidence) -> Evidence:
        provenance = dict(evidence.provenance)
        provenance.pop("authorization_decision", None)
        return replace(evidence, provenance=provenance)

    with pytest.raises(ValueError, match="authorization_decision"):
        transition_to_confirmed(candidate_from_valid(mutate))


def test_transition_rejects_unknown_policy():
    def mutate(evidence: Evidence) -> Evidence:
        provenance = dict(evidence.provenance)
        provenance["authorization_decision"] = "UNKNOWN"
        return replace(evidence, provenance=provenance)

    with pytest.raises(ValueError, match="authorization_decision_not_deny"):
        transition_to_confirmed(candidate_from_valid(mutate))


def test_transition_rejects_allow_policy_presented_as_idor():
    def mutate(evidence: Evidence) -> Evidence:
        provenance = dict(evidence.provenance)
        provenance["authorization_decision"] = "ALLOW"
        expected = dict(evidence.expected)
        expected["policy_decision"] = "ALLOW"
        return replace(evidence, provenance=provenance, expected=expected)

    with pytest.raises(ValueError, match="authorization_decision_not_deny"):
        transition_to_confirmed(candidate_from_valid(mutate))


def test_transition_rejects_deny_without_protected_content():
    def mutate(evidence: Evidence) -> Evidence:
        observed = dict(evidence.observed)
        observed["body"] = {"message": "same public response"}
        return replace(evidence, observed=observed)

    with pytest.raises(ValueError, match="protected_content_not_observed"):
        transition_to_confirmed(candidate_from_valid(mutate))


def test_transition_rejects_missing_baseline_protected_content():
    def mutate(evidence: Evidence) -> Evidence:
        observed = dict(evidence.observed)
        expected = dict(evidence.expected)
        expected["body"] = {"status": "baseline captured without protected fields"}
        return replace(evidence, expected=expected, observed=observed)

    with pytest.raises(ValueError, match="baseline_protected_content_missing"):
        transition_to_confirmed(candidate_from_valid(mutate))


def test_transition_rejects_identity_resource_and_request_binding_mismatch():
    def mutate(evidence: Evidence) -> Evidence:
        provenance = dict(evidence.provenance)
        request = dict(evidence.request)
        request["identity"] = "alice"
        request["url"] = request["url"].replace("order-alice", "order-other")
        provenance["resource_id"] = "order-other"
        return replace(evidence, provenance=provenance, request=request)

    with pytest.raises(ValueError, match="identity_binding_mismatch"):
        transition_to_confirmed(candidate_from_valid(mutate))


def test_transition_rejects_contradictory_policy_and_observation():
    def mutate(evidence: Evidence) -> Evidence:
        observed = dict(evidence.observed)
        observed["status"] = 403
        return replace(evidence, observed=observed)

    with pytest.raises(ValueError, match="observed_result_was_not_allowed"):
        transition_to_confirmed(candidate_from_valid(mutate))


def test_evidence_mutation_after_validation_invalidates_confirmation():
    finding = valid_run().findings[0]
    finding.evidence[0].observed["body"] = {"owner_id": "tampered"}
    with pytest.raises(ValueError, match="finding_evidence_mutated_after_validation"):
        finding.to_dict(public=True)
    assert finding.state is FindingState.HYPOTHESIS
    assert finding.validation_status == "REJECTED"


def test_validation_errors_after_confirmation_invalidate_reporting():
    finding = valid_run().findings[0]
    finding.validation_errors = ["forged_validation_note"]
    with pytest.raises(ValueError, match="finding_validation_metadata_invalid"):
        finding.to_dict(public=True)
    assert finding.state is FindingState.HYPOTHESIS


def test_confirmation_requires_evidence_asset_and_endpoint_binding():
    finding = valid_run().findings[0]
    finding.state = FindingState.VALIDATION_PENDING
    finding.asset = "https://other.example"
    finding.endpoint = "/api/v1/admin/{resource_id}"
    with pytest.raises(ValueError, match="asset_binding_mismatch|endpoint_binding_mismatch"):
        finding.transition(FindingState.CONFIRMED)


def test_policy_decision_enum_survives_persistence_redaction_boundary():
    assert sanitize_value({"authorization_decision": "DENY"})["authorization_decision"] == "DENY"
    assert sanitize_value({"authorization": "Bearer secret-value"})["authorization"] == "[REDACTED]"


def test_report_generation_rejects_mutated_confirmed_finding():
    run = valid_run()
    run.findings[0].evidence[0].observed["body"] = {"owner_id": "tampered"}
    with pytest.raises(ValueError, match="finding_evidence_mutated_after_validation"):
        markdown_report(run)


def test_unrelated_static_category_cannot_be_confirmed_with_idor_evidence():
    run = valid_run()
    finding = replace(
        run.findings[0],
        state=FindingState.VALIDATION_PENDING,
        vulnerability_class="Cross-Site Scripting",
    )
    with pytest.raises(ValueError, match="unsupported_confirmation_category"):
        transition_to_confirmed(finding)


def test_repository_load_downgrades_forged_report_ready_record(tmp_path: Path):
    db_path = str(tmp_path / "phase24.sqlite3")
    store = Store(db_path)
    run = valid_run()
    repository = BugBountyRepository(store)
    repository.save(run)
    row = store.db.execute(
        "SELECT finding_json FROM bugbounty_findings WHERE run_id = ?",
        (run.run_id,),
    ).fetchone()
    forged = json.loads(row["finding_json"])
    forged["state"] = "REPORT_READY"
    forged["evidence"] = []
    forged.pop("validation", None)
    store.db.execute(
        "UPDATE bugbounty_findings SET finding_json = ? WHERE run_id = ?",
        (json.dumps(forged), run.run_id),
    )
    store.db.commit()

    loaded = repository.get(run.run_id)
    assert loaded is not None
    finding = loaded["findings"][0]
    assert finding["state"] == "HYPOTHESIS"
    assert finding["validation"]["status"] == "REJECTED"
    assert finding["validation"]["errors"]
    assert loaded["metrics"]["confirmed_findings"] == 0


def test_api_sanitizer_suppresses_stale_report_and_reconciles_metrics():
    run = valid_run().to_dict(public=True)
    run["findings"][0]["state"] = "REPORT_READY"
    run["findings"][0]["evidence"] = []
    run["metrics"]["confirmed_findings"] = 1
    sanitized, report_suppressed = AgentAPIService._sanitize_bug_bounty_run(run)
    assert report_suppressed is True
    assert sanitized["findings"][0]["state"] == "HYPOTHESIS"
    assert sanitized["metrics"]["confirmed_findings"] == 0
    assert "report_suppressed_after_finding_validation" in sanitized["limitations"]


def test_repository_preserves_per_case_status_ledger_and_sanitized_validation(tmp_path: Path):
    db_path = str(tmp_path / "phase24-ledger.sqlite3")
    store = Store(db_path)
    run = valid_run()
    repository = BugBountyRepository(store)
    repository.save(run)
    loaded = repository.get(run.run_id)
    assert loaded is not None
    assert loaded["tests"]
    statuses = {item["status"] for item in loaded["tests"]}
    assert "VULNERABLE" in statuses
    assert all("authorization" not in json.dumps(item).lower() or "[REDACTED]" in json.dumps(item) or "sha256:" in json.dumps(item) for item in loaded["tests"])
    assert loaded["findings"][0]["validation"]["status"] == "VALIDATED"


def test_run_serialization_preserves_conservative_statuses_and_evidence():
    run = valid_run()
    static = BugBountyAgent().run(
        scope_payload(active=False),
        source_documents={"local.py": "element.innerHTML = request.args['html']"},
        persist=False,
    )
    payload = run.to_dict(public=True)
    static_payload = static.to_dict(public=True)
    assert payload["tests"]
    assert payload["findings"][0]["state"] == "REPORT_READY"
    assert payload["findings"][0]["validation"]["status"] == "VALIDATED"
    assert static_payload["findings"][0]["state"] == "HYPOTHESIS"
    assert static_payload["findings"][0]["validation"]["status"] == "NOT_VALIDATED"


def test_legitimate_membership_and_tenant_policy_cases_remain_safe():
    result = evaluate_cross_application_corpus()
    cases = {case["case_id"]: case for case in result["cases"]}
    assert cases["V-01"]["predicted_outcome"] == "SAFE"
    assert cases["P-01"]["predicted_outcome"] == "SAFE"
    assert cases["R-01"]["predicted_outcome"] == "INCONCLUSIVE"
