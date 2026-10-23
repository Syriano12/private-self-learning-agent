from __future__ import annotations

from private_agent.bugbounty.cross_application import (
    ConsoleApplication,
    CrossApplicationSample,
    LedgerApplication,
    TenantVaultApplication,
    build_cross_application_corpus,
    evaluate_cross_application_corpus,
)
from private_agent.bugbounty.engines import IDORBOLAEngine
from private_agent.bugbounty.models import FindingState, PolicyDecision


def test_phase20_false_positives_are_resolved_by_observed_policy_semantics() -> None:
    result = evaluate_cross_application_corpus()
    assert result["phase20_baseline"]["false_positives"] == 2
    assert result["phase20_false_positive_resolution"]
    cases = {case["case_id"]: case for case in result["cases"]}
    assert cases["V-01"]["expected_outcome"] == "SAFE"
    assert cases["V-01"]["predicted_outcome"] == "SAFE"
    assert cases["P-01"]["expected_outcome"] == "SAFE"
    assert cases["P-01"]["predicted_outcome"] == "SAFE"
    assert result["held_out"]["false_positives"] == 0


def test_genuine_unauthorized_access_still_requires_policy_deny_and_is_confirmed() -> None:
    result = evaluate_cross_application_corpus()
    cases = {case["case_id"]: case for case in result["cases"]}
    for case_id in ("L-02", "C-02", "V-02", "P-02"):
        assert cases[case_id]["predicted_outcome"] == "VULNERABLE"
        assert cases[case_id]["evidence_quality"] == "observed_authorization_comparison"
        assert cases[case_id]["evidence_errors"] == []


def test_authorized_non_owner_is_not_confirmed_when_policy_adapter_allows_it() -> None:
    app = TenantVaultApplication(0)
    url = app.url_for(app.endpoints[0], "vault-a")
    observation = app.authorization_decision("carol", "GET", url, resource_id="vault-a")
    assert observation.decision is PolicyDecision.ALLOW
    response = app.request("carol", "GET", url)
    assert response.status == 200
    assert response.body["classification"] == "restricted"


def test_missing_policy_semantics_produces_suspected_not_confirmed() -> None:
    app = LedgerApplication(1)
    app.authorization_decision = None
    sample = next(item for item in build_cross_application_corpus() if item.oracle.case_id == "L-02")
    sample = CrossApplicationSample(sample.application_id, app, sample.scope, sample.oracle)
    findings, tests, _ = IDORBOLAEngine(sample.scope).scan(sample.application)
    assert findings
    assert any(item.state is FindingState.HYPOTHESIS for item in findings)
    assert not any(item.state is FindingState.REPORT_READY for item in findings)
    assert any(item.status == "SUSPECTED" for item in tests)
    assert not any(item.status == "VULNERABLE" for item in tests)


def test_role_policy_observation_is_operation_specific_and_not_owner_based() -> None:
    app = ConsoleApplication(0)
    url = app.url_for(app.endpoints[0], "audit-7")
    root = app.authorization_decision("root", "GET", url, resource_id="audit-7")
    analyst = app.authorization_decision("analyst", "GET", url, resource_id="audit-7")
    assert root.decision is PolicyDecision.ALLOW
    assert analyst.decision is PolicyDecision.DENY
    assert app.request("analyst", "GET", url).status == 403


def test_operation_policy_can_allow_read_and_deny_write() -> None:
    app = ConsoleApplication(2)
    url = app.url_for(app.endpoints[0], "audit-7")
    read = app.authorization_decision("analyst", "GET", url, resource_id="audit-7")
    write = app.authorization_decision("analyst", "POST", url, resource_id="audit-7")
    assert read.decision is PolicyDecision.ALLOW
    assert write.decision is PolicyDecision.DENY
    assert app.request("analyst", "GET", url).status == 200
    assert app.request("analyst", "POST", url).status == 403


def test_phase21_verdict_is_limited_and_not_production_accuracy() -> None:
    result = evaluate_cross_application_corpus()
    assert result["verdict"] == "GENERALIZATION DEMONSTRATED ON LIMITED LOCAL APPLICATIONS"
    assert result["external_targets_contacted"] == 0
    assert any("synthetic" in item.lower() for item in result["application_specific_limitations"])
