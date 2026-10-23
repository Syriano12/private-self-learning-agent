from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from private_agent.bugbounty import BugBountyAgent, BugBountyRepository, DeterministicFixtureApp, FindingState, ScopeDecision, ScopeEnforcedFixtureClient, ScopeViolation, StaticSecurityAnalyzer, evaluate_local_benchmark, markdown_report
from private_agent.bugbounty.models import Evidence, Finding, Severity, ScopeDefinition
from private_agent.storage import Store


def scope_payload(*, active: bool = True) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "program_id": "test-program",
        "authorization_reference": "local-fixture-authorization",
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "assets": [{"asset_id": "fixture-api", "host": "fixture.local", "path_prefix": "/api/v1", "methods": ["GET"], "active_allowed": active}],
        "permitted_identities": ["alice", "bob", "admin"],
        "permitted_resources": ["order-alice", "order-bob", "comment-bob", "report-admin"],
        "permitted_methods": ["GET"],
        "prohibited_methods": ["DELETE", "PATCH", "PUT", "POST"],
        "active_testing": active,
    }


def test_scope_requires_explicit_authorization_and_never_expands_to_redirected_host():
    scope = ScopeDefinition.from_dict(scope_payload())
    assert scope.authorize("https://fixture.local/api/v1/orders/order-alice", mode="active", identity="alice", resource_id="order-alice")[0] is ScopeDecision.ALLOWED
    assert scope.authorize("https://other.example/api/v1/orders/order-alice", mode="active", identity="alice", resource_id="order-alice")[0] is ScopeDecision.DENIED
    assert scope.authorize("https://fixture.local/api/v1/orders/order-alice", method="POST", mode="active", identity="alice", resource_id="order-alice")[0] is ScopeDecision.DENIED
    assert scope.authorize("https://fixture.local/api/v1/orders/order-alice", mode="active", identity="alice", resource_id="not-declared")[0] is ScopeDecision.DENIED


def test_scope_enforced_client_blocks_before_fixture_side_effect():
    scope = ScopeDefinition.from_dict(scope_payload())
    fixture = DeterministicFixtureApp("secure")
    client = ScopeEnforcedFixtureClient(fixture, scope)
    with pytest.raises(ScopeViolation):
        client.request("alice", "GET", "https://other.example/api/v1/orders/order-alice", resource_id="order-alice")
    with pytest.raises(ScopeViolation):
        client.request("alice", "POST", "https://fixture.local/api/v1/orders/order-alice", resource_id="order-alice")
    assert client.calls == 0


def test_active_scope_requires_explicit_identity_and_resource_binding():
    scope = ScopeDefinition.from_dict(scope_payload())
    decision, reason = scope.authorize(
        "https://fixture.local/api/v1/orders/order-alice",
        mode="active",
    )
    assert decision is ScopeDecision.DENIED
    assert reason == "identity_required_for_active_testing"


def test_scope_path_prefix_is_segment_bounded():
    scope = ScopeDefinition.from_dict(scope_payload())
    decision, reason = scope.authorize(
        "https://fixture.local/api/v10/orders/order-alice",
        mode="active",
        identity="alice",
        resource_id="order-alice",
    )
    assert decision is ScopeDecision.DENIED
    assert reason == "asset_out_of_scope"


@pytest.mark.parametrize(
    "url",
    (
        "ftp://fixture.local/api/v1/orders/order-alice",
        "https://user:password@fixture.local/api/v1/orders/order-alice",
        "https://fixture.local/api/v1/../admin/orders/order-alice",
        "https://fixture.local/api/v1/%2e%2e/admin/orders/order-alice",
    ),
)
def test_scope_rejects_non_http_credentials_and_path_traversal(url):
    scope = ScopeDefinition.from_dict(scope_payload())
    decision, reason = scope.authorize(
        url,
        mode="active",
        identity="alice",
        resource_id="order-alice",
    )
    assert decision is ScopeDecision.DENIED
    assert reason in {"scheme_not_permitted", "userinfo_not_permitted", "path_traversal_not_permitted"}


def test_scope_rejects_ambiguous_boolean_and_list_shapes():
    payload = scope_payload()
    payload["active_testing"] = "false"
    with pytest.raises(ValueError, match="active_testing_must_be_boolean"):
        ScopeDefinition.from_dict(payload)

    payload = scope_payload()
    payload["permitted_identities"] = "alice"
    with pytest.raises(ValueError, match="permitted_identities_must_be_list"):
        ScopeDefinition.from_dict(payload)

    payload = scope_payload()
    payload["assets"][0]["active_allowed"] = "false"
    with pytest.raises(ValueError, match="active_allowed_must_be_boolean"):
        ScopeDefinition.from_dict(payload)


def test_fixture_client_rejects_resource_url_mismatch_before_request():
    scope = ScopeDefinition.from_dict(scope_payload())
    fixture = DeterministicFixtureApp("secure")
    client = ScopeEnforcedFixtureClient(fixture, scope)
    with pytest.raises(ScopeViolation, match="resource_url_mismatch"):
        client.request(
            "alice",
            "GET",
            "https://fixture.local/api/v1/orders/order-bob",
            resource_id="order-alice",
        )
    assert client.calls == 0


def test_expired_or_ambiguous_scope_fails_closed():
    payload = scope_payload()
    payload["issued_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    payload["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="scope_expired"):
        ScopeDefinition.from_dict(payload)
    payload = scope_payload()
    payload["ambiguous"] = True
    scope = ScopeDefinition.from_dict(payload)
    assert scope.authorize("https://fixture.local/api/v1/orders/order-alice", mode="active")[0] is ScopeDecision.AMBIGUOUS


@pytest.mark.parametrize("variant,expected_class", [
    ("secure", 0),
    ("idor_horizontal", 1),
    ("idor_vertical", 1),
    ("idor_nested", 1),
])
def test_idor_engine_distinguishes_secure_and_controlled_vulnerable_fixtures(variant, expected_class):
    run = BugBountyAgent().run(scope_payload(), fixture_variant=variant, persist=False)
    assert len([item for item in run.findings if item.state is FindingState.REPORT_READY]) == expected_class
    assert run.metrics()["tests"] >= 1
    if expected_class:
        finding = run.findings[0]
        assert finding.vulnerability_class == "IDOR/BOLA"
        assert finding.evidence[0].provenance["scope_reference"] == "local-fixture-authorization"
        assert all("fixture.local" in evidence.request["url"] for evidence in finding.evidence)


def test_false_positive_dynamic_response_difference_is_not_reported():
    run = BugBountyAgent().run(scope_payload(), fixture_variant="false_positive", persist=False)
    assert run.findings == []
    assert any(item.status == "SECURE" for item in run.tests)


def test_unavailable_fixture_remains_inconclusive_not_confirmed():
    run = BugBountyAgent().run(scope_payload(), fixture_variant="inconclusive", persist=False)
    assert run.findings == []
    assert any(item.status == "INCONCLUSIVE" for item in run.tests)
    assert not any(item.state is FindingState.CONFIRMED for item in run.findings)


def test_passive_recon_and_static_analysis_are_bounded_and_advisory():
    run = BugBountyAgent().run(
        scope_payload(active=False),
        fixture_variant="secure",
        urls=("https://fixture.local/api/v1/orders/{order_id}",),
        openapi={"servers": [{"url": "https://fixture.local"}], "paths": {"/api/v1/orders/{order_id}": {"get": {"operationId": "readOrder"}}}},
        source_documents={"local.py": "subprocess.run(user_value, shell=True)"},
        persist=False,
    )
    assert run.status == "COMPLETED_WITH_LIMITATIONS"
    assert any(item.state is FindingState.HYPOTHESIS for item in run.findings)
    assert run.findings[0].limitations
    assert "active_fixture_testing_not_run_without_explicit_active_scope" in run.limitations


def test_passive_recon_caps_url_and_document_input_counts():
    run = BugBountyAgent().run(
        scope_payload(active=False),
        urls=tuple("https://fixture.local/api/v1/orders/order-alice" for _ in range(501)),
        source_documents={f"source-{index}.py": "permission check" for index in range(101)},
        persist=False,
    )
    assert "url_input_limit_reached" in run.limitations
    assert "source_document_limit_reached" in run.limitations


def test_repository_persists_run_and_findings_across_sqlite_reopen(tmp_path: Path):
    db_path = tmp_path / "bounty.sqlite3"
    store = Store(db_path)
    run = BugBountyAgent(store=store).run(scope_payload(), fixture_variant="idor_horizontal", persist=True)
    assert BugBountyRepository(store).get(run.run_id)["findings"]
    store.close()
    reopened = Store(db_path)
    stored = BugBountyRepository(reopened).get(run.run_id)
    assert stored["run_id"] == run.run_id
    assert stored["findings"][0]["state"] == "REPORT_READY"
    assert BugBountyRepository(reopened).list_runs(limit=10)[0]["run_id"] == run.run_id
    patterns = BugBountyRepository(reopened).search_patterns(vulnerability_class="IDOR/BOLA")
    assert len(patterns) == 1
    template = BugBountyRepository(reopened).regression_template(patterns[0]["pattern_id"])
    assert template["kind"] == "non_executable_regression_template"
    BugBountyRepository(reopened).record_feedback(patterns[0]["pattern_id"], label="FALSE_POSITIVE", actor="owner", reason="manual review rejected this fixture result")
    assert BugBountyRepository(reopened).search_patterns(vulnerability_class="IDOR/BOLA") == []
    assert BugBountyRepository(reopened).regression_template(patterns[0]["pattern_id"]) is None
    reopened.close()


def test_markdown_report_is_explicit_about_local_scope_and_unverified_static_results():
    run = BugBountyAgent().run(scope_payload(active=False), source_documents={"local.py": "subprocess.run(x, shell=True)"}, persist=False)
    report = markdown_report(run)
    assert "External targets: `none`" in report
    assert "HYPOTHESIS" in report
    assert "unverified" not in report.lower() or "not a confirmed vulnerability" in report.lower()


def test_authorization_reference_is_opaque_in_public_report_and_run():
    payload = scope_payload()
    payload["authorization_reference"] = "SUPER-SECRET-AUTH-REF"
    run = BugBountyAgent().run(payload, fixture_variant="idor_horizontal", persist=False)
    public_payload = json.dumps(run.to_dict(public=True), ensure_ascii=False)
    assert "SUPER-SECRET-AUTH-REF" not in public_payload
    assert "Authorization reference digest:" in markdown_report(run)
    assert "SUPER-SECRET-AUTH-REF" not in markdown_report(run)


def test_finding_lifecycle_rejects_confirmation_without_evidence():
    finding = Finding("F-X", "IDOR/BOLA", "x", "fixture.local", "/x", FindingState.OBSERVED, Severity.HIGH, 0.5, "scope")
    finding.transition(FindingState.HYPOTHESIS)
    finding.transition(FindingState.VALIDATION_PENDING)
    with pytest.raises(ValueError, match="finding_evidence_required"):
        finding.transition(FindingState.CONFIRMED)


def test_evidence_redacts_credentials_and_bounds_json():
    evidence = Evidence(
        kind="test",
        source="fixture",
        expected={"ok": True},
        observed={"authorization": "Bearer secret", "value": "ok"},
        provenance={"token": "do-not-store"},
        request={"cookie": "secret-cookie"},
        response={"set-cookie": "secret"},
    )
    rendered = json.dumps(evidence.to_dict())
    assert "Bearer secret" not in rendered
    assert "secret-cookie" not in rendered
    assert "do-not-store" not in rendered


def test_evidence_rejects_non_finite_and_non_json_values_without_calling_hooks():
    class HostileValue:
        def __str__(self):
            raise AssertionError("untrusted string hook must not run")

    with pytest.raises(ValueError, match="evidence_non_finite_number"):
        Evidence(kind="test", source="fixture", expected={"score": float("nan")}, observed={}, provenance={})
    with pytest.raises(ValueError, match="evidence_value_not_json_primitive"):
        Evidence(kind="test", source="fixture", expected={"value": HostileValue()}, observed={}, provenance={})


def test_static_evidence_redacts_secret_shaped_source_text():
    run = BugBountyAgent().run(scope_payload(active=False), source_documents={"local.py": "authorization: Bearer super-secret-value"}, persist=False)
    # The source is an advisory artifact; regardless of detector matches, raw credential text must not persist.
    rendered = json.dumps(run.to_dict())
    assert "super-secret-value" not in rendered


def test_static_analyzer_covers_additional_injection_and_auth_patterns_as_hypotheses():
    findings = StaticSecurityAnalyzer().analyze(
        {
            "local.py": """
                requests.get(request.args['url'])
                payload = pickle.loads(request.body)
                node.innerHTML = request.args['html']
                path = open(request.args['path'])
                token = jwt.decode(value, key, verify_signature=False)
                return redirect(request.args['next'])
            """
        },
        "local-static-authorization",
    )
    classes = {finding.vulnerability_class for finding in findings}
    assert {"SSRF", "Unsafe Deserialization", "Cross-Site Scripting", "Path Traversal", "Authentication", "Open Redirect"} <= classes
    assert all(finding.state is FindingState.HYPOTHESIS for finding in findings)


def test_local_benchmark_reports_actual_fixture_metrics_without_external_targets():
    benchmark = evaluate_local_benchmark()
    assert benchmark["passed"] == benchmark["total"] == 6
    assert benchmark["external_targets_contacted"] == 0
    assert benchmark["true_positives"] == 3
    assert benchmark["false_positives"] == 0
    assert benchmark["false_negatives"] == 0
    assert benchmark["true_negatives"] == 3
    assert benchmark["precision"] == 1.0
    assert benchmark["recall"] == 1.0
