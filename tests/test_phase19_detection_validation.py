from __future__ import annotations

import json

from private_agent.bugbounty.validation import (
    EXPECTED_OUTCOMES,
    build_independent_corpus,
    evaluate_independent_corpus,
    validate_confirmed_evidence,
)
from private_agent.bugbounty.engines import IDORBOLAEngine


def test_independent_corpus_keeps_ground_truth_outside_fixture_objects() -> None:
    samples = build_independent_corpus()
    assert len(samples) >= 12
    assert {sample.oracle.split for sample in samples} == {"train", "validation", "held_out"}
    assert not any(hasattr(sample.fixture, "expected_outcome") for sample in samples)
    assert not any(hasattr(sample.fixture, "ground_truth") for sample in samples)
    assert all(sample.oracle.expected_outcome in EXPECTED_OUTCOMES for sample in samples)
    assert len({sample.oracle.case_id for sample in samples}) == len(samples)


def test_independent_corpus_evaluation_is_reproducible_and_reports_splits() -> None:
    first = evaluate_independent_corpus()
    second = evaluate_independent_corpus()
    assert first == second
    assert first["external_targets_contacted"] == 0
    assert first["ground_truth_accessible_to_engine"] is False
    assert set(first["splits"]) == {"train", "validation", "held_out"}
    assert first["splits"]["held_out"]["total"] >= 4
    assert "precision" in first["splits"]["held_out"]
    assert "recall" in first["splits"]["held_out"]
    assert "f1" in first["splits"]["held_out"]


def test_inconclusive_and_error_cases_are_not_counted_as_true_negatives() -> None:
    result = evaluate_independent_corpus()
    assert result["overall"]["inconclusive"] >= 1
    assert result["overall"]["errors"] >= 1
    assert result["overall"]["true_negatives"] < result["overall"]["determinate_total"]
    assert result["overall"]["inconclusive"] + result["overall"]["errors"] >= 2
    assert result["overall"]["inconclusive_excluded_from_metrics"] is True


def test_confirmed_findings_have_boundary_evidence_and_no_status_only_confirmation() -> None:
    result = evaluate_independent_corpus()
    confirmed = [case for case in result["cases"] if case["predicted_outcome"] == "VULNERABLE"]
    assert confirmed
    for case in confirmed:
        assert case["evidence_errors"] == []
        assert case["evidence_quality"] == "observed_authorization_comparison"
        assert case["status_only_proof"] is False
        assert case["evidence_count"] >= 1


def test_results_include_false_positive_false_negative_and_inconclusive_accounting() -> None:
    result = evaluate_independent_corpus()
    overall = result["overall"]
    assert {"true_positives", "false_positives", "true_negatives", "false_negatives"} <= set(overall)
    assert {"inconclusive", "errors", "overclaims", "unresolved_expected_vulnerable"} <= set(overall)
    assert result["per_class"]
    assert all("expected_outcome" in case and "predicted_outcome" in case for case in result["cases"])
    json.dumps(result, sort_keys=True)


def test_case_order_and_held_out_variants_do_not_change_aggregate_results() -> None:
    samples = build_independent_corpus()
    forward = evaluate_independent_corpus(samples)
    reverse = evaluate_independent_corpus(tuple(reversed(samples)))
    assert forward["overall"] == reverse["overall"]
    assert forward["splits"] == reverse["splits"]
    by_id = {item["case_id"]: item["predicted_outcome"] for item in forward["cases"]}
    assert by_id["H-001"] == "VULNERABLE"  # changed resource identifier
    assert by_id["H-002"] == "SAFE"  # changed path, preserved denial
    assert by_id["H-003"] == "SAFE"  # response formatting/public summary only
    assert by_id["V-003"] == "INCONCLUSIVE"  # unavailable evidence is not a negative


def test_evidence_validator_rejects_status_only_or_missing_identity_resource() -> None:
    sample = next(sample for sample in build_independent_corpus() if sample.oracle.expected_outcome == "VULNERABLE")
    findings, _, _ = IDORBOLAEngine(sample.scope).scan(sample.fixture)
    assert findings
    assert validate_confirmed_evidence(findings[0]) == []
