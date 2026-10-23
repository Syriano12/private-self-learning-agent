from __future__ import annotations

import json

from private_agent.bugbounty.cross_application import (
    build_cross_application_corpus,
    evaluate_cross_application_corpus,
)


def test_cross_application_corpus_uses_multiple_independent_authorization_models() -> None:
    samples = build_cross_application_corpus()
    application_ids = {sample.application_id for sample in samples}
    assert len(application_ids) >= 4
    assert {sample.oracle.split for sample in samples} == {"train", "validation", "held_out"}
    assert len({type(sample.application).__name__ for sample in samples}) >= 4
    assert all(not hasattr(sample.application, "expected_outcome") for sample in samples)
    assert all(not hasattr(sample.application, "ground_truth") for sample in samples)


def test_cross_application_split_is_by_application_not_individual_case() -> None:
    samples = build_cross_application_corpus()
    splits_by_application: dict[str, set[str]] = {}
    for sample in samples:
        splits_by_application.setdefault(sample.application_id, set()).add(sample.oracle.split)
    assert all(len(splits) == 1 for splits in splits_by_application.values())
    assert len(splits_by_application) >= 4


def test_cross_application_evaluation_is_reproducible_and_reports_each_application() -> None:
    first = evaluate_cross_application_corpus()
    second = evaluate_cross_application_corpus()
    assert first == second
    assert first["external_targets_contacted"] == 0
    assert first["ground_truth_accessible_to_engine"] is False
    assert first["held_out_applications"]
    assert len(first["applications"]) >= 4
    assert set(first["per_application"]) == set(first["applications"])
    assert all(item["case_count"] >= 1 for item in first["per_application"].values())


def test_cross_application_metrics_do_not_hide_false_positives_or_inconclusive_cases() -> None:
    result = evaluate_cross_application_corpus()
    overall = result["overall"]
    assert result["phase20_baseline"]["false_positives"] == 2
    assert overall["false_positives"] == 0
    assert overall["inconclusive"] >= 1
    assert overall["inconclusive_excluded_from_metrics"] is True
    assert overall["errors"] == 0
    assert result["held_out"]["false_positives"] == 0
    assert result["held_out"]["precision"] == 1.0
    json.dumps(result, sort_keys=True)


def test_cross_application_findings_have_observed_boundary_evidence() -> None:
    result = evaluate_cross_application_corpus()
    confirmed = [case for case in result["cases"] if case["predicted_outcome"] == "VULNERABLE"]
    assert confirmed
    assert all(case["evidence_quality"] == "observed_authorization_comparison" for case in confirmed)
    assert all(case["evidence_errors"] == [] for case in confirmed)
    assert all(case["status_only_proof"] is False for case in confirmed)


def test_cross_application_reports_application_specific_limitations() -> None:
    result = evaluate_cross_application_corpus()
    assert result["application_specific_limitations"]
    assert any("owner" in item.lower() or "tenant" in item.lower() for item in result["application_specific_limitations"])
    assert result["verdict"] in {
        "GENERALIZATION DEMONSTRATED ON LIMITED LOCAL APPLICATIONS",
        "PARTIAL GENERALIZATION",
        "GENERALIZATION NOT DEMONSTRATED",
    }
