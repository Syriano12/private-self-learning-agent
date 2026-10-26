from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from private_agent.bugbounty.programs import ProgramAnalysisRepository, analyze_program
from private_agent.bugbounty.passive_research import PassiveResearchAdapter
from private_agent.bugbounty.research import ResearchStageRepository, ResearchStageStatus, ResearchStageType, StageBudget
from private_agent.bugbounty.stage_execution import ResearchStageExecutor
import private_agent.storage as storage_module
from private_agent.storage import Store
from private_agent.worker import BackgroundWorker
from tests.test_program_analysis import scope_payload
from tests.test_phase12_api import APITestTool, make_service


def test_background_worker_completes_local_program_analysis_stage_without_network(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.PROGRAM_ANALYSIS,
        owner_user_id="legacy-owner",
    )
    worker = BackgroundWorker(service.orchestrator, worker_id="stage-worker-local", idle_poll_interval_seconds=0.01)
    result = worker.run_once()
    assert result["status"] == "STAGE_COMPLETED"
    assert result["stage_id"] == stage.stage_id
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None
    assert persisted.stage_status == ResearchStageStatus.COMPLETED.value
    assert persisted.result["analysis_id"] == analysis.analysis_id
    assert persisted.result["scope_status"] == "READY_FOR_SCOPE_REVIEW"
    store.close()


def test_background_worker_performs_bounded_passive_research_with_source_provenance(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    calls: list[tuple[str, int]] = []

    def fake_fetch(url: str, *, max_chars: int) -> dict[str, object]:
        calls.append((url, max_chars))
        return {"url": url, "status": 200, "title": "Public policy", "text": "scope /api/v1 and reporting policy"}

    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.PASSIVE_RESEARCH,
        owner_user_id="legacy-owner",
        budget=StageBudget(max_items=4, timeout_seconds=30, max_output_bytes=8_000),
        input_payload={"sources": [{"source_type": "hackerone_security", "url": "https://hackerone.com/security"}]},
    )
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="stage-worker-passive",
        stage_executor=ResearchStageExecutor(passive_adapter=PassiveResearchAdapter(fetcher=fake_fetch)),
    )
    result = worker.run_once()
    assert result["status"] == "STAGE_COMPLETED"
    assert calls == [("https://hackerone.com/security", 8_000)]
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None and persisted.stage_status == ResearchStageStatus.COMPLETED.value
    assert persisted.result["kind"] == "PASSIVE_RESEARCH"
    assert persisted.result["network_accessed"] is True
    assert persisted.result["retrievals"][0]["source_type"] == "hackerone_security"
    assert persisted.result["retrievals"][0]["status_code"] == 200
    assert persisted.result["retrievals"][0]["content_sha256"]
    assert persisted.result["retrievals"][0]["provenance"]["authorization_basis"] == "public_source_only"
    store.close()


def test_passive_source_unavailable_is_fail_closed_without_local_fallback(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )

    def unavailable_fetch(url: str, *, max_chars: int) -> dict[str, object]:
        del url, max_chars
        return {"status": "error", "error": "fetch_failed:network_blocked"}

    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.PASSIVE_RESEARCH,
        owner_user_id="legacy-owner",
        input_payload={"sources": [{"source_type": "hackerone_security", "url": "https://hackerone.com/security"}]},
    )
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="stage-worker-passive-blocked",
        stage_executor=ResearchStageExecutor(passive_adapter=PassiveResearchAdapter(fetcher=unavailable_fetch)),
    )
    result = worker.run_once()
    assert result["status"] == "STAGE_UNAVAILABLE"
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None
    assert persisted.stage_status == ResearchStageStatus.UNAVAILABLE.value
    assert persisted.error_code == "passive_sources_unavailable"
    assert persisted.result["network_accessed"] is True
    assert persisted.result["retrievals"][0]["status_code"] is None
    store.close()


def test_hypothesis_stage_uses_passive_source_evidence_without_confirming_finding(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    passive_result = {
        "kind": "PASSIVE_RESEARCH",
        "network_accessed": True,
        "retrievals": [
            {
                "source_type": "hackerone_security",
                "source_url": "https://hackerone.com/security",
                "content_sha256": "a" * 64,
                "text_excerpt": "documented API route /api/v1/orders/{order_id}",
            }
        ],
    }
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.HYPOTHESIS_GENERATION,
        owner_user_id="legacy-owner",
        input_payload={
            "artifacts": [
                {"name": "passive-result.json", "content_type": "application/json", "content": json.dumps(passive_result)}
            ]
        },
    )
    result = BackgroundWorker(service.orchestrator, worker_id="stage-worker-hypothesis-passive").run_once()
    assert result["status"] == "STAGE_COMPLETED"
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None
    assert persisted.result["network_accessed"] is True
    assert persisted.result["hypotheses"]
    assert persisted.result["hypotheses"][0]["status"] == "UNVERIFIED"
    assert "public_source_is_not_authorization" in persisted.result["limitations"]
    store.close()


def test_report_preparation_preserves_public_source_provenance_as_unverified(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    passive_result = {
        "kind": "PASSIVE_RESEARCH",
        "network_accessed": True,
        "retrievals": [
            {
                "source_type": "hackerone_security",
                "source_url": "https://hackerone.com/security",
                "retrieved_at": "2026-10-26T00:00:00+00:00",
                "status_code": 200,
                "content_sha256": "b" * 64,
                "text_excerpt": "documented API route /api/v1/orders/{order_id}",
            }
        ],
    }
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.REPORT_PREPARATION,
        owner_user_id="legacy-owner",
        input_payload={
            "artifacts": [
                {"name": "passive-result.json", "content_type": "application/json", "content": json.dumps(passive_result)}
            ]
        },
    )
    result = BackgroundWorker(service.orchestrator, worker_id="stage-worker-report-passive").run_once()
    assert result["status"] == "STAGE_COMPLETED"
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None
    assert persisted.result["report_status"] == "PUBLIC_PASSIVE_UNVERIFIED"
    assert persisted.result["network_accessed"] is True
    assert persisted.result["finding_confirmation_allowed"] is False
    assert persisted.result["source_retrievals"][0]["content_sha256"] == "b" * 64
    assert "/api/v1/orders/{order_id}" in persisted.result["observed_endpoints"]
    store.close()


@pytest.mark.parametrize(
    "stage_type",
    [
        ResearchStageType.INVENTORY,
        ResearchStageType.PASSIVE_RESEARCH,
        ResearchStageType.HYPOTHESIS_GENERATION,
        ResearchStageType.VERIFICATION,
        ResearchStageType.REPORT_PREPARATION,
    ],
)
def test_background_worker_marks_external_stage_unavailable_without_fallback(tmp_path: Path, stage_type: ResearchStageType) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        stage_type,
        owner_user_id="legacy-owner",
    )
    worker = BackgroundWorker(service.orchestrator, worker_id="stage-worker-blocked", idle_poll_interval_seconds=0.01)
    result = worker.run_once()
    assert result["status"] == "STAGE_UNAVAILABLE"
    assert result["error_code"] == "external_runtime_unavailable"
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None and persisted.stage_status == ResearchStageStatus.UNAVAILABLE.value
    assert persisted.result == {}
    store.close()


def test_stage_claim_recovery_marks_expired_running_stage_unknown_without_retry(tmp_path: Path) -> None:
    store = Store(tmp_path / "stage-lease.sqlite3")
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.PROGRAM_ANALYSIS,
        owner_user_id="legacy-owner",
    )
    claim = ResearchStageRepository(store).claim_next("worker-a", lease_ttl_seconds=1)
    assert claim is not None and claim["stage_id"] == stage.stage_id
    with store._db_lock, store.db:
        expired = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        store.db.execute(
            "UPDATE bugbounty_research_stages SET lease_expires_at = ? WHERE stage_id = ?",
            (expired, stage.stage_id),
        )
    next_claim = ResearchStageRepository(store).claim_next("worker-b", lease_ttl_seconds=1)
    assert next_claim is None
    recovered = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert recovered is not None
    assert recovered.stage_status == ResearchStageStatus.UNKNOWN.value
    assert recovered.error_code == "stage_lease_expired"
    raw = store.db.execute(
        "SELECT lease_id, worker_id, lease_expires_at FROM bugbounty_research_stages WHERE stage_id = ?",
        (stage.stage_id,),
    ).fetchone()
    assert raw is not None and tuple(raw) == ("", "", "")
    store.close()


def test_stage_claim_is_atomic_under_two_worker_threads(tmp_path: Path) -> None:
    store = Store(tmp_path / "stage-race.sqlite3")
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.PROGRAM_ANALYSIS,
        owner_user_id="legacy-owner",
    )
    repository = ResearchStageRepository(store)
    results: list[dict | None] = []
    lock = threading.Lock()

    def claim(worker_id: str) -> None:
        value = repository.claim_next(worker_id, lease_ttl_seconds=10)
        with lock:
            results.append(value)

    threads = [threading.Thread(target=claim, args=(f"worker-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    claimed = [value for value in results if value is not None]
    assert len(claimed) == 1
    assert claimed[0]["stage_id"] == stage.stage_id
    store.close()


def test_existing_schema9_database_upgrades_stage_lease_columns(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "schema9.sqlite3"
    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 9)
    legacy = Store(path)
    legacy.close()
    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 10)
    upgraded = Store(path)
    versions = [row["schema_version"] for row in upgraded.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")]
    assert versions[-1] == 10
    columns = {row["name"] for row in upgraded.db.execute("PRAGMA table_info(bugbounty_research_stages)")}
    assert {"lease_id", "lease_expires_at", "attempt_count"}.issubset(columns)
    upgraded.close()


def test_existing_schema10_database_upgrades_stage_input_json(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "schema10.sqlite3"
    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 10)
    legacy = Store(path)
    legacy.close()
    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 11)
    upgraded = Store(path)
    versions = [row["schema_version"] for row in upgraded.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")]
    assert versions[-1] == 11
    columns = {row["name"] for row in upgraded.db.execute("PRAGMA table_info(bugbounty_research_stages)")}
    assert "input_json" in columns
    upgraded.close()


def test_inventory_stage_analyzes_bounded_local_artifacts_without_network(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.INVENTORY,
        owner_user_id="legacy-owner",
        input_payload={
            "artifacts": [
                {
                    "name": "openapi.json",
                    "content_type": "application/json",
                    "content": json.dumps({"openapi": "3.0.0", "paths": {"/users": {"get": {}}, "/admin": {"post": {}}}}),
                },
                {
                    "name": "app.js",
                    "content_type": "text/javascript",
                    "content": "const token = 'secret-token'; fetch('/api/users');",
                },
            ]
        },
    )
    result = BackgroundWorker(service.orchestrator, worker_id="stage-worker-inventory").run_once()
    assert result["status"] == "STAGE_COMPLETED"
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None and persisted.stage_status == ResearchStageStatus.COMPLETED.value
    assert "/users" in persisted.result["endpoints"]
    assert "secret-token" not in json.dumps(persisted.result)
    assert persisted.result["network_accessed"] is False
    store.close()


def test_inventory_stage_fits_result_inside_output_budget(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.INVENTORY,
        owner_user_id="legacy-owner",
        budget=StageBudget(max_items=10_000, timeout_seconds=10, max_output_bytes=512),
        input_payload={
            "artifacts": [
                {
                    "name": "routes.js",
                    "content_type": "text/javascript",
                    "content": " ".join(f"fetch('/route-{index}-{'x' * 300}')" for index in range(100)),
                }
            ]
        },
    )
    result = BackgroundWorker(service.orchestrator, worker_id="stage-worker-output-limit").run_once()
    assert result["status"] == "STAGE_COMPLETED"
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None
    assert len(json.dumps(persisted.result, ensure_ascii=False).encode("utf-8")) <= 512
    assert "stage_output_limit_reached" in persisted.result["limitations"]
    store.close()


def test_passive_research_stage_analyzes_local_artifacts_without_network(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.PASSIVE_RESEARCH,
        owner_user_id="legacy-owner",
        input_payload={
            "artifacts": [
                {
                    "name": "openapi.json",
                    "content_type": "application/json",
                    "content": json.dumps({"openapi": "3.0.0", "paths": {"/orders/{id}": {"get": {}}}}),
                }
            ]
        },
    )
    result = BackgroundWorker(service.orchestrator, worker_id="stage-worker-passive-local").run_once()
    assert result["status"] == "STAGE_COMPLETED"
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None
    assert persisted.result["kind"] == "LOCAL_PASSIVE_OBSERVATIONS"
    assert persisted.result["network_accessed"] is False
    assert persisted.result["observations"]
    store.close()


def test_hypothesis_generation_uses_local_evidence_without_confirming_findings(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.HYPOTHESIS_GENERATION,
        owner_user_id="legacy-owner",
        input_payload={
            "artifacts": [
                {
                    "name": "openapi.json",
                    "content_type": "application/json",
                    "content": json.dumps({"openapi": "3.0.0", "paths": {"/orders/{id}": {"get": {}}, "/admin/users": {"get": {}}}}),
                }
            ]
        },
    )
    result = BackgroundWorker(service.orchestrator, worker_id="stage-worker-hypotheses-local").run_once()
    assert result["status"] == "STAGE_COMPLETED"
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None
    assert persisted.result["kind"] == "LOCAL_HYPOTHESES"
    assert persisted.result["network_accessed"] is False
    assert persisted.result["hypotheses"]
    assert all(item["status"] == "UNVERIFIED" for item in persisted.result["hypotheses"])
    assert all(item["evidence"] and item["verification_conditions"] and item["alternative_explanations"] for item in persisted.result["hypotheses"])
    assert all(
        item["reasoning"] and item["confidence"] == 0.35 and item["confidence_justification"]
        and item["safe_verification_plan"]
        for item in persisted.result["hypotheses"]
    )
    assert persisted.result["generation_mode"] == "deterministic_evidence_rules"
    assert persisted.result["llm_inference"]["used"] is False
    assert "llm_inference_not_used" in persisted.result["limitations"]
    assert "confirmed" not in json.dumps(persisted.result).lower()
    store.close()


def test_verification_stage_runs_only_bounded_local_fixtures(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.VERIFICATION,
        owner_user_id="legacy-owner",
        input_payload={
            "artifacts": [
                {
                    "name": "fixture.json",
                    "content_type": "application/json",
                    "content": json.dumps(
                        {
                            "fixture_type": "local_verification",
                            "cases": [
                                {"case_id": "owner-read", "expected": {"status": "allowed"}, "observed": {"status": "allowed"}},
                                {"case_id": "other-owner-read", "expected": {"status": "denied"}, "observed": {"status": "allowed"}},
                            ],
                        }
                    ),
                }
            ]
        },
    )
    result = BackgroundWorker(service.orchestrator, worker_id="stage-worker-verification-local").run_once()
    assert result["status"] == "STAGE_COMPLETED"
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None
    assert persisted.result["kind"] == "LOCAL_FIXTURE_VERIFICATION"
    assert persisted.result["network_accessed"] is False
    assert persisted.result["finding_confirmation_allowed"] is False
    assert [case["matched"] for case in persisted.result["cases"]] == [True, False]
    store.close()


def test_report_preparation_builds_bounded_local_report_from_stage_evidence(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    inventory = {"kind": "LOCAL_INVENTORY", "network_accessed": False, "endpoints": ["GET /orders/{id}"], "urls_observed": [], "packages": [], "files": [], "limitations": []}
    hypotheses = {"kind": "LOCAL_HYPOTHESES", "network_accessed": False, "hypotheses": [{"hypothesis_id": "HYP-LOCAL-1", "category": "IDOR_BOLA", "title": "Dynamic resource authorization review", "status": "UNVERIFIED", "evidence": [{"artifact_name": "openapi.json", "content_sha256": "a" * 64, "observation": "GET /orders/{id}"}], "verification_conditions": ["fixture"], "alternative_explanations": ["middleware"]}], "limitations": ["hypotheses_are_unverified"], "artifact_count": 1}
    verification = {"kind": "LOCAL_FIXTURE_VERIFICATION", "network_accessed": False, "finding_confirmation_allowed": False, "cases": [{"case_id": "owner-read", "matched": True, "expected_sha256": "b" * 64, "observed_sha256": "b" * 64}], "limitations": ["local_fixture_only"], "fixture_artifact_count": 1}
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.REPORT_PREPARATION,
        owner_user_id="legacy-owner",
        input_payload={
            "artifacts": [
                {"name": "inventory.json", "content_type": "application/json", "content": json.dumps(inventory)},
                {"name": "hypotheses.json", "content_type": "application/json", "content": json.dumps(hypotheses)},
                {"name": "verification.json", "content_type": "application/json", "content": json.dumps(verification)},
            ]
        },
    )
    result = BackgroundWorker(service.orchestrator, worker_id="stage-worker-report-local").run_once()
    assert result["status"] == "STAGE_COMPLETED"
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None
    assert persisted.result["kind"] == "LOCAL_REPORT"
    assert persisted.result["network_accessed"] is False
    assert "GET /orders/{id}" in persisted.result["report_markdown"]
    assert "HYP-LOCAL-1" in persisted.result["report_markdown"]
    assert "UNVERIFIED" in persisted.result["report_markdown"]
    assert "finding_confirmation_allowed=false" in persisted.result["report_markdown"]
    store.close()


def test_report_budget_does_not_keep_items_removed_from_structured_result(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    analysis = ProgramAnalysisRepository(store).save(
        analyze_program("https://example.com/policy", json.dumps(scope_payload())),
        owner_user_id="legacy-owner",
    )
    hypotheses = {
        "kind": "LOCAL_HYPOTHESES",
        "network_accessed": False,
        "hypotheses": [
            {"hypothesis_id": f"HYP-LOCAL-{index}", "category": "API_SURFACE_REVIEW", "title": "x" * 120, "status": "UNVERIFIED", "evidence": []}
            for index in range(20)
        ],
        "limitations": [],
    }
    stage = ResearchStageRepository(store).create(
        analysis.analysis_id,
        ResearchStageType.REPORT_PREPARATION,
        owner_user_id="legacy-owner",
        budget=StageBudget(max_items=100, timeout_seconds=10, max_output_bytes=768),
        input_payload={"artifacts": [{"name": "hypotheses.json", "content_type": "application/json", "content": json.dumps(hypotheses)}]},
    )
    assert BackgroundWorker(service.orchestrator, worker_id="stage-worker-report-budget").run_once()["status"] == "STAGE_COMPLETED"
    persisted = ResearchStageRepository(store).get(stage.stage_id, owner_user_id="legacy-owner")
    assert persisted is not None
    assert len(json.dumps(persisted.result, ensure_ascii=False).encode("utf-8")) <= 768
    structured_ids = {str(item.get("hypothesis_id")) for item in persisted.result.get("hypotheses", [])}
    for index in range(20):
        hypothesis_id = f"HYP-LOCAL-{index}"
        if hypothesis_id in persisted.result.get("report_markdown", ""):
            assert hypothesis_id in structured_ids
    store.close()
