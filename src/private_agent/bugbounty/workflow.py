"""End-to-end bounded Bug Bounty workflow for authorized local research."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from private_agent.bugbounty.engines import AttackSurfaceInventory, DeterministicPlanner, IDORBOLAEngine, PassiveRecon, StaticSecurityAnalyzer
from private_agent.bugbounty.fixtures import DeterministicFixtureApp
from private_agent.bugbounty.models import BugBountyRun, Finding, FindingState, ScopeDefinition, public_reference, reconcile_bug_bounty_metrics, stable_digest, utc_now, validate_confirmed_evidence


class BugBountyRepository:
    """Bounded additive persistence on the existing Store connection."""

    def __init__(self, store: Any) -> None:
        if store is None or not hasattr(store, "db") or not hasattr(store, "_db_lock"):
            raise ValueError("bugbounty_store_required")
        self.store = store
        with self.store._db_lock, self.store.db:
            self.store.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS bugbounty_runs (
                  run_id TEXT PRIMARY KEY, program_id TEXT NOT NULL, status TEXT NOT NULL,
                  mode TEXT NOT NULL, scope_json TEXT NOT NULL, inventory_json TEXT NOT NULL,
                  plan_json TEXT NOT NULL, metrics_json TEXT NOT NULL, limitations_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bugbounty_findings (
                  finding_id TEXT NOT NULL, run_id TEXT NOT NULL, state TEXT NOT NULL,
                  vulnerability_class TEXT NOT NULL, severity TEXT NOT NULL,
                  finding_json TEXT NOT NULL, created_at TEXT NOT NULL,
                  PRIMARY KEY(run_id, finding_id),
                  FOREIGN KEY(run_id) REFERENCES bugbounty_runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_bugbounty_findings_run ON bugbounty_findings(run_id, state);
                CREATE TABLE IF NOT EXISTS bugbounty_patterns (
                  pattern_id TEXT PRIMARY KEY, finding_id TEXT NOT NULL, vulnerability_class TEXT NOT NULL,
                  status TEXT NOT NULL, pattern_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bugbounty_pattern_feedback (
                  feedback_id TEXT PRIMARY KEY, pattern_id TEXT NOT NULL, label TEXT NOT NULL,
                  actor TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL,
                  FOREIGN KEY(pattern_id) REFERENCES bugbounty_patterns(pattern_id)
                );
                CREATE INDEX IF NOT EXISTS idx_bugbounty_patterns_class ON bugbounty_patterns(vulnerability_class, status);
                """
            )
            columns = {str(row["name"]) for row in self.store.db.execute("PRAGMA table_info(bugbounty_runs)").fetchall()}
            if "tests_json" not in columns:
                self.store.db.execute("ALTER TABLE bugbounty_runs ADD COLUMN tests_json TEXT NOT NULL DEFAULT '[]'")

    def save(self, run: BugBountyRun) -> None:
        payload = run.to_dict(public=True)
        with self.store._db_lock, self.store.db:
            self.store.db.execute(
                """INSERT OR REPLACE INTO bugbounty_runs
                   (run_id, program_id, status, mode, scope_json, inventory_json,
                    plan_json, metrics_json, limitations_json, tests_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run.run_id,
                    run.scope.program_id,
                    run.status,
                    run.mode,
                    json.dumps(payload["scope"], ensure_ascii=False, sort_keys=True),
                    json.dumps(payload["inventory"], ensure_ascii=False, sort_keys=True),
                    json.dumps(payload["plan"], ensure_ascii=False, sort_keys=True),
                    json.dumps(payload["metrics"], ensure_ascii=False, sort_keys=True),
                    json.dumps(payload["limitations"], ensure_ascii=False, sort_keys=True),
                    json.dumps(payload["tests"], ensure_ascii=False, sort_keys=True),
                    run.created_at,
                ),
            )
            self.store.db.execute("DELETE FROM bugbounty_findings WHERE run_id = ?", (run.run_id,))
            for finding in run.findings[:100]:
                self.store.db.execute(
                    """INSERT INTO bugbounty_findings
                       (finding_id, run_id, state, vulnerability_class, severity, finding_json, created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        finding.finding_id,
                        run.run_id,
                        finding.state.value,
                        finding.vulnerability_class,
                        finding.severity.value,
                        json.dumps(finding.to_dict(public=True), ensure_ascii=False, sort_keys=True),
                        run.created_at,
                    ),
                )
                if finding.state in {FindingState.CONFIRMED, FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED}:
                    pattern_id = "PATTERN-" + stable_digest([run.run_id, finding.finding_id])[:16].upper()
                    pattern = {
                        "pattern_id": pattern_id,
                        "finding_id": finding.finding_id,
                        "vulnerability_class": finding.vulnerability_class,
                        "preconditions": finding.preconditions[:20],
                        "reproduction_steps": finding.reproduction_steps[:20],
                        "regression": finding.regression,
                        "provenance": {"run_id": run.run_id, "scope_reference": public_reference(finding.scope_reference)},
                    }
                    self.store.db.execute(
                        """INSERT OR REPLACE INTO bugbounty_patterns
                           (pattern_id, finding_id, vulnerability_class, status, pattern_json, created_at)
                           VALUES (?,?,?,?,?,?)""",
                        (pattern_id, finding.finding_id, finding.vulnerability_class, "ACTIVE", json.dumps(pattern, ensure_ascii=False, sort_keys=True), run.created_at),
                    )

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self.store._db_lock:
            row = self.store.db.execute("SELECT * FROM bugbounty_runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            findings = self.store.db.execute("SELECT finding_json FROM bugbounty_findings WHERE run_id = ? ORDER BY finding_id", (run_id,)).fetchall()
        result = dict(row)
        for key in ("scope_json", "inventory_json", "plan_json", "metrics_json", "limitations_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        raw_tests = result.pop("tests_json", "[]")
        try:
            parsed_tests = json.loads(raw_tests)
        except (TypeError, ValueError):
            parsed_tests = []
        result["tests"] = [item for item in parsed_tests[:100] if isinstance(item, dict)] if isinstance(parsed_tests, list) else []
        loaded_findings: list[dict[str, Any]] = []
        for item in findings:
            raw_finding = json.loads(item["finding_json"])
            try:
                loaded_findings.append(Finding.from_dict(raw_finding).to_dict(public=True))
            except (TypeError, ValueError) as exc:
                loaded_findings.append({
                    **raw_finding,
                    "state": FindingState.HYPOTHESIS.value,
                    "validation": {
                        "status": "REJECTED",
                        "errors": [f"persisted_finding_deserialization_failed:{type(exc).__name__}"],
                        "evidence_sha256": "",
                        "validator": "idor_bola_semantic_v1",
                    },
                })
        result["findings"] = loaded_findings
        result["metrics"] = reconcile_bug_bounty_metrics(result.get("metrics"), loaded_findings, result["tests"])
        return result

    def list_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValueError("bugbounty_limit_invalid")
        with self.store._db_lock:
            rows = self.store.db.execute("SELECT run_id, program_id, status, mode, metrics_json, created_at FROM bugbounty_runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "program_id": row["program_id"],
                "status": row["status"],
                "mode": row["mode"],
                "metrics": json.loads(row["metrics_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def search_patterns(self, *, vulnerability_class: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValueError("bugbounty_pattern_limit_invalid")
        query = "SELECT * FROM bugbounty_patterns WHERE status = 'ACTIVE'"
        params: list[Any] = []
        if vulnerability_class:
            query += " AND vulnerability_class = ?"
            params.append(vulnerability_class)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.store._db_lock:
            rows = self.store.db.execute(query, tuple(params)).fetchall()
        return [{**dict(row), "pattern": json.loads(row["pattern_json"])} for row in rows]

    def regression_template(self, pattern_id: str) -> dict[str, Any] | None:
        with self.store._db_lock:
            row = self.store.db.execute("SELECT pattern_json, status FROM bugbounty_patterns WHERE pattern_id = ?", (pattern_id,)).fetchone()
        if row is None or row["status"] != "ACTIVE":
            return None
        pattern = json.loads(row["pattern_json"])
        return {"pattern_id": pattern_id, "kind": "non_executable_regression_template", "steps": pattern.get("reproduction_steps", []), "preconditions": pattern.get("preconditions", []), "regression": pattern.get("regression", "")}

    def record_feedback(self, pattern_id: str, *, label: str, actor: str, reason: str) -> None:
        if label not in {"FALSE_POSITIVE", "CONFIRMED_USEFUL", "REJECTED"}:
            raise ValueError("bugbounty_feedback_label_invalid")
        if not actor or len(actor) > 256 or not reason or len(reason) > 1000:
            raise ValueError("bugbounty_feedback_metadata_invalid")
        feedback_id = "FEEDBACK-" + stable_digest([pattern_id, label, actor, reason, utc_now()])[:16].upper()
        with self.store._db_lock, self.store.db:
            row = self.store.db.execute("SELECT 1 FROM bugbounty_patterns WHERE pattern_id = ?", (pattern_id,)).fetchone()
            if row is None:
                raise ValueError("bugbounty_pattern_not_found")
            self.store.db.execute(
                "INSERT INTO bugbounty_pattern_feedback(feedback_id, pattern_id, label, actor, reason, created_at) VALUES (?,?,?,?,?,?)",
                (feedback_id, pattern_id, label, actor, reason, utc_now()),
            )
            if label in {"FALSE_POSITIVE", "REJECTED"}:
                self.store.db.execute("UPDATE bugbounty_patterns SET status = 'DISABLED' WHERE pattern_id = ?", (pattern_id,))


@dataclass
class BugBountyAgent:
    store: Any | None = None
    recon: PassiveRecon = PassiveRecon()
    planner: DeterministicPlanner = DeterministicPlanner()
    static_analyzer: StaticSecurityAnalyzer = StaticSecurityAnalyzer()

    def run(
        self,
        scope_payload: Mapping[str, Any],
        *,
        fixture_variant: str | None = None,
        urls: tuple[str, ...] = (),
        openapi: Mapping[str, Any] | None = None,
        source_documents: Mapping[str, str] | None = None,
        mode: str = "deterministic",
        persist: bool = True,
    ) -> BugBountyRun:
        if mode != "deterministic":
            raise ValueError("only_deterministic_provider_is_available")
        scope = ScopeDefinition.from_dict(scope_payload)
        fixture = DeterministicFixtureApp(fixture_variant) if fixture_variant else None
        inventory = self.recon.analyze(urls=urls, openapi=openapi, source_documents=source_documents, fixture=fixture)
        plan = self.planner.build(inventory, scope)
        findings = []
        tests = []
        limitations = list(inventory.limitations)
        if fixture is not None:
            if scope.active_testing:
                fixture_findings, fixture_tests, fixture_limits = IDORBOLAEngine(scope).scan(fixture)
                findings.extend(fixture_findings)
                tests.extend(fixture_tests)
                limitations.extend(fixture_limits)
            else:
                limitations.append("active_fixture_testing_not_run_without_explicit_active_scope")
        if source_documents:
            findings.extend(self.static_analyzer.analyze(source_documents, scope.authorization_reference))
        if not fixture and not source_documents and not urls and openapi is None:
            limitations.append("no_local_target_artifact_or_fixture_supplied")
        for finding in findings:
            if finding.state in {FindingState.CONFIRMED, FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED}:
                errors = validate_confirmed_evidence(finding)
                if errors:
                    finding._invalidate_confirmation(errors)
                    limitations.append(f"finding_confirmation_rejected:{finding.finding_id}")
        status = "COMPLETED" if not limitations else "COMPLETED_WITH_LIMITATIONS"
        inventory_payload = inventory.to_dict()
        run_identity_inventory = {
            "assets": inventory_payload["assets"][:25],
            "endpoints": inventory_payload["endpoints"][:25],
            "observation_ids": [item["observation_id"] for item in inventory_payload["observations"][:25]],
            "limitations": inventory_payload["limitations"][:25],
        }
        run_id = "RUN-" + stable_digest({"scope": scope.to_dict(), "mode": mode, "fixture": fixture_variant, "inventory": run_identity_inventory})[:16].upper()
        run = BugBountyRun(run_id, scope, status, mode, inventory_payload, plan, findings, tests, sorted(set(limitations)))
        if persist and self.store is not None:
            BugBountyRepository(self.store).save(run)
        return run

    def run_benchmark(self, *, fixture_variant: str, store: Any | None = None) -> BugBountyRun:
        now = datetime.now(timezone.utc)
        scope = {
            "program_id": "local-benchmark",
            "authorization_reference": "fixture-authorization-local-only",
            "issued_at": now.isoformat(),
            "expires_at": (now.replace(year=now.year + 1)).isoformat(),
            "assets": [{"asset_id": "fixture-api", "host": "fixture.local", "path_prefix": "/api/v1", "methods": ["GET"], "active_allowed": True}],
            "permitted_identities": ["alice", "bob", "admin"],
            "permitted_resources": ["order-alice", "order-bob", "comment-bob", "report-admin"],
            "permitted_methods": ["GET"],
            "prohibited_methods": ["DELETE", "PATCH", "PUT", "POST"],
            "active_testing": True,
        }
        agent = BugBountyAgent(store=store or self.store)
        return agent.run(scope, fixture_variant=fixture_variant, persist=bool(store or self.store))


def markdown_report(run: BugBountyRun) -> str:
    # Force the same production serialization gate used by persistence/API output.
    run.to_dict(public=True)
    lines = [
        f"# Bug Bounty Research Run {run.run_id}",
        "",
        f"- Status: `{run.status}`",
        f"- Program: `{run.scope.program_id}`",
        f"- Authorization reference digest: `{public_reference(run.scope.authorization_reference)}`",
        f"- Mode: `{run.mode}`",
        "- External targets: `none`",
        "",
        "## Scope and safety",
        "",
        "Active requests were permitted only by the explicit scope definition and were executed against the deterministic in-memory fixture. Redirects, discovered hosts, third-party services, and caller assumptions never expand scope.",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(run.metrics(), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Findings",
        "",
    ]
    if not run.findings:
        lines.append("No findings were confirmed by the supplied evidence.")
    for finding in run.findings:
        lines.extend([
            f"### {finding.finding_id}: {finding.title}",
            "",
            f"- Class: `{finding.vulnerability_class}`",
            f"- State: `{finding.state.value}`",
            f"- Severity: `{finding.severity.value}`",
            f"- Confidence: `{finding.confidence:.2f}`",
            f"- Asset/endpoint: `{finding.asset}{finding.endpoint}`",
            f"- Expected: {finding.expected_behavior}",
            f"- Observed: {finding.observed_behavior}",
            f"- Impact: {finding.impact}",
            f"- Remediation: {finding.remediation}",
            "",
            "#### Reproduction",
            "",
            *[f"{index}. {step}" for index, step in enumerate(finding.reproduction_steps, 1)],
            "",
            "#### Limitations",
            "",
            *[f"- {item}" for item in finding.limitations],
            "",
        ])
    lines.extend(["## Plan", "", "```json", json.dumps(run.plan, ensure_ascii=False, indent=2), "```", "", "## Limitations", "", *[f"- {item}" for item in run.limitations], ""])
    return "\n".join(lines)


def evaluate_local_benchmark() -> dict[str, Any]:
    """Run the committed deterministic corpus and return measured results."""
    expected = {
        "secure": 0,
        "idor_horizontal": 1,
        "idor_vertical": 1,
        "idor_nested": 1,
        "false_positive": 0,
        "inconclusive": 0,
    }
    cases: list[dict[str, Any]] = []
    for variant, expected_findings in expected.items():
        run = BugBountyAgent().run(
            {
                "program_id": "local-benchmark",
                "authorization_reference": "local-benchmark-fixture",
                "issued_at": "2025-01-01T00:00:00+00:00",
                "expires_at": "2099-01-01T00:00:00+00:00",
                "assets": [{"asset_id": "fixture-api", "host": "fixture.local", "path_prefix": "/api/v1", "methods": ["GET"], "active_allowed": True}],
                "permitted_identities": ["alice", "bob", "admin"],
                "permitted_resources": ["order-alice", "order-bob", "comment-bob", "report-admin"],
                "permitted_methods": ["GET"],
                "prohibited_methods": ["DELETE", "PATCH", "PUT", "POST"],
                "active_testing": True,
            },
            fixture_variant=variant,
            persist=False,
        )
        confirmed = run.metrics()["confirmed_findings"]
        cases.append({"fixture": variant, "expected_confirmed": expected_findings, "actual_confirmed": confirmed, "passed": confirmed == expected_findings, "metrics": run.metrics()})
    passed = sum(1 for case in cases if case["passed"])
    true_positives = sum(
        1 for case in cases if case["expected_confirmed"] > 0 and case["actual_confirmed"] > 0
    )
    false_positives = sum(
        1 for case in cases if case["expected_confirmed"] == 0 and case["actual_confirmed"] > 0
    )
    false_negatives = sum(
        1 for case in cases if case["expected_confirmed"] > 0 and case["actual_confirmed"] == 0
    )
    true_negatives = sum(
        1 for case in cases if case["expected_confirmed"] == 0 and case["actual_confirmed"] == 0
    )
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    return {
        "cases": cases,
        "passed": passed,
        "total": len(cases),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": true_positives / precision_denominator if precision_denominator else None,
        "recall": true_positives / recall_denominator if recall_denominator else None,
        "recall_note": "Precision and recall are measured only against this explicit local fixture ground truth; arbitrary external-target recall is not measured.",
        "external_targets_contacted": 0,
    }
