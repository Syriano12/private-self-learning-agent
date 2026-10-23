from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from private_agent.security import ActionRequest, OwnerPermissionRule, SecurityController
from private_agent.storage import Store, UnsupportedSchemaVersion

from tests.test_phase12_api import (
    APITestTool,
    auth_headers,
    client_for,
    create_high_task,
    make_service,
)


def network_action() -> ActionRequest:
    return ActionRequest(
        task_id="phase14a2-task",
        action_id="phase14a2-action",
        tool_or_skill="network",
        requested_capabilities=["NETWORK_ACCESS"],
        declared_capabilities=["NETWORK_ACCESS"],
    )


def deny_rule(rule_id: str = "phase14a2-deny") -> OwnerPermissionRule:
    return OwnerPermissionRule(rule_id, "DENY", "TOOL", "network", reason=f"reason:{rule_id}")


def test_execution_and_blocked_audit_use_effective_owner_policy_version(tmp_path: Path) -> None:
    store = Store(tmp_path / "audit-version.sqlite3")
    controller = SecurityController(store=store)
    action = ActionRequest(
        task_id="phase14a2-task",
        action_id="phase14a2-action",
        tool_or_skill="local",
        requested_capabilities=[],
        declared_capabilities=[],
    )

    store.create_owner_permission_rule(deny_rule(), actor="owner", reason="create deny")
    authorization = controller.authorize(action, boundary=True)
    assert authorization.allowed is True
    operation = lambda: {"ok": True}
    controller.execute_authorized(
        action,
        operation,
        authorization=authorization,
        operation_binding=controller._bind_operation(action, operation, kind="test"),
    )
    execution = controller.audit.list(task_id=action.task_id)[-1]
    assert execution["event_type"] == "ACTION_EXECUTED"
    assert execution["policy_version"] == 2

    store.disable_owner_permission_rule("phase14a2-deny", actor="owner", reason="disable deny")
    controller.record_blocked(action, reason="blocked by test")
    blocked = controller.audit.list(task_id=action.task_id)[-1]
    assert blocked["event_type"] == "ACTION_BLOCKED"
    assert blocked["policy_version"] == 3
    store.close()


def test_owner_policy_change_then_api_resume_rejects_old_approval_without_side_effect(tmp_path: Path) -> None:
    tool = APITestTool("network", capabilities=["NETWORK_ACCESS"], risk="high")
    service, store, _ = make_service(tmp_path, tool)
    client = client_for(service)
    result = service.orchestrator.run("request a controlled high risk action")
    task_id = str(result["task_id"])

    approval_id = client.get("/api/v1/approvals", headers=auth_headers()).json()[0]["approval_id"]
    approved = client.post(f"/api/v1/approvals/{approval_id}/approve", json={}, headers=auth_headers())
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"

    store.create_owner_permission_rule(deny_rule("resume-deny"), actor="owner", reason="deny before resume")
    assert store.latest_owner_policy_version() == 2

    resumed = client.post(f"/api/v1/tasks/{task_id}/resume", json={}, headers=auth_headers())
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "BLOCKED"
    assert tool.calls == 0
    store.close()


def test_legacy_allow_requires_exact_binding_and_current_version(tmp_path: Path) -> None:
    store = Store(tmp_path / "allow-helper.sqlite3")
    controller = SecurityController(store=store)
    action = network_action()
    decision = controller.preflight(action)
    request = controller.approval_gate.get(decision.approval_id)
    assert request is not None
    controller.approval_gate.approve(request.approval_id)

    assert controller.approval_gate.allow(request) is False
    assert controller.approval_gate.allow(
        request,
        action=action,
        policy_version=store.latest_owner_policy_version(),
        risk_level=decision.risk_level,
    ) is True

    store.create_owner_permission_rule(deny_rule(), actor="owner", reason="invalidate helper approval")
    assert controller.approval_gate.allow(
        request,
        action=action,
        policy_version=store.latest_owner_policy_version(),
        risk_level=decision.risk_level,
    ) is False
    legacy = controller.approval_gate.check("publish")
    assert legacy is not None and legacy.status == "PENDING"
    assert controller.approval_gate.allow(legacy) is False
    store.close()


def test_schema_v3_marker_gap_is_repaired_without_inconsistent_objects(tmp_path: Path) -> None:
    path = tmp_path / "marker-gap.sqlite3"
    store = Store(path)
    store.db.execute("DELETE FROM task_state_schema WHERE schema_version = 2")
    store.db.commit()
    store.close()

    reopened = Store(path)
    versions = [row["schema_version"] for row in reopened.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")]
    assert versions == [1, 2, 3, 4, 5, 6, 7]
    for object_type, name in (
        ("table", "owner_permission_rules"),
        ("table", "owner_policy_audit"),
        ("index", "idx_owner_rules_active_scope"),
        ("index", "idx_owner_rules_expiry"),
        ("table", "owner_policy_state"),
        ("table", "owner_permission_rule_history"),
    ):
        assert reopened.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?", (object_type, name)
        ).fetchone()
    reopened.close()


def test_locked_sqlite_version_allocation_fails_closed_without_partial_data(tmp_path: Path) -> None:
    path = tmp_path / "busy.sqlite3"
    first = Store(path)
    second = Store(path)
    second.db.execute("PRAGMA busy_timeout = 0")
    first.db.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(RuntimeError, match="owner_policy_version_busy"):
            second.create_owner_permission_rule(deny_rule("busy-rule"), actor="owner", reason="must not persist")
        assert second.latest_owner_policy_version() == 0
        assert second.owner_permission_rules(active_only=False) == []
        assert second.owner_policy_audit() == []
        assert second.owner_permission_rule_history() == []
    finally:
        first.db.rollback()
        first.close()
        second.close()


def test_migration_failure_rolls_back_all_new_schema_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "migration-rollback.sqlite3"
    original = Store._apply_migration

    def fail_at_queue(store: Store, version: int) -> None:
        if version == 4:
            raise RuntimeError("injected_migration_failure")
        original(store, version)

    monkeypatch.setattr(Store, "_apply_migration", fail_at_queue)
    with pytest.raises(RuntimeError, match="injected_migration_failure"):
        Store(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT schema_version FROM task_state_schema").fetchall() == []
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN "
            "('owner_permission_rules', 'owner_policy_state', 'task_queue')"
        ).fetchall() == []
    finally:
        connection.close()

    monkeypatch.undo()
    reopened = Store(path)
    assert [row["schema_version"] for row in reopened.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")] == [1, 2, 3, 4, 5, 6, 7]
    reopened.close()


def test_schema5_rejects_wrong_index_definition_on_reopen(tmp_path: Path) -> None:
    path = tmp_path / "index-drift.sqlite3"
    store = Store(path)
    store.db.execute("DROP INDEX idx_knowledge_claims_freshness")
    store.db.execute("CREATE INDEX idx_knowledge_claims_freshness ON knowledge_claims(claim_id)")
    store.db.commit()
    store.close()

    with pytest.raises(UnsupportedSchemaVersion, match="invalid_index:idx_knowledge_claims_freshness"):
        Store(path)
