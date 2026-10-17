from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from private_agent.security import ActionRequest, OwnerPermissionRule, PolicyEngine
from private_agent.storage import Store, UnsupportedSchemaVersion


def make_action(*, tool: str = "web_research", capabilities: list[str] | None = None, goal: str = "") -> ActionRequest:
    caps = capabilities or ["READ_WORKSPACE"]
    return ActionRequest(
        task_id="permission-task",
        action_id="permission-action",
        tool_or_skill=tool,
        requested_capabilities=caps,
        declared_capabilities=list(caps),
        goal=goal,
    )


def rule(rule_id: str, effect: str, scope_type: str, scope_key: str, *, conditions: dict | None = None) -> OwnerPermissionRule:
    return OwnerPermissionRule(
        rule_id=rule_id,
        effect=effect,
        scope_type=scope_type,
        scope_key=scope_key,
        conditions=conditions or {},
        reason=f"test reason for {rule_id}",
    )


def test_new_database_runs_monotonic_migration_and_creates_owner_tables(tmp_path: Path) -> None:
    path = tmp_path / "fresh.sqlite3"
    store = Store(path)
    versions = [row["schema_version"] for row in store.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")]
    assert versions == [1, 2, 3, 4]
    assert store.db.execute("SELECT name FROM sqlite_master WHERE name = 'owner_permission_rules'").fetchone()
    assert store.db.execute("SELECT name FROM sqlite_master WHERE name = 'owner_policy_audit'").fetchone()
    store.close()


def test_existing_schema_v1_is_upgraded_on_reopen(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    first = Store(path)
    first.db.execute("DELETE FROM task_state_schema WHERE schema_version = 2")
    first.db.execute("DROP TABLE owner_policy_audit")
    first.db.execute("DROP TABLE owner_permission_rules")
    first.db.commit()
    first.close()

    upgraded = Store(path)
    assert [row["schema_version"] for row in upgraded.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")] == [1, 2, 3, 4]
    assert upgraded.db.execute("SELECT name FROM sqlite_master WHERE name = 'owner_permission_rules'").fetchone()
    upgraded.close()


def test_unsupported_future_schema_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    store = Store(path)
    store.db.execute("INSERT INTO task_state_schema(schema_version, applied_at) VALUES (999, 'future')")
    store.db.commit()
    store.close()
    with pytest.raises(UnsupportedSchemaVersion):
        Store(path)


def test_rule_creation_is_versioned_and_audited_in_one_transaction(tmp_path: Path) -> None:
    store = Store(tmp_path / "rules.sqlite3")
    created = store.create_owner_permission_rule(rule("deny-tool", "DENY", "TOOL", "web_research"), actor="owner", reason="block tool")
    assert created["policy_version"] == 1
    saved = store.owner_permission_rules(active_only=False)[0]
    assert saved["rule_id"] == "deny-tool"
    audit = store.owner_policy_audit()
    assert len(audit) == 1
    assert audit[0]["event_type"] == "RULE_CREATED"
    assert audit[0]["policy_version"] == 1
    store.close()


def test_disabling_rule_advances_version_and_preserves_audit(tmp_path: Path) -> None:
    store = Store(tmp_path / "disable.sqlite3")
    store.create_owner_permission_rule(rule("deny-tool", "DENY", "TOOL", "web_research"), actor="owner", reason="block tool")
    updated = store.disable_owner_permission_rule("deny-tool", actor="owner", reason="unblock after review")
    assert updated["status"] == "DISABLED"
    assert updated["policy_version"] == 2
    assert [item["event_type"] for item in store.owner_policy_audit()] == ["RULE_CREATED", "RULE_DISABLED"]
    assert store.owner_permission_rules() == []
    store.close()


def test_rule_transaction_rolls_back_duplicate_without_partial_audit(tmp_path: Path) -> None:
    store = Store(tmp_path / "rollback.sqlite3")
    store.create_owner_permission_rule(rule("same", "DENY", "TOOL", "web_research"), actor="owner", reason="first")
    with pytest.raises(sqlite3.IntegrityError):
        store.create_owner_permission_rule(rule("same", "ALLOW", "TOOL", "web_research"), actor="owner", reason="duplicate")
    assert len(store.owner_policy_audit()) == 1
    assert store.latest_owner_policy_version() == 1
    store.close()


def test_validation_rejects_code_like_or_unsupported_conditions() -> None:
    with pytest.raises(ValueError):
        rule("bad", "ALLOW", "TOOL", "x", conditions={"expression": "__import__('os')"})
    with pytest.raises(ValueError):
        OwnerPermissionRule("bad", "ALLOW", "GLOBAL", "*", reason="")


def test_owner_deny_wins_over_more_specific_allow(tmp_path: Path) -> None:
    store = Store(tmp_path / "precedence.sqlite3")
    store.create_owner_permission_rule(rule("global-allow", "ALLOW", "GLOBAL", "*"), actor="owner", reason="baseline")
    store.create_owner_permission_rule(rule("tool-deny", "DENY", "TOOL", "web_research"), actor="owner", reason="explicit deny")
    decision = PolicyEngine(store=store).evaluate(make_action())
    assert decision.decision == "DENY"
    assert any("tool-deny" in reason for reason in decision.reasons)
    store.close()


def test_owner_require_approval_wins_over_allow_without_changing_safety_invariants(tmp_path: Path) -> None:
    store = Store(tmp_path / "require-approval.sqlite3")
    store.create_owner_permission_rule(rule("global-allow", "ALLOW", "GLOBAL", "*"), actor="owner", reason="baseline")
    store.create_owner_permission_rule(rule("network-approval", "REQUIRE_APPROVAL", "CAPABILITY", "NETWORK_ACCESS"), actor="owner", reason="review network")
    decision = PolicyEngine(store=store).evaluate(make_action(capabilities=["NETWORK_ACCESS"]))
    assert decision.decision == "REQUIRE_APPROVAL"
    assert decision.approval_required is True
    store.close()


def test_owner_allow_is_scoped_and_does_not_override_unknown_or_undeclared_capability(tmp_path: Path) -> None:
    store = Store(tmp_path / "allow.sqlite3")
    store.create_owner_permission_rule(rule("tool-allow", "ALLOW", "TOOL", "web_research"), actor="owner", reason="allow only this tool")
    engine = PolicyEngine(store=store)
    assert engine.evaluate(make_action()).decision == "ALLOW"
    assert engine.evaluate(make_action(tool="other_tool")).decision == "ALLOW"
    unknown = make_action(capabilities=["NOT_A_CAPABILITY"])
    unknown.declared_capabilities = ["NOT_A_CAPABILITY"]
    assert engine.evaluate(unknown).decision == "DENY"
    undeclared = make_action(capabilities=["NETWORK_ACCESS"])
    undeclared.declared_capabilities = []
    assert engine.evaluate(undeclared).decision == "DENY"
    store.close()


def test_rules_and_policy_version_survive_reopen(tmp_path: Path) -> None:
    path = tmp_path / "restart.sqlite3"
    store = Store(path)
    store.create_owner_permission_rule(rule("deny-tool", "DENY", "TOOL", "web_research"), actor="owner", reason="persist deny")
    store.close()
    reopened = Store(path)
    engine = PolicyEngine(store=reopened)
    decision = engine.evaluate(make_action())
    assert decision.decision == "DENY"
    assert decision.policy_version == 1
    assert len(reopened.owner_policy_audit()) == 1
    reopened.close()
