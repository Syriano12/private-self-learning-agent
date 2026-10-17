from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from private_agent import storage as storage_module
from private_agent.security import ActionRequest, OwnerPermissionRule, PolicyEngine, SecurityController
from private_agent.storage import Store, UnsupportedSchemaVersion


def action(*, tool: str = "network", capabilities: list[str] | None = None) -> ActionRequest:
    caps = capabilities or ["NETWORK_ACCESS"]
    return ActionRequest(
        task_id="phase14a1-task",
        action_id="phase14a1-action",
        tool_or_skill=tool,
        requested_capabilities=caps,
        declared_capabilities=list(caps),
    )


def rule(rule_id: str, effect: str, scope_type: str = "TOOL", scope_key: str = "network", **kwargs) -> OwnerPermissionRule:
    return OwnerPermissionRule(rule_id, effect, scope_type, scope_key, reason=f"reason:{rule_id}", **kwargs)


def test_old_approval_is_not_revalidated_after_owner_policy_change_and_disable(tmp_path: Path) -> None:
    store = Store(tmp_path / "approval-invalidation.sqlite3")
    controller = SecurityController(store=store)
    target = action()
    first = controller.preflight(target)
    controller.approval_gate.approve(first.approval_id)
    assert controller.authorize(target, approval_id=first.approval_id, boundary=True).allowed

    store.create_owner_permission_rule(rule("temporary-deny", "DENY"), actor="owner", reason="temporary deny")
    store.disable_owner_permission_rule("temporary-deny", actor="owner", reason="remove deny")

    current = store.latest_owner_policy_version()
    assert current > first.policy_version
    stale = controller.authorize(target, approval_id=first.approval_id, boundary=True)
    assert stale.allowed is False
    assert stale.decision == "REQUIRE_APPROVAL"
    assert stale.policy_version == current
    store.close()


def test_owner_allow_overrides_default_high_risk_approval_only_after_invariants(tmp_path: Path) -> None:
    store = Store(tmp_path / "allow-high-risk.sqlite3")
    store.create_owner_permission_rule(
        rule("allow-network", "ALLOW", "CAPABILITY", "NETWORK_ACCESS"), actor="owner", reason="allow reviewed network"
    )
    engine = PolicyEngine(store=store)
    allowed = engine.evaluate(action())
    assert allowed.decision == "ALLOW"
    assert allowed.policy_version == store.latest_owner_policy_version()

    unknown = action(capabilities=["NOT_A_CAPABILITY"])
    unknown.declared_capabilities = ["NOT_A_CAPABILITY"]
    assert engine.evaluate(unknown).decision == "DENY"
    undeclared = action()
    undeclared.declared_capabilities = []
    assert engine.evaluate(undeclared).decision == "DENY"

    quarantine = action()
    quarantine.skill_status = "QUARANTINED"
    assert engine.evaluate(quarantine).decision == "QUARANTINE"
    forbidden = action()
    forbidden.action_type = "bypass_approval"
    assert engine.evaluate(forbidden).decision == "DENY"
    undeclared_resource = action(capabilities=["READ_WORKSPACE"])
    undeclared_resource.network_required = True
    assert engine.evaluate(undeclared_resource).decision == "DENY"
    store.close()


def test_owner_require_approval_precedes_more_specific_allow(tmp_path: Path) -> None:
    store = Store(tmp_path / "owner-precedence.sqlite3")
    store.create_owner_permission_rule(rule("global-review", "REQUIRE_APPROVAL", "GLOBAL", "*"), actor="owner", reason="review all")
    store.create_owner_permission_rule(rule("tool-allow", "ALLOW", "TOOL", "network"), actor="owner", reason="allow tool")
    decision = PolicyEngine(store=store).evaluate(action())
    assert decision.decision == "REQUIRE_APPROVAL"
    assert any("global-review" in reason for reason in decision.reasons)
    store.close()


def test_marker_version_two_with_missing_required_objects_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "inconsistent-v2.sqlite3"
    store = Store(path)
    store.db.execute("DROP TABLE owner_permission_rules")
    store.db.commit()
    store.close()
    with pytest.raises(UnsupportedSchemaVersion, match="inconsistent_schema_version:2"):
        Store(path)


def test_owner_rule_history_is_versioned_and_reconstructable(tmp_path: Path) -> None:
    store = Store(tmp_path / "history.sqlite3")
    store.create_owner_permission_rule(rule("history-rule", "DENY"), actor="owner", reason="create")
    store.disable_owner_permission_rule("history-rule", actor="owner", reason="disable")
    history = store.owner_permission_rule_history("history-rule")
    assert [item["event_type"] for item in history] == ["RULE_CREATED", "RULE_DISABLED"]
    assert [item["policy_version"] for item in history] == [1, 2]
    assert all(item["actor"] == "owner" and item["reason"] for item in history)
    store.close()


def test_concurrent_rule_creations_allocate_unique_monotonic_versions(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.sqlite3"
    first = Store(path)
    second = Store(path)

    def create(store: Store, name: str) -> int:
        return int(store.create_owner_permission_rule(rule(name, "DENY", scope_key=name), actor="owner", reason=name)["policy_version"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        versions = sorted(pool.map(lambda item: create(*item), [(first, "tool-a"), (second, "tool-b")]))
    assert versions == [1, 2]
    assert [item["policy_version"] for item in first.owner_permission_rule_history()] == [1, 2]
    first.close()
    second.close()


def test_expiry_is_normalized_to_utc_and_compared_by_instant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = "2026-01-01T06:00:00+00:00"
    monkeypatch.setattr(storage_module, "now", lambda: fixed)
    store = Store(tmp_path / "expiry.sqlite3")
    future = OwnerPermissionRule(
        "future", "DENY", "TOOL", "network", expires_at="2026-01-01T10:00:00+03:00", reason="future"
    )
    past = OwnerPermissionRule(
        "past", "DENY", "TOOL", "network", expires_at="2026-01-01T08:00:00+03:00", reason="past"
    )
    store.create_owner_permission_rule(future, actor="owner", reason="future")
    store.create_owner_permission_rule(past, actor="owner", reason="past")
    active = {item["rule_id"] for item in store.owner_permission_rules()}
    assert active == {"future"}
    all_rules = {item["rule_id"]: item for item in store.owner_permission_rules(active_only=False)}
    assert all_rules["future"]["expires_at"] == "2026-01-01T07:00:00+00:00"
    assert all_rules["past"]["expires_at"] == "2026-01-01T05:00:00+00:00"
    store.close()


def test_disable_requires_non_empty_bounded_actor_and_reason(tmp_path: Path) -> None:
    store = Store(tmp_path / "validation.sqlite3")
    store.create_owner_permission_rule(rule("validation", "DENY"), actor="owner", reason="create")
    with pytest.raises(ValueError, match="owner_actor_required"):
        store.disable_owner_permission_rule("validation", actor="", reason="disable")
    with pytest.raises(ValueError, match="owner_rule_reason_required"):
        store.disable_owner_permission_rule("validation", actor="owner", reason="")
    store.close()
