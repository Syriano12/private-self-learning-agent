from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest

from private_agent.core.execution import ExecutionContext, GenericExecutor, StepResult
from private_agent.core.knowledge import KnowledgeClaim, KnowledgeEvidence, KnowledgeSource, KnowledgeValidationError, sha256_hex
from private_agent.core.observation import Observation
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.core.verification import VerificationResult, VerifierRegistry, WebResearchVerifier
import private_agent.security as security_module
import private_agent.storage as storage_module
from private_agent.security import ActionRequest, ApprovalGate, PolicyEngine, SecurityController, SecurityPolicy
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class _Tool:
    permission_level = "public_read"
    required_capabilities: list[str] = []
    declared_capabilities: list[str] = []
    risk_level = "low"
    network_required = False

    def __init__(self, name: str = "safe") -> None:
        self.name = name
        self.calls = 0

    def description(self) -> str:
        return self.name

    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    def output_schema(self) -> dict:
        return {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

    def execute(self, inputs: dict, context: ExecutionContext) -> ToolResult:
        self.calls += 1
        return ToolResult.success({"ok": True})


def _high_risk_action() -> ActionRequest:
    return ActionRequest(
        task_id="phase1-task",
        action_id="phase1-action",
        tool_or_skill="network-tool",
        requested_capabilities=["NETWORK_ACCESS"],
        declared_capabilities=["NETWORK_ACCESS"],
        network_required=True,
        input_fingerprint="input-fingerprint",
    )


def test_policy_property_is_a_defensive_snapshot() -> None:
    controller = SecurityController()
    exposed = controller.policy_engine.policy
    exposed.denied_capabilities.append("SYSTEM_COMMAND")

    assert controller.policy_engine.policy.denied_capabilities == []
    decision = controller.authorize(
        ActionRequest(
            task_id="policy-task",
            action_id="safe-action",
            tool_or_skill="read-only",
            requested_capabilities=[],
            declared_capabilities=[],
        )
    )
    assert decision.decision == "ALLOW"


def test_authoritative_policy_cannot_be_mutated_through_internal_state() -> None:
    controller = SecurityController()
    action = _high_risk_action()
    assert controller.policy_engine.evaluate(action).decision == "REQUIRE_APPROVAL"

    internal = controller.policy_engine._policy
    with pytest.raises(AttributeError, match="clear|security_policy_immutable"):
        internal.approval_required_risks.clear()
    with pytest.raises(AttributeError, match="security_policy_immutable"):
        internal.medium_requires_approval = False
    with pytest.raises(AttributeError, match="policy_authority_immutable"):
        controller.policy_engine._policy = SecurityPolicy()

    assert controller.policy_engine.evaluate(action).decision == "REQUIRE_APPROVAL"


@pytest.mark.parametrize("attack", ["policy_dict", "engine_dict", "controller_dict"])
def test_f02_dict_authority_attacks_are_rejected_at_real_executor_boundary(attack: str) -> None:
    controller = SecurityController()
    tool = _Tool("network-tool")
    tool.required_capabilities = ["NETWORK_ACCESS"]
    tool.declared_capabilities = ["NETWORK_ACCESS"]
    tool.risk_level = "high"
    tool.network_required = True
    registry = ToolRegistry()
    registry.register(tool)
    executor = GenericExecutor(registry, controller)
    step = PlanStep("f02-step", "network-tool", "network-tool", {}, "network", [])
    action = controller.action_for_tool("f02-task", step.id, tool, {}, goal="f02-goal")

    assert controller.policy_engine.evaluate(action).decision == "REQUIRE_APPROVAL"
    with pytest.raises(AttributeError):
        if attack == "policy_dict":
            controller.policy_engine._policy.__dict__["approval_required_risks"] = ()
        elif attack == "engine_dict":
            controller.policy_engine.__dict__["_policy"] = SecurityPolicy(approval_required_risks=[])
        else:
            controller.__dict__["policy_engine"] = PolicyEngine(SecurityPolicy(approval_required_risks=[]))

    with pytest.raises(AttributeError):
        if attack == "policy_dict":
            object.__setattr__(controller.policy_engine, "_policy", SecurityPolicy(approval_required_risks=[]))
        elif attack == "engine_dict":
            object.__setattr__(controller.policy_engine, "store", object())
        else:
            object.__setattr__(controller, "policy_engine", PolicyEngine(SecurityPolicy(approval_required_risks=[])))

    assert controller.policy_engine.evaluate(action).decision == "REQUIRE_APPROVAL"
    blocked = executor.execute_step(
        step,
        ExecutionContext(
            task_id="f02-task",
            approved_permissions={"public_read"},
            metadata={"goal": "f02-goal"},
        ),
    )
    assert blocked.status == "blocked"
    assert blocked.error is not None and blocked.error.code == "permission_not_approved"
    assert tool.calls == 0

    pending = controller.authorize(action, boundary=True)
    assert pending.decision == "REQUIRE_APPROVAL"
    controller.approval_gate.approve(pending.approval_id, actor="owner")
    allowed = executor.execute_step(
        step,
        ExecutionContext(
            task_id="f02-task",
            approved_permissions={"public_read"},
            metadata={"goal": "f02-goal"},
            approval_ids={step.id: pending.approval_id},
        ),
    )
    assert allowed.status == "success"
    assert tool.calls == 1


def test_f02_fix_preserves_owner_approval_policy_version_and_audit(tmp_path: Path) -> None:
    store = Store(tmp_path / "f02-policy-version.sqlite3")
    controller = SecurityController(store=store)
    tool = _Tool("network-tool")
    tool.required_capabilities = ["NETWORK_ACCESS"]
    tool.declared_capabilities = ["NETWORK_ACCESS"]
    tool.risk_level = "high"
    tool.network_required = True
    registry = ToolRegistry()
    registry.register(tool)
    executor = GenericExecutor(registry, controller)
    step = PlanStep("f02-version-step", "network-tool", "network-tool", {}, "network", [])
    action = controller.action_for_tool("f02-version-task", step.id, tool, {}, goal="f02-version-goal")

    pending = controller.authorize(action, boundary=True)
    assert pending.decision == "REQUIRE_APPROVAL"
    controller.approval_gate.approve(pending.approval_id, actor="owner")
    controller.update_policy(SecurityPolicy(policy_version=pending.policy_version + 1), actor="owner")

    stale = executor.execute_step(
        step,
        ExecutionContext(
            task_id="f02-version-task",
            approved_permissions={"public_read"},
            metadata={"goal": "f02-version-goal"},
            approval_ids={step.id: pending.approval_id},
        ),
    )
    assert stale.status == "blocked"
    assert tool.calls == 0
    events = controller.audit.list(task_id="f02-version-task")
    assert any(event["event_type"] == "APPROVAL_INVALIDATED" for event in events)
    assert any(event["event_type"] == "POLICY_CHANGED" for event in controller.audit.list())
    assert not any(event["event_type"] == "ACTION_EXECUTED" for event in events)

    fresh = controller.authorize(action, boundary=True)
    assert fresh.decision == "REQUIRE_APPROVAL"
    controller.approval_gate.approve(fresh.approval_id, actor="owner")
    allowed = executor.execute_step(
        step,
        ExecutionContext(
            task_id="f02-version-task",
            approved_permissions={"public_read"},
            metadata={"goal": "f02-version-goal"},
            approval_ids={step.id: fresh.approval_id},
        ),
    )
    assert allowed.status == "success"
    assert tool.calls == 1
    executed = [event for event in controller.audit.list(task_id="f02-version-task") if event["event_type"] == "ACTION_EXECUTED"]
    assert len(executed) == 1
    assert executed[0]["policy_version"] == pending.policy_version + 1
    store.close()


def test_f01_authoritative_request_map_mutation_cannot_execute_without_owner_approval() -> None:
    controller = SecurityController()
    tool = _Tool("network-tool")
    tool.required_capabilities = ["NETWORK_ACCESS"]
    tool.declared_capabilities = ["NETWORK_ACCESS"]
    tool.risk_level = "high"
    tool.network_required = True
    registry = ToolRegistry()
    registry.register(tool)
    executor = GenericExecutor(registry, controller)
    step = PlanStep("f01-map-step", "network-tool", "network-tool", {}, "network", [])
    action = controller.action_for_tool("f01-map-task", step.id, tool, {}, goal="f01-map-goal")

    pending = controller.authorize(action, boundary=True)
    assert pending.decision == "REQUIRE_APPROVAL"
    try:
        controller.approval_gate._requests[pending.approval_id].status = "APPROVED"
    except (AttributeError, KeyError, TypeError):
        pass

    blocked = executor.execute_step(
        step,
        ExecutionContext(
            task_id="f01-map-task",
            approved_permissions={"public_read"},
            metadata={"goal": "f01-map-goal"},
            approval_ids={step.id: pending.approval_id},
        ),
    )
    assert blocked.status == "blocked"
    assert blocked.error is not None and blocked.error.code == "permission_not_approved"
    assert tool.calls == 0


def test_f02_descriptor_replacement_cannot_bypass_policy_at_executor_boundary() -> None:
    controller = SecurityController()
    tool = _Tool("descriptor-network-tool")
    tool.required_capabilities = ["NETWORK_ACCESS"]
    tool.declared_capabilities = ["NETWORK_ACCESS"]
    tool.risk_level = "high"
    tool.network_required = True
    registry = ToolRegistry()
    registry.register(tool)
    executor = GenericExecutor(registry, controller)
    step = PlanStep("f02-descriptor-step", "descriptor-network-tool", "descriptor-network-tool", {}, "network", [])
    action = controller.action_for_tool("f02-descriptor-task", step.id, tool, {}, goal="f02-descriptor-goal")

    assert controller.policy_engine.evaluate(action).decision == "REQUIRE_APPROVAL"
    try:
        PolicyEngine._policy.replace(
            controller.policy_engine,
            SecurityPolicy(policy_version=999, approval_required_risks=[], always_approval_capabilities=[]),
        )
    except (AttributeError, TypeError):
        pass
    descriptor = PolicyEngine.__dict__["_policy"]
    forged_token = object()
    with pytest.raises(AttributeError):
        descriptor._update_tokens.add(forged_token)
        descriptor._replace_existing(
            controller.policy_engine,
            SecurityPolicy(policy_version=999, approval_required_risks=[], always_approval_capabilities=[]),
            _token=forged_token,
        )

    blocked = executor.execute_step(
        step,
        ExecutionContext(
            task_id="f02-descriptor-task",
            approved_permissions={"public_read"},
            metadata={"goal": "f02-descriptor-goal"},
        ),
    )
    assert controller.policy_engine.evaluate(action).decision == "REQUIRE_APPROVAL"
    assert blocked.status == "blocked"
    assert blocked.error is not None and blocked.error.code == "permission_not_approved"
    assert tool.calls == 0


def test_request_return_is_detached_and_cannot_grant_execution() -> None:
    controller = SecurityController()
    tool = _Tool("network-tool")
    tool.required_capabilities = ["NETWORK_ACCESS"]
    tool.declared_capabilities = ["NETWORK_ACCESS"]
    tool.network_required = True
    registry = ToolRegistry()
    registry.register(tool)
    executor = GenericExecutor(registry, controller)
    step = PlanStep("approval-action", "network-tool", "network-tool", {}, "network", [])
    action = controller.action_for_tool("approval-task", step.id, tool, {}, goal="approval goal")
    decision = controller.policy_engine.evaluate(action)
    exposed = controller.approval_gate.request(action, decision, ttl_seconds=300)

    exposed.status = "APPROVED"
    exposed.actor = "forged-caller"
    exposed.decided_at = datetime.now(timezone.utc).isoformat()
    exposed.policy_version = 999999
    exposed.tool_or_skill = "different-tool"
    exposed.goal = "different goal"
    exposed.input_fingerprint = "different-input"
    exposed.requested_capabilities.clear()

    blocked = executor.execute_step(
        step,
        ExecutionContext(
            task_id="approval-task",
            approved_permissions={"public_read"},
            metadata={"goal": "approval goal"},
            approval_ids={step.id: exposed.approval_id},
        ),
    )
    assert blocked.status == "blocked"
    assert blocked.error is not None and blocked.error.code == "permission_not_approved"
    assert tool.calls == 0
    authoritative = controller.approval_gate.get(exposed.approval_id)
    assert authoritative is not None
    assert authoritative.status == "PENDING"
    assert authoritative.actor == "system"

    approved_view = controller.approval_gate.approve(exposed.approval_id, actor="owner")
    approved_view.status = "DENIED"
    assert controller.approval_gate.get(exposed.approval_id).status == "APPROVED"
    allowed = executor.execute_step(
        step,
        ExecutionContext(
            task_id="approval-task",
            approved_permissions={"public_read"},
            metadata={"goal": "approval goal"},
            approval_ids={step.id: exposed.approval_id},
        ),
    )
    assert allowed.status == "success"
    assert tool.calls == 1


def test_public_store_cannot_forge_persisted_approval(tmp_path: Path) -> None:
    store = Store(tmp_path / "approval-persistence-boundary.sqlite3")
    controller = SecurityController(store=store)
    tool = _Tool("network-tool")
    tool.required_capabilities = ["NETWORK_ACCESS"]
    tool.declared_capabilities = ["NETWORK_ACCESS"]
    tool.risk_level = "high"
    tool.network_required = True
    registry = ToolRegistry()
    registry.register(tool)
    executor = GenericExecutor(registry, controller)
    step = PlanStep("persisted-approval-step", "network-tool", "network-tool", {}, "network", [])
    action = controller.action_for_tool("persisted-approval-task", step.id, tool, {}, goal="persisted approval goal")
    pending = controller.authorize(action, boundary=True)
    assert pending.decision == "REQUIRE_APPROVAL"
    request = controller.approval_gate.get(pending.approval_id)
    assert request is not None

    forged = request.to_dict()
    forged.update({"status": "APPROVED", "actor": "forged-caller", "decided_at": datetime.now(timezone.utc).isoformat()})
    with pytest.raises(ValueError, match="approval_transition_requires_gate"):
        store.save_approval_request(forged)

    blocked = executor.execute_step(
        step,
        ExecutionContext(
            task_id="persisted-approval-task",
            approved_permissions={"public_read"},
            metadata={"goal": "persisted approval goal"},
            approval_ids={step.id: pending.approval_id},
        ),
    )
    assert blocked.status == "blocked"
    assert tool.calls == 0

    controller.approval_gate.approve(pending.approval_id, actor="owner")
    allowed = executor.execute_step(
        step,
        ExecutionContext(
            task_id="persisted-approval-task",
            approved_permissions={"public_read"},
            metadata={"goal": "persisted approval goal"},
            approval_ids={step.id: pending.approval_id},
        ),
    )
    assert allowed.status == "success"
    assert tool.calls == 1
    store.close()


def test_f01_imported_writer_token_cannot_forge_approval_at_executor_boundary(tmp_path: Path) -> None:
    store = Store(tmp_path / "approval-imported-token.sqlite3")
    controller = SecurityController(store=store)
    tool = _Tool("network-token-tool")
    tool.required_capabilities = ["NETWORK_ACCESS"]
    tool.declared_capabilities = ["NETWORK_ACCESS"]
    tool.risk_level = "high"
    tool.network_required = True
    registry = ToolRegistry()
    registry.register(tool)
    executor = GenericExecutor(registry, controller)
    step = PlanStep("imported-token-step", "network-token-tool", "network-token-tool", {}, "network", [])
    action = controller.action_for_tool("imported-token-task", step.id, tool, {}, goal="imported-token-goal")
    pending = controller.authorize(action, boundary=True)
    assert pending.decision == "REQUIRE_APPROVAL"
    request = controller.approval_gate.get(pending.approval_id)
    assert request is not None
    forged = request.to_dict()
    forged.update({"status": "APPROVED", "actor": "imported-forger", "decided_at": datetime.now(timezone.utc).isoformat()})

    imported_token = getattr(storage_module, "_APPROVAL_WRITER_TOKEN", object())
    with pytest.raises((AttributeError, TypeError, ValueError)):
        store.save_approval_request(forged, _writer_token=imported_token)

    blocked = executor.execute_step(
        step,
        ExecutionContext(
            task_id="imported-token-task",
            approved_permissions={"public_read"},
            metadata={"goal": "imported-token-goal"},
            approval_ids={step.id: pending.approval_id},
        ),
    )
    assert blocked.status == "blocked"
    assert blocked.error is not None and blocked.error.code == "permission_not_approved"
    assert tool.calls == 0

    controller.approval_gate.approve(pending.approval_id, actor="owner")
    allowed = executor.execute_step(
        step,
        ExecutionContext(
            task_id="imported-token-task",
            approved_permissions={"public_read"},
            metadata={"goal": "imported-token-goal"},
            approval_ids={step.id: pending.approval_id},
        ),
    )
    assert allowed.status == "success"
    assert tool.calls == 1
    store.close()


def test_f02_descriptor_registry_cannot_replace_policy_at_executor_boundary() -> None:
    controller = SecurityController()
    tool = _Tool("descriptor-tool")
    tool.required_capabilities = ["NETWORK_ACCESS"]
    tool.declared_capabilities = ["NETWORK_ACCESS"]
    tool.risk_level = "high"
    tool.network_required = True
    registry = ToolRegistry()
    registry.register(tool)
    executor = GenericExecutor(registry, controller)
    step = PlanStep("descriptor-step", "descriptor-tool", "descriptor-tool", {}, "network", [])
    action = controller.action_for_tool("descriptor-task", step.id, tool, {}, goal="descriptor-goal")
    assert controller.policy_engine.evaluate(action).decision == "REQUIRE_APPROVAL"

    descriptor = type(controller.policy_engine).__dict__["_policy"]
    descriptor_values = getattr(security_module, "_descriptor_values", None)
    if callable(descriptor_values):
        values = descriptor_values(descriptor)
        values[controller.policy_engine] = SecurityPolicy(approval_required_risks=[])
    replace_existing = getattr(descriptor, "_replace_existing", None)
    if callable(replace_existing):
        with pytest.raises((TypeError, AttributeError)):
            replace_existing(controller.policy_engine, SecurityPolicy(approval_required_risks=[]))

    assert controller.policy_engine.evaluate(action).decision == "REQUIRE_APPROVAL"
    blocked = executor.execute_step(
        step,
        ExecutionContext(
            task_id="descriptor-task",
            approved_permissions={"public_read"},
            metadata={"goal": "descriptor-goal"},
        ),
    )
    assert blocked.status == "blocked"
    assert blocked.error is not None and blocked.error.code == "permission_not_approved"
    assert tool.calls == 0


def test_f03_private_verified_claim_override_is_not_available(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    content = b"caller-controlled content"
    source = KnowledgeSource(
        source_id="f03-private-override-source",
        canonical_uri="https://attacker.example/f03-private-override",
        source_type="web",
        source_class="primary",
        retrieved_at=now,
        content_sha256=sha256_hex(content),
        content=content,
    )
    claim = KnowledgeClaim(
        claim_id="f03-private-override-claim",
        domain="security",
        subject="agent",
        predicate="supports",
        object_text="caller asserted verified fact",
        knowledge_type="fact",
        verification_status="verified",
        status="SUPPORTED",
        confidence=1.0,
        observed_at=now,
        last_verified_at=now,
    )
    store = Store(tmp_path / "f03-private-override.sqlite3")
    saved_source = store._insert_knowledge_source(source)
    with pytest.raises((TypeError, KnowledgeValidationError)):
        store._insert_knowledge_claim(claim, allow_verified=True)
    saved = store._insert_knowledge_claim(claim)
    assert saved.verification_status == "unverified"
    assert saved.status == "PROPOSED"
    assert store.search_knowledge_claims("caller asserted verified fact") == []
    assert saved_source.source_id == source.source_id
    store.close()


def test_all_approval_gate_views_are_detached_from_authoritative_state() -> None:
    controller = SecurityController()
    gate = controller.approval_gate

    def create_request(action_id: str):
        action = ActionRequest(
            task_id=f"approval-view-{action_id}",
            action_id=action_id,
            tool_or_skill="network-tool",
            requested_capabilities=["NETWORK_ACCESS"],
            declared_capabilities=["NETWORK_ACCESS"],
            network_required=True,
            input_fingerprint=f"input-{action_id}",
        )
        return gate.request(action, controller.policy_engine.evaluate(action), ttl_seconds=300)

    requested = create_request("request")
    requested.status = "APPROVED"
    assert gate.get(requested.approval_id).status == "PENDING"

    approved = gate.approve(create_request("approve").approval_id)
    approved.status = "DENIED"
    assert gate.get(approved.approval_id).status == "APPROVED"

    denied = create_request("deny")
    denied_view = gate.deny(denied.approval_id)
    denied_view.status = "APPROVED"
    assert gate.get(denied.approval_id).status == "DENIED"

    cancelled = create_request("cancel")
    cancelled_view = gate.cancel(cancelled.approval_id)
    cancelled_view.status = "APPROVED"
    assert gate.get(cancelled.approval_id).status == "CANCELLED"

    legacy = gate.check("send_message")
    assert legacy is not None
    legacy.status = "APPROVED"
    assert gate.get(legacy.approval_id).status == "PENDING"


def test_caller_asserted_knowledge_cannot_become_active_verified(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    content = b"attacker-controlled source bytes"
    source = KnowledgeSource(
        source_id="attacker-source",
        canonical_uri="https://attacker.example/claim",
        source_type="web",
        source_class="primary",
        retrieved_at=now,
        content_sha256=sha256_hex(content),
        content=content,
    )
    claim = KnowledgeClaim(
        claim_id="attacker-claim",
        domain="security",
        subject="agent",
        predicate="supports",
        object_text="attacker asserted fact",
        knowledge_type="fact",
        verification_status="verified",
        status="SUPPORTED",
        confidence=1.0,
        observed_at=now,
        last_verified_at=now,
    )
    evidence = KnowledgeEvidence(
        evidence_id="attacker-evidence",
        claim_id=claim.claim_id,
        source_id=source.source_id,
        excerpt="attacker asserted evidence",
        verification_status="verified",
        observed_at=now,
    )

    database = tmp_path / "caller-asserted-knowledge.sqlite3"
    store = Store(database)
    saved = store.save_knowledge_bundle(source, claim, [evidence])

    assert saved["claim"]["verification_status"] == "unverified"
    assert saved["claim"]["status"] == "PROPOSED"
    assert store.search_knowledge_claims("attacker asserted fact") == []
    assert store.search_knowledge("attacker asserted fact") == []
    refreshed = KnowledgeClaim(**{**claim.to_dict(), "version": 2, "claim_fingerprint": ""})
    refreshed_result = store.refresh_knowledge_claim(
        refreshed,
        [KnowledgeEvidence(
            evidence_id="attacker-evidence-v2",
            claim_id=refreshed.claim_id,
            source_id=source.source_id,
            excerpt="attacker asserted refreshed evidence",
            verification_status="verified",
            observed_at=now,
        )],
    )
    assert refreshed_result.verification_status == "unverified"
    assert refreshed_result.status == "PROPOSED"
    assert store.search_knowledge_claims("attacker asserted fact") == []
    store.close()
    reopened = Store(database)
    source_row = reopened.get_knowledge_source(source.source_id)
    assert source_row is not None
    assert source_row["provenance_attested"] is False
    assert source_row["effective_source_class"] == "unverified"
    assert reopened.search_knowledge_claims("attacker asserted fact") == []
    reopened.close()


def test_f03_trust_set_mutation_cannot_promote_caller_knowledge(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    content = b"attacker-controlled source bytes"
    source = KnowledgeSource(
        source_id="f03-mutated-source",
        canonical_uri="https://attacker.example/f03-mutated",
        source_type="web",
        source_class="primary",
        retrieved_at=now,
        content_sha256=sha256_hex(content),
        content=content,
    )
    claim = KnowledgeClaim(
        claim_id="f03-mutated-claim",
        domain="security",
        subject="agent",
        predicate="supports",
        object_text="caller asserted active fact",
        knowledge_type="fact",
        verification_status="verified",
        status="SUPPORTED",
        confidence=1.0,
        observed_at=now,
        last_verified_at=now,
    )
    evidence = KnowledgeEvidence(
        evidence_id="f03-mutated-evidence",
        claim_id=claim.claim_id,
        source_id=source.source_id,
        excerpt="caller asserted evidence",
        verification_status="verified",
        observed_at=now,
    )

    database = tmp_path / "f03-mutated-trust.sqlite3"
    store = Store(database)
    saved = store.save_knowledge_bundle(source, claim, [evidence])
    assert saved["claim"]["verification_status"] == "unverified"
    try:
        store._trusted_knowledge_source_ids.add(source.source_id)
    except (AttributeError, TypeError):
        pass

    refreshed = KnowledgeClaim(**{**claim.to_dict(), "version": 2, "claim_fingerprint": ""})
    refreshed_result = store.refresh_knowledge_claim(
        refreshed,
        [KnowledgeEvidence(
            evidence_id="f03-mutated-evidence-v2",
            claim_id=refreshed.claim_id,
            source_id=source.source_id,
            locator="https://attacker.example/f03-mutated#evidence",
            excerpt="caller asserted refreshed evidence",
            verification_status="verified",
            observed_at=now,
        )],
    )
    assert refreshed_result.verification_status == "unverified"
    assert refreshed_result.status == "PROPOSED"
    assert store.search_knowledge_claims("caller asserted active") == []
    assert store.search_knowledge("caller asserted active") == []
    store.close()

    reopened = Store(database)
    assert reopened.search_knowledge_claims("caller asserted active") == []
    assert reopened.search_knowledge("caller asserted active") == []
    reopened.close()


def test_f03_direct_legacy_projection_cannot_forge_ingestion_attestation(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    content = b"caller-controlled bytes with a valid hash"
    source = KnowledgeSource(
        source_id="f03-direct-source",
        canonical_uri="https://attacker.example/f03-direct",
        source_type="web",
        source_class="primary",
        retrieved_at=now,
        content_sha256=sha256_hex(content),
        content=content,
    )
    claim = KnowledgeClaim(
        claim_id="f03-direct-claim",
        domain="security",
        subject="agent",
        predicate="supports",
        object_text="directly projected caller fact",
        knowledge_type="fact",
        verification_status="verified",
        status="SUPPORTED",
        confidence=1.0,
        observed_at=now,
        last_verified_at=now,
    )
    evidence = KnowledgeEvidence(
        evidence_id="f03-direct-evidence",
        claim_id=claim.claim_id,
        source_id=source.source_id,
        excerpt="caller supplied evidence",
        verification_status="verified",
        observed_at=now,
    )
    store = Store(tmp_path / "f03-direct-projection.sqlite3")

    with pytest.raises(KnowledgeValidationError, match="knowledge_ingestion_attestation_required"):
        store._project_knowledge_claim(claim, source, evidence)

    assert store.search_knowledge("directly projected caller fact") == []
    assert store.db.execute(
        "SELECT 1 FROM knowledge_items WHERE id = ?", (claim.claim_id,)
    ).fetchone() is None
    store.close()


def test_sealed_controller_rejects_authority_rebinding_and_unsealing() -> None:
    for attribute, value in (
        ("_security_state_sealed", False),
        ("policy_engine", PolicyEngine()),
        ("approval_gate", ApprovalGate()),
    ):
        controller = SecurityController()
        with pytest.raises(AttributeError, match="security_state_immutable"):
            setattr(controller, attribute, value)


def test_public_approval_snapshot_cannot_be_mutated_into_approval() -> None:
    controller = SecurityController()
    action = _high_risk_action()
    decision = controller.preflight(action)
    request = controller.approval_gate.get(decision.approval_id)
    assert request is not None and request.status == "PENDING"

    request.status = "APPROVED"
    assert controller.approval_gate.get(decision.approval_id).status == "PENDING"
    boundary = controller.authorize(action, approval_id=decision.approval_id, boundary=True)
    assert boundary.decision == "REQUIRE_APPROVAL"
    assert boundary.allowed is False


def test_owner_permission_rule_rejects_empty_actor(tmp_path: Path) -> None:
    store = Store(tmp_path / "owner-actor.sqlite3")
    with pytest.raises(ValueError, match="owner_actor_required"):
        store.create_owner_permission_rule(
            {
                "rule_id": "phase1-empty-actor",
                "effect": "DENY",
                "scope_type": "GLOBAL",
                "scope_key": "*",
                "conditions": {},
                "reason": "must have an owner",
            },
            actor="",
            reason="must have an owner",
        )
    store.close()


def test_skill_action_uses_declared_contract_capabilities() -> None:
    controller = SecurityController()
    candidate = SimpleNamespace(
        name="network-skill",
        version=1,
        status="QUARANTINED",
        declared_capabilities=["network"],
        metadata={},
        required_resources={},
    )

    action = controller.action_for_skill("task", "invoke", candidate, {}, goal="goal")
    assert action.requested_capabilities == ["NETWORK_ACCESS"]
    assert action.declared_capabilities == ["NETWORK_ACCESS"]
    assert action.network_required is True


def test_web_research_verifier_rejects_invalid_source_thresholds() -> None:
    verifier = WebResearchVerifier()
    observation = Observation(
        "task",
        "step",
        "research",
        "success",
        {"sources": [{"url": "https://valid.example", "text": "valid evidence"}], "errors": []},
    )

    with pytest.raises(ValueError, match="minimum_sources_invalid"):
        verifier.verify(observation, {"minimum_sources": True})
    with pytest.raises(ValueError, match="minimum_sources_invalid"):
        verifier.verify(observation, {"minimum_sources": 0})


def test_web_research_required_terms_only_use_valid_sources() -> None:
    observation = Observation(
        "task",
        "step",
        "research",
        "success",
        {
            "sources": [
                {"url": "https://valid.example", "text": "ordinary evidence"},
                {"text": "required-term hidden in invalid source"},
            ],
            "errors": [],
        },
    )
    result = WebResearchVerifier().verify(observation, {"minimum_sources": 1, "required_terms": ["required-term"]})
    assert result.status == "FAILED"
    assert result.details["missing_terms"] == ["required-term"]


def test_web_research_verifier_rejects_credentialed_or_non_http_sources() -> None:
    observation = Observation(
        "task",
        "step",
        "research",
        "success",
        {
            "sources": [
                {"url": "https://user:password@example.com/private", "text": "secret source"},
                {"url": "file:///tmp/target", "text": "host source"},
            ],
            "errors": [],
        },
    )
    result = WebResearchVerifier().verify(observation, {"minimum_sources": 1})
    assert result.status == "INSUFFICIENT"
    assert {item["type"]: item["value"] for item in result.evidence}["valid_source_count"] == 0


def test_custom_verifier_cannot_turn_failed_execution_into_verified() -> None:
    class UnsafeVerifier:
        name = "unsafe"

        def verify(self, observation: Observation, criteria: dict) -> VerificationResult:
            return VerificationResult("VERIFIED", "untrusted verifier claim", verifier_type=self.name)

    failed = Observation("task", "step", "tool", "failed", observed_error={"code": "tool_failed"})
    result = VerifierRegistry([UnsafeVerifier()]).verify("unsafe", failed, {})
    assert result.status == "FAILED"
    assert result.details["code"] == "verification_requires_successful_execution"


def test_observation_rejects_non_json_output_and_enforces_bound() -> None:
    observation = Observation("task", "step", "tool", "success", observed_output=object())
    with pytest.raises(ValueError, match="observation_not_json_safe"):
        observation.to_dict()

    oversized = Observation("task", "step", "tool", "success", observed_output={"value": "x" * 150_000})
    with pytest.raises(ValueError, match="observation_too_large"):
        oversized.to_dict()


def test_step_and_verification_results_reject_non_json_and_oversized_payloads() -> None:
    with pytest.raises(ValueError, match="step_result_not_json_safe"):
        StepResult("step", "tool", "success", output=object()).to_dict()
    with pytest.raises(ValueError, match="step_result_too_large"):
        StepResult("step", "tool", "success", output={"value": "x" * 150_000}).to_dict()

    verification = VerificationResult(
        "VERIFIED",
        "claimed",
        evidence=[object()],
        verifier_type="test",
    )
    with pytest.raises(ValueError, match="verification_result_not_json_safe"):
        verification.to_dict()


def test_execute_plan_orders_dependencies_before_side_effects() -> None:
    first = _Tool("first")
    second = _Tool("second")
    registry = ToolRegistry()
    registry.register(first)
    registry.register(second)
    executor = GenericExecutor(registry, SecurityController())
    plan = TaskPlan(
        "ordered goal",
        [
            PlanStep("second", "second", "second", {}, "second", depends_on=["first"]),
            PlanStep("first", "first", "first", {}, "first"),
        ],
        "dependency order",
    )

    results = executor.execute_plan(plan, ExecutionContext(task_id="ordered-task", metadata={"goal": "ordered goal"}))
    assert [result.step_id for result in results] == ["first", "second"]
    assert [result.status for result in results] == ["success", "success"]
    assert first.calls == 1
    assert second.calls == 1


def test_execute_plan_rejects_cyclic_dependencies_before_any_side_effect() -> None:
    first = _Tool("first")
    second = _Tool("second")
    registry = ToolRegistry()
    registry.register(first)
    registry.register(second)
    executor = GenericExecutor(registry, SecurityController())
    plan = TaskPlan(
        "cyclic goal",
        [
            PlanStep("first", "first", "first", {}, "first", depends_on=["second"]),
            PlanStep("second", "second", "second", {}, "second", depends_on=["first"]),
        ],
        "cycle",
    )

    results = executor.execute_plan(plan, ExecutionContext(task_id="cyclic-task", metadata={"goal": "cyclic goal"}))
    assert results
    assert all(result.status == "blocked" for result in results)
    assert all(result.error and result.error.code == "invalid_plan_dependencies" for result in results)
    assert first.calls == 0
    assert second.calls == 0


def test_execute_plan_rejects_malformed_dependency_types_before_side_effect() -> None:
    tool = _Tool("malformed")
    registry = ToolRegistry()
    registry.register(tool)
    executor = GenericExecutor(registry, SecurityController())
    plan = TaskPlan(
        "malformed goal",
        [PlanStep("malformed", "malformed", "malformed", {}, "bad", depends_on="not-a-list")],
        "malformed dependencies",
    )

    results = executor.execute_plan(plan, ExecutionContext(task_id="malformed-task", metadata={"goal": "malformed goal"}))
    assert results and results[0].status == "blocked"
    assert results[0].error and results[0].error.code == "invalid_plan_dependencies"
    assert tool.calls == 0


def test_owner_policy_version_allocation_is_serialized_across_threads(tmp_path: Path) -> None:
    store = Store(tmp_path / "owner-policy-concurrency.sqlite3")

    def create(index: int) -> int:
        result = store.create_owner_permission_rule(
            {
                "rule_id": f"concurrent-{index}",
                "effect": "ALLOW",
                "scope_type": "TOOL",
                "scope_key": f"tool-{index}",
                "conditions": {},
                "reason": "concurrency regression",
            },
            actor="owner",
            reason="concurrency regression",
        )
        return int(result["policy_version"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        versions = list(pool.map(create, range(8)))

    assert sorted(versions) == list(range(1, 9))
    assert store.latest_owner_policy_version() == 8


def test_malformed_queued_state_is_blocked_and_not_claimed_for_execution(tmp_path: Path) -> None:
    store = Store(tmp_path / "malformed-queued-state.sqlite3")
    created = store.create_queued_task("malformed state")
    task_id = created["task_id"]
    store.db.execute("UPDATE task_states SET state_json = ? WHERE task_id = ?", ("{broken", task_id))
    store.db.commit()

    claim = store.claim_next_task("worker-malformed")
    assert claim is not None
    assert claim["kind"] == "BLOCKED"
    assert claim["error_code"] == "malformed_task_state"
    queue = store.task_queue_record(task_id)
    assert queue is not None and queue["queue_status"] == "BLOCKED"
    assert store.claim_next_task("worker-malformed-again") is None
    store.close()


def test_expired_lease_with_malformed_state_is_blocked_during_recovery(tmp_path: Path) -> None:
    store = Store(tmp_path / "expired-malformed-state.sqlite3")
    created = store.create_queued_task("expired malformed state")
    task_id = created["task_id"]
    claimed = store.claim_next_task("worker-expired")
    assert claimed is not None and claimed["kind"] == "CLAIMED"
    store.db.execute("UPDATE task_states SET state_json = ? WHERE task_id = ?", ("{broken", task_id))
    store.db.execute(
        "UPDATE task_queue SET lease_expires_at = ? WHERE task_id = ?",
        ("2000-01-01T00:00:00+00:00", task_id),
    )
    store.db.commit()

    recovered = store.claim_next_task("worker-recovery")
    assert recovered is not None
    assert recovered["kind"] == "BLOCKED"
    assert recovered["error_code"] == "malformed_task_state"
    assert store.task_queue_record(task_id)["queue_status"] == "BLOCKED"
    store.close()
