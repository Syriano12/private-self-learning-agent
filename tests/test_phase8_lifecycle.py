from __future__ import annotations

import json
import time
from pathlib import Path

from private_agent.core.learning import LearningEngine, SQLiteLearningMemory
from private_agent.core.memory import SQLiteExperienceMemory
from private_agent.core.reflection import ReflectionEngine, SQLiteReflectionMemory
from private_agent.skills.lifecycle import (
    ApprovalDecision,
    CrossVerificationReport,
    MutationTestingEngine,
    RuntimeVerificationReport,
    SandboxCapabilities,
    SandboxResult,
    SkillCandidate,
    SkillContract,
    SkillContractValidator,
    SkillCrossVerifier,
    SkillLifecycleManager,
    SkillRuntimeMonitor,
    SkillRuntimeVerifier,
    SkillSandbox,
    SkillStaticAnalyzer,
)
from private_agent.skills.registry import SkillRegistry
from private_agent.security import SecurityController
from private_agent.storage import Store


SAFE_SOURCE = """\
def run(value):
    if value > 0:
        return {'result': 'positive'}
    return {'result': 'non_positive'}
"""

EXPECTED_INPUTS = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False,
}
EXPECTED_OUTPUTS = {
    "type": "object",
    "properties": {"result": {"type": "string"}},
    "required": ["result"],
    "additionalProperties": False,
}


def safe_implementation(inputs: dict) -> dict:
    return {"result": "positive" if inputs["value"] > 0 else "non_positive"}


def failing_implementation(inputs: dict) -> dict:
    raise RuntimeError("controlled failure")


class ToggleImplementation:
    def __init__(self, failing: bool = False) -> None:
        self.failing = failing

    def __call__(self, inputs: dict) -> dict:
        if self.failing:
            raise RuntimeError("degraded skill")
        return {"result": "positive" if inputs["value"] > 0 else "non_positive"}


def kill_comparison_mutants(mutated_source: str, mutant) -> bool:
    return mutant.category == "comparison_inversion" and "<= 0" in mutated_source


def survive_all_mutants(mutated_source: str, mutant) -> bool:
    return False


def candidate(
    skill_id: str,
    *,
    name: str = "positive_skill",
    version: int = 1,
    source: str = SAFE_SOURCE,
    implementation=safe_implementation,
    runner=kill_comparison_mutants,
    capabilities: list[str] | None = None,
    metadata: dict | None = None,
    parent_version: int | None = None,
) -> SkillCandidate:
    return SkillCandidate(
        skill_id=skill_id,
        name=name,
        version=version,
        description="deterministic positive classification",
        source=source,
        entry_point="run",
        declared_capabilities=capabilities or [],
        expected_inputs=EXPECTED_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        tests=["positive input returns positive", "zero input returns non_positive"],
        metadata={"invariants": ["non_empty_result"], **(metadata or {})},
        provenance={"source": "phase8-test"},
        parent_version=parent_version,
        implementation=implementation,
        mutation_test_runner=runner,
    )


def manager(store: Store, *, monitor: SkillRuntimeMonitor | None = None, **kwargs) -> SkillLifecycleManager:
    security_controller = kwargs.pop("security_controller", SecurityController(store=store))
    return SkillLifecycleManager(
        SkillRegistry(store),
        mutation_engine=MutationTestingEngine(max_mutants=2, threshold=1.0),
        monitor=monitor,
        security_controller=security_controller,
        **kwargs,
    )


def test_candidate_contract_and_ast_analysis_are_deterministic() -> None:
    safe = candidate("safe-analysis")
    contract = SkillContract.from_candidate(safe)
    assert SkillContractValidator().validate(safe, contract).passed is True
    report = SkillStaticAnalyzer().analyze(safe, contract)
    assert report.passed is True
    assert report.imports == []
    assert report.findings == []

    unsafe = candidate(
        "unsafe-analysis",
        source="import os\ndef run(value):\n    return eval(value)\n",
        implementation=None,
        runner=None,
    )
    unsafe_report = SkillStaticAnalyzer().analyze(unsafe, SkillContract.from_candidate(unsafe))
    rule_ids = {finding.rule_id for finding in unsafe_report.findings}
    assert unsafe_report.passed is False
    assert {"dangerous_import", "dynamic_code_execution"}.issubset(rule_ids)


def test_ast_detects_undeclared_network_and_filesystem_capabilities() -> None:
    network = candidate("network-analysis", source="import socket\ndef run(value):\n    return {'result': 'x'}\n")
    report = SkillStaticAnalyzer().analyze(network, SkillContract.from_candidate(network))
    assert any(item.rule_id == "undeclared_network" for item in report.findings)

    filesystem = candidate("filesystem-analysis", source="def run(value):\n    return open('/tmp/x').read()\n")
    report = SkillStaticAnalyzer().analyze(filesystem, SkillContract.from_candidate(filesystem))
    assert any(item.rule_id == "undeclared_filesystem_access" for item in report.findings)


def test_mutation_engine_generates_bounded_mutants_without_running_untrusted_runner() -> None:
    safe = candidate("mutation")
    original = safe.source
    killed = MutationTestingEngine(max_mutants=2, threshold=1.0).run(safe, kill_comparison_mutants)
    assert killed.total_mutants >= 1
    assert killed.killed_mutants == 0
    assert killed.passed is False
    assert killed.reason == "external_sandbox_required"
    assert safe.source == original

    survived = MutationTestingEngine(max_mutants=2, threshold=1.0).run(safe, survive_all_mutants)
    assert survived.survived_mutants == survived.total_mutants
    assert survived.passed is False
    assert survived.reason == "external_sandbox_required"


def test_mutation_requires_a_configured_test_runner() -> None:
    safe = candidate("no-runner", runner=None)
    report = MutationTestingEngine(max_mutants=2).run(safe)
    assert report.passed is False
    assert report.reason == "mutation_test_runner_required"


def test_sandbox_capabilities_are_unavailable_without_external_runtime() -> None:
    capabilities = SandboxCapabilities.detect()
    assert isinstance(capabilities.to_dict(), dict)
    assert "filesystem_restrictions" in capabilities.unavailable
    network = candidate("network-skill", capabilities=["network"])
    result = SkillSandbox(capabilities).execute(network, SkillContract.from_candidate(network), {"value": 1})
    assert result.status == "QUARANTINED"
    assert result.error == "external_sandbox_unavailable"


def test_sandbox_execution_is_disabled_and_runtime_output_verification_remains_available() -> None:
    def slow(inputs: dict) -> dict:
        time.sleep(0.2)
        return {"result": "late"}

    slow_candidate = candidate("slow", implementation=slow)
    contract = SkillContract(max_execution_time=0.05, max_output_size=1000)
    result = SkillSandbox().execute(slow_candidate, contract, {"value": 1})
    assert result.status == "QUARANTINED"
    assert result.error == "external_sandbox_unavailable"

    verifier = SkillRuntimeVerifier()
    bad = verifier.verify(
        safe_candidate := candidate("bad-output"),
        SkillContract.from_candidate(safe_candidate),
        {"value": 1},
        SandboxResult("PASSED", output={"wrong": True}),
    )
    assert bad.passed is False
    assert any("missing_required_input" not in error for error in bad.errors)
    assert any("invalid_type" not in error for error in bad.errors)


def test_cross_verification_and_contract_policy_reject_unsafe_capabilities() -> None:
    cross = SkillCrossVerifier().verify(candidate("cross"), {"result": "ok"})
    assert cross.passed is True
    failed = SkillCrossVerifier().verify(candidate("cross-fail", metadata={"invariants": ["non_empty_result"]}), {"result": ""})
    assert failed.passed is False

    network = candidate("network-contract", capabilities=["network"])
    contract = SkillContract(network_allowed=False, declared_capabilities=["network"])
    report = SkillContractValidator().validate(network, contract)
    assert report.passed is False
    assert "network_capability_not_allowed_by_contract" in report.errors


def test_candidate_metadata_is_quarantined_persisted_and_retrievable_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "agent.sqlite3"
    store = Store(path)
    lifecycle = manager(store)
    result = lifecycle.submit(candidate("candidate-a"), inputs={"value": 1})
    assert result.approval.status == "QUARANTINED"
    assert result.candidate.status == "QUARANTINED"
    assert result.analysis.passed is True
    assert result.mutation.passed is False
    assert result.mutation.reason == "external_sandbox_required"
    assert result.sandbox.status == "QUARANTINED"
    assert result.sandbox.error == "external_sandbox_required"
    active = lifecycle.registry.active("positive_skill")
    assert active is None
    persisted = lifecycle.registry.get_version("positive_skill", 1)
    assert persisted is not None
    assert persisted["status"] == "QUARANTINED"
    assert persisted["analysis"]["passed"] is True
    store.close()

    restarted = Store(path)
    loaded = SkillRegistry(restarted).active("positive_skill")
    assert loaded is None
    persisted_after_restart = SkillRegistry(restarted).get_version("positive_skill", 1)
    assert persisted_after_restart is not None
    assert persisted_after_restart["candidate"]["skill_id"] == "candidate-a"
    assert persisted_after_restart["status"] == "QUARANTINED"
    restarted.close()


def test_candidate_b_is_rejected_and_never_activated(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    lifecycle = manager(store)
    unsafe = candidate(
        "candidate-b",
        source="import subprocess\ndef run(value):\n    return {'result': 'unsafe'}\n",
        implementation=None,
        runner=None,
    )
    result = lifecycle.submit(unsafe, inputs={"value": 1})
    assert result.approval.status == "REJECTED"
    assert lifecycle.registry.active("positive_skill") is None
    assert result.candidate.status == "REJECTED"
    assert any(finding.blocking for finding in result.analysis.findings)
    store.close()


def test_weak_candidate_is_quarantined_when_mutants_survive(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    lifecycle = manager(store)
    weak = candidate("weak", runner=survive_all_mutants)
    result = lifecycle.submit(weak, inputs={"value": 1})
    assert result.approval.status == "QUARANTINED"
    assert result.candidate.status == "QUARANTINED"
    assert lifecycle.registry.active("positive_skill") is None
    store.close()


def test_versioning_never_activates_untrusted_versions(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    lifecycle = manager(store)
    first = candidate("version-1", version=1)
    first_result = lifecycle.submit(first, inputs={"value": 1})
    assert first_result.approval.approved is False
    conflict = candidate("version-conflict", version=1)
    conflict_result = lifecycle.submit(conflict, inputs={"value": 1})
    assert conflict_result.approval.status == "QUARANTINED"
    assert lifecycle.registry.active("positive_skill") is None

    second = candidate("version-2", version=2, parent_version=1)
    assert lifecycle.submit(second, inputs={"value": 1}).approval.approved is False
    assert lifecycle.registry.active("positive_skill") is None
    assert lifecycle.registry.get_version("positive_skill", 1)["status"] == "QUARANTINED"
    assert lifecycle.registry.get_version("positive_skill", 2)["status"] == "QUARANTINED"
    store.close()


def test_candidate_c_is_quarantined_before_runtime_or_learning(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    experience_memory = SQLiteExperienceMemory(store)
    reflection_memory = SQLiteReflectionMemory(store)

    class ReflectionDouble:
        def __init__(self) -> None:
            self.calls = []

        def reflect_for_experience(self, experience):
            self.calls.append(experience.task_id)
            return []

    class LearningDouble:
        def __init__(self) -> None:
            self.calls = []

        def learn(self, insights):
            self.calls.append(list(insights))
            return []

    reflection = ReflectionDouble()
    learning = LearningDouble()
    v1_impl = ToggleImplementation(False)
    v2_impl = ToggleImplementation(False)
    lifecycle = manager(
        store,
        monitor=SkillRuntimeMonitor(min_executions=3, max_failure_rate=0.5),
        experience_memory=experience_memory,
        reflection_engine=reflection,
        learning_engine=learning,
    )
    first = candidate("runtime-v1", name="runtime_skill", version=1, implementation=v1_impl)
    second = candidate("runtime-v2", name="runtime_skill", version=2, implementation=v2_impl, parent_version=1)
    first_result = lifecycle.submit(first, inputs={"value": 1})
    second_result = lifecycle.submit(second, inputs={"value": 1})
    assert first_result.approval.approved is False
    assert second_result.approval.approved is False
    assert first_result.sandbox.error == "external_sandbox_required"
    assert second_result.sandbox.error == "external_sandbox_required"
    assert lifecycle.registry.active("runtime_skill") is None
    assert reflection.calls == []
    assert learning.calls == []
    store.close()


def test_active_execution_stays_quarantined_without_external_runtime(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    lifecycle = manager(store)
    assert lifecycle.submit(candidate("runtime-ok"), inputs={"value": 1}).approval.approved is False
    result = lifecycle.execute_active("positive_skill", {"value": 1})
    assert result.status == "QUARANTINED"
    assert result.sandbox.error == "active_skill_not_found"
    store.close()


def test_persisted_reports_are_sanitized() -> None:
    store = Store(":memory:")
    lifecycle = manager(store)
    unsafe = candidate("secret-skill")
    unsafe.metadata["authorization"] = "Bearer very-secret-token"
    result = lifecycle.submit(unsafe, inputs={"value": 1})
    serialized = json.dumps(result.to_dict())
    assert "very-secret-token" not in serialized
    assert "[REDACTED]" in serialized or "authorization" not in serialized
    store.close()
