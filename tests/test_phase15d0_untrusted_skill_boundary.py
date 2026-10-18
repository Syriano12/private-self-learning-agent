from __future__ import annotations

import json
from pathlib import Path

import pytest

from private_agent.security import SecurityController
from private_agent.skills.lifecycle import (
    MutationReport,
    MutationTestingEngine,
    SandboxCapabilities,
    SkillCandidate,
    SkillContract,
    SkillContractValidator,
    SkillLifecycleManager,
    SkillSandbox,
    SkillStaticAnalyzer,
    decode_sandbox_result,
)
from private_agent.skills.registry import SkillRegistry
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


def make_candidate(
    implementation,
    mutation_test_runner=None,
) -> SkillCandidate:
    return SkillCandidate(
        skill_id="phase15d0-skill",
        name="phase15d0_skill",
        version=1,
        description="untrusted execution boundary test",
        source=SAFE_SOURCE,
        entry_point="run",
        expected_inputs=EXPECTED_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        tests=["returns a result"],
        provenance={"source": "phase15d0-test"},
        implementation=implementation,
        mutation_test_runner=mutation_test_runner,
    )


def test_candidate_metadata_serialization_does_not_deepcopy_untrusted_callables(tmp_path: Path) -> None:
    marker = tmp_path / "deepcopy-called"

    class UntrustedCallable:
        def __call__(self, _inputs: dict) -> dict:
            marker.write_text("called", encoding="utf-8")
            return {"result": "unexpected"}

        def __deepcopy__(self, _memo):
            marker.write_text("deepcopied", encoding="utf-8")
            raise AssertionError("untrusted callable was deep-copied")

    candidate = make_candidate(UntrustedCallable(), UntrustedCallable())

    payload = candidate.to_dict()

    assert "implementation" not in payload
    assert "mutation_test_runner" not in payload
    assert not marker.exists()


def test_binary_pickle_payload_is_rejected_without_pickle_deserialization() -> None:
    pickle_payload = b"\x80\x04\x95\x12\x00\x00\x00malicious-pickle-payload"

    with pytest.raises(ValueError, match="sandbox_result_malformed"):
        decode_sandbox_result(pickle_payload)


def test_malformed_and_oversized_worker_messages_are_rejected() -> None:
    with pytest.raises(ValueError, match="sandbox_result_malformed"):
        decode_sandbox_result(b"not-json")

    oversized = json.dumps({"status": "success", "output": "x" * 128}).encode("utf-8")
    with pytest.raises(ValueError, match="sandbox_result_too_large"):
        decode_sandbox_result(oversized, max_bytes=64)

    with pytest.raises(ValueError, match="sandbox_result_unexpected_fields"):
        decode_sandbox_result(b'{"status":"success","unexpected":true}')

    with pytest.raises(ValueError, match="sandbox_result_invalid_duration"):
        decode_sandbox_result(b'{"status":"success","output":null,"duration_ms":"not-a-number"}')
    with pytest.raises(ValueError, match="sandbox_result_missing_output"):
        decode_sandbox_result(b'{"status":"success"}')
    with pytest.raises(ValueError, match="sandbox_result_missing_error"):
        decode_sandbox_result(b'{"status":"failed"}')


def test_primitive_worker_protocol_accepts_only_bounded_json_values() -> None:
    payload = json.dumps(
        {"status": "success", "output": {"result": "ok"}, "error": "", "duration_ms": 1.5}
    ).encode("utf-8")

    assert decode_sandbox_result(payload) == {
        "status": "success",
        "output": {"result": "ok"},
        "error": "",
        "duration_ms": 1.5,
    }


def test_skill_sandbox_fails_closed_without_executing_candidate_implementation(tmp_path: Path) -> None:
    marker = tmp_path / "implementation-called"

    def untrusted_implementation(_inputs: dict) -> dict:
        marker.write_text("called", encoding="utf-8")
        return {"result": "unexpected"}

    candidate = make_candidate(untrusted_implementation)
    result = SkillSandbox(SandboxCapabilities.detect()).execute(
        candidate,
        SkillContract.from_candidate(candidate),
        {"value": 1},
    )

    assert result.status == "QUARANTINED"
    assert result.error == "external_sandbox_unavailable"
    assert not marker.exists()


def test_candidate_supplied_mutation_runner_is_never_called_in_host(tmp_path: Path) -> None:
    marker = tmp_path / "mutation-runner-called"

    def untrusted_runner(_source: str, _mutant) -> bool:
        marker.write_text("called", encoding="utf-8")
        return True

    candidate = make_candidate(lambda _inputs: {"result": "unexpected"}, untrusted_runner)
    report = MutationTestingEngine(max_mutants=1, threshold=1.0).run(candidate)

    assert report.passed is False
    assert report.reason == "external_sandbox_required"
    assert not marker.exists()


def test_submit_retains_static_analysis_but_quarantines_runtime_execution(tmp_path: Path) -> None:
    implementation_calls: list[str] = []
    runner_calls: list[str] = []

    def untrusted_implementation(_inputs: dict) -> dict:
        implementation_calls.append("called")
        return {"result": "unexpected"}

    def untrusted_runner(_source: str, _mutant) -> bool:
        runner_calls.append("called")
        return True

    store = Store(tmp_path / "phase15d0.sqlite3")
    lifecycle = SkillLifecycleManager(
        SkillRegistry(store),
        mutation_engine=MutationTestingEngine(max_mutants=1, threshold=1.0),
        security_controller=SecurityController(store=store),
    )
    candidate = make_candidate(untrusted_implementation, untrusted_runner)

    result = lifecycle.submit(candidate, inputs={"value": 1})

    assert result.analysis.passed is True
    assert result.candidate.status == "QUARANTINED"
    assert result.mutation.reason == "external_sandbox_required"
    assert result.sandbox.error == "external_sandbox_required"
    assert implementation_calls == []
    assert runner_calls == []
    store.close()


def test_static_analysis_and_contract_validation_remain_available_without_execution() -> None:
    candidate = make_candidate(None, None)
    contract = SkillContract.from_candidate(candidate)

    assert SkillContractValidator().validate(candidate, contract).passed is True
    assert SkillStaticAnalyzer().analyze(candidate, contract).passed is True


def test_lifecycle_rejects_injected_execution_providers() -> None:
    class FakeMutationEngine:
        pass

    class FakeSandbox:
        pass

    store = Store(":memory:")
    with pytest.raises(TypeError, match="untrusted_mutation_engine_not_supported"):
        SkillLifecycleManager(SkillRegistry(store), mutation_engine=FakeMutationEngine())
    with pytest.raises(TypeError, match="untrusted_sandbox_provider_not_supported"):
        SkillLifecycleManager(SkillRegistry(store), sandbox=FakeSandbox())
    with pytest.raises(TypeError, match="untrusted_sandbox_capabilities_not_supported"):
        SkillSandbox(FakeSandbox())
    store.close()


def test_activation_guard_rejects_provider_replacement_without_side_effects(tmp_path: Path, monkeypatch) -> None:
    implementation_calls: list[str] = []

    def untrusted_implementation(_inputs: dict) -> dict:
        implementation_calls.append("called")
        return {"result": "unexpected"}

    class FakeSandbox:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _candidate, _contract, _inputs):
            self.calls += 1
            raise AssertionError("injected sandbox must not execute")

    store = Store(tmp_path / "activation-guard.sqlite3")
    lifecycle = SkillLifecycleManager(
        SkillRegistry(store),
        mutation_engine=MutationTestingEngine(max_mutants=1, threshold=1.0),
        security_controller=SecurityController(store=store),
    )
    fake = FakeSandbox()
    lifecycle.sandbox = fake
    passed_mutation = MutationReport(
        True,
        1,
        1,
        0,
        1.0,
        lifecycle.mutation_engine.threshold,
        reason="trusted_static_test_gate",
    )
    monkeypatch.setattr(
        lifecycle.mutation_engine,
        "run",
        lambda _candidate, _runner=None: passed_mutation,
    )

    result = lifecycle.submit(make_candidate(untrusted_implementation), inputs={"value": 1})

    assert result.approval.status == "QUARANTINED"
    assert result.sandbox.error == "untrusted_sandbox_provider_not_supported"
    assert fake.calls == 0
    assert implementation_calls == []
    store.close()
