from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from private_agent.skills.sandbox_contract import (
    RuntimeCapabilityReport,
    SandboxResult,
    SandboxSpec,
    SandboxUnavailable,
    ValidationOutcome,
    detect_podman,
)
from private_agent.skills.sandbox_runtime import (
    REQUIRED_RUNTIME_CONTROLS,
    ExternalSandboxRunner,
    PodmanSandboxAdapter,
    SandboxAttestation,
    SandboxRuntimeAdapter,
    require_runtime_capability,
    validate_attestation,
    validate_sandbox_result,
)


IMAGE_DIGEST = "sha256:" + "a" * 64
MANIFEST_DIGEST = "sha256:" + "b" * 64


def valid_spec(**overrides: object) -> SandboxSpec:
    values: dict[str, object] = {
        "runtime_name": "podman",
        "runtime_version": "5.0.0",
        "image_digest": IMAGE_DIGEST,
        "command_manifest_digest": MANIFEST_DIGEST,
        "timeout_seconds": 30.0,
        "cpu_limit_millis": 1000,
        "memory_limit_bytes": 64 * 1024 * 1024,
        "process_limit": 16,
        "output_limit_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return SandboxSpec(**values)  # type: ignore[arg-type]


def unavailable_report() -> RuntimeCapabilityReport:
    return detect_podman(which=lambda _name: None)


def forged_attestation(
    *,
    controls: tuple[str, ...] = tuple(sorted(REQUIRED_RUNTIME_CONTROLS)),
    outcome: ValidationOutcome = ValidationOutcome.VALID,
    runtime_identity: str = "podman@5.0.0",
    image_digest: str = IMAGE_DIGEST,
    command_manifest_digest: str = MANIFEST_DIGEST,
) -> SandboxAttestation:
    return SandboxAttestation(
        runtime_identity=runtime_identity,
        image_digest=image_digest,
        command_manifest_digest=command_manifest_digest,
        verified_controls=controls,
        evidence={control: f"evidence:{control}" for control in controls},
        validation_outcome=outcome,
    )


def test_runtime_adapter_interface_is_abstract_and_podman_adapter_is_read_only() -> None:
    with pytest.raises(TypeError):
        SandboxRuntimeAdapter()  # type: ignore[abstract]
    assert isinstance(PodmanSandboxAdapter(), SandboxRuntimeAdapter)


def test_required_controls_cover_every_external_boundary() -> None:
    assert REQUIRED_RUNTIME_CONTROLS == frozenset(
        {
            "runtime_identity",
            "image_identity",
            "command_manifest_provenance",
            "rootless",
            "resource_limits",
            "process_cleanup",
            "filesystem_restrictions",
            "network_policy",
            "output_limits",
        }
    )


def test_attestation_is_immutable_and_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError, match="attestation_runtime_identity_invalid"):
        SandboxAttestation(
            runtime_identity="host@1.0.0",
            image_digest=IMAGE_DIGEST,
            command_manifest_digest=MANIFEST_DIGEST,
        )

    attestation = forged_attestation(outcome=ValidationOutcome.SANDBOX_UNAVAILABLE)
    with pytest.raises(FrozenInstanceError):
        attestation.runtime_identity = "podman@5.0.1"  # type: ignore[misc]
    with pytest.raises(TypeError, match="does not support item assignment"):
        attestation.evidence["rootless"] = "tampered"  # type: ignore[index]

    with pytest.raises(ValueError, match="attestation_factory_forbidden"):
        SandboxAttestation._from_trusted_adapter(
            runtime_identity="podman@5.0.0",
            image_digest=IMAGE_DIGEST,
            command_manifest_digest=MANIFEST_DIGEST,
            verified_controls=tuple(sorted(REQUIRED_RUNTIME_CONTROLS)),
            evidence={control: "evidence" for control in REQUIRED_RUNTIME_CONTROLS},
            token=object(),
        )


def test_missing_attestation_controls_fail_closed_before_runtime_report() -> None:
    spec = valid_spec()
    attestation = forged_attestation(controls=("runtime_identity",))

    with pytest.raises(SandboxUnavailable, match="attestation_controls_missing"):
        validate_attestation(spec, unavailable_report(), attestation)


def test_forged_complete_attestation_is_not_trusted() -> None:
    spec = valid_spec()
    attestation = forged_attestation()

    with pytest.raises(SandboxUnavailable, match="attestation_not_trusted"):
        validate_attestation(spec, unavailable_report(), attestation)


def test_attestation_identity_mismatch_fails_before_execution() -> None:
    spec = valid_spec()
    attestation = forged_attestation(image_digest="sha256:" + "c" * 64)

    with pytest.raises(SandboxUnavailable, match="attestation_image_mismatch"):
        validate_attestation(spec, unavailable_report(), attestation)


def test_attestation_unknown_outcome_is_unavailable() -> None:
    spec = valid_spec()
    attestation = forged_attestation(outcome=ValidationOutcome.VALIDATION_UNKNOWN)

    with pytest.raises(SandboxUnavailable, match="attestation_not_trusted"):
        validate_attestation(spec, unavailable_report(), attestation)


def test_runtime_report_with_caller_supplied_valid_fields_is_not_available() -> None:
    report = RuntimeCapabilityReport(
        runtime_name="podman",
        executable="/usr/bin/podman",
        version="5.0.0",
        detected=True,
        version_verified=True,
        rootless_verified=True,
        security_controls_verified=True,
        image_identity_verified=True,
        command_manifest_verified=True,
        validation_outcome=ValidationOutcome.VALID,
        command_manifest_pinned=True,
        command_manifest_provenance_verified=True,
        network_policy_verified=True,
    )

    assert report.available is False
    with pytest.raises(SandboxUnavailable, match="runtime_capability_unavailable"):
        require_runtime_capability(report)


def test_result_validation_rejects_wrong_type_and_unknown_outcome() -> None:
    with pytest.raises(SandboxUnavailable, match="sandbox_result_malformed"):
        validate_sandbox_result(valid_spec(), forged_attestation(), object())  # type: ignore[arg-type]

    unknown = SandboxResult(status="FAILED", error="not validated")
    with pytest.raises(SandboxUnavailable, match="sandbox_result_not_validated"):
        validate_sandbox_result(valid_spec(), forged_attestation(), unknown)


def test_result_timeout_metadata_remains_bounded_and_explicit() -> None:
    result = SandboxResult(
        status="FAILED",
        error="timeout",
        timed_out=True,
        duration_ms=30.5,
        validation_outcome=ValidationOutcome.VALIDATION_UNKNOWN,
    )

    assert result.timed_out is True
    assert result.duration_ms == 30.5
    with pytest.raises(ValueError, match="duration_ms_invalid"):
        SandboxResult(status="FAILED", duration_ms=-1)


def test_result_identity_mismatch_is_rejected_before_any_success_claim() -> None:
    result = SandboxResult(
        status="FAILED",
        runtime_identity="podman@5.0.0",
        image_digest=IMAGE_DIGEST,
        command_manifest_digest=MANIFEST_DIGEST,
        validation_outcome=ValidationOutcome.VALIDATION_UNKNOWN,
    )

    with pytest.raises(SandboxUnavailable, match="sandbox_result_not_validated"):
        validate_sandbox_result(valid_spec(), forged_attestation(), result)


def test_podman_adapter_attestation_never_fabricates_missing_controls() -> None:
    adapter = PodmanSandboxAdapter()

    with pytest.raises(SandboxUnavailable, match="runtime_missing"):
        adapter.attest(valid_spec(), unavailable_report())

    with pytest.raises(SandboxUnavailable, match="podman_execution_not_enabled"):
        adapter.execute(valid_spec(), forged_attestation())


def test_external_runner_returns_quarantine_when_runtime_is_missing() -> None:
    result = ExternalSandboxRunner().execute(valid_spec())

    assert result.status == "QUARANTINED"
    assert result.validation_outcome is ValidationOutcome.SANDBOX_UNAVAILABLE
    assert result.error == "runtime_missing"


def test_external_runner_rejects_non_spec_inputs_without_side_effects() -> None:
    result = ExternalSandboxRunner().execute(object())  # type: ignore[arg-type]

    assert result.status == "QUARANTINED"
    assert result.error == "invalid_sandbox_spec"
    assert result.validation_outcome is ValidationOutcome.SANDBOX_UNAVAILABLE


def test_external_runner_does_not_accept_a_custom_execution_adapter() -> None:
    class HostFallbackAdapter(PodmanSandboxAdapter):
        def execute(self, spec: SandboxSpec, attestation: SandboxAttestation) -> SandboxResult:
            raise AssertionError("host fallback must never be called")

    with pytest.raises(TypeError, match="custom_adapter_injection_disabled"):
        ExternalSandboxRunner(adapter=HostFallbackAdapter())  # type: ignore[arg-type]


def test_external_runner_does_not_accept_mutated_builtin_adapter_instance() -> None:
    adapter = PodmanSandboxAdapter()
    adapter.inspect = lambda _spec: (_ for _ in ()).throw(AssertionError("injected inspect called"))  # type: ignore[method-assign]

    with pytest.raises(TypeError, match="custom_adapter_injection_disabled"):
        ExternalSandboxRunner(adapter=adapter)


def test_external_runner_does_not_expose_mutable_adapter_state() -> None:
    runner = ExternalSandboxRunner()

    with pytest.raises(AttributeError):
        runner._adapter = PodmanSandboxAdapter()  # type: ignore[attr-defined]
