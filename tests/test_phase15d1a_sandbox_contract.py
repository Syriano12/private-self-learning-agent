from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from private_agent.skills.lifecycle import SandboxCapabilities, SkillCandidate, SkillContract, SkillSandbox
from private_agent.skills.sandbox_contract import (
    MAX_CPU_LIMIT_MILLIS,
    MAX_MEMORY_LIMIT_BYTES,
    MAX_OUTPUT_LIMIT_BYTES,
    MAX_PROCESS_LIMIT,
    MAX_TIMEOUT_SECONDS,
    SandboxResult,
    SandboxSpec,
    SandboxUnavailable,
    ValidationOutcome,
    RuntimeCapabilityReport,
    detect_podman,
    require_available,
)


DIGEST = "sha256:" + "a" * 64


def valid_spec(**overrides) -> SandboxSpec:
    values = {
        "runtime_name": "podman",
        "runtime_version": "5.0.0",
        "image_digest": DIGEST,
        "command_manifest_digest": "sha256:" + "b" * 64,
        "timeout_seconds": 30.0,
        "cpu_limit_millis": 1000,
        "memory_limit_bytes": 64 * 1024 * 1024,
        "process_limit": 16,
        "output_limit_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return SandboxSpec(**values)


def fake_podman_runner(*, version: str = "5.0.0", image_digest: str = DIGEST, info: dict | None = None):
    host_info = info or {
        "host": {
            "security": {"rootless": True, "seccompProfile": "builtin"},
            "cgroupVersion": "v2",
            "ociRuntime": {"name": "crun"},
        }
    }

    def run(argv, **_kwargs):
        if argv[1:3] == ["version", "--format"]:
            return SimpleNamespace(returncode=0, stdout=version, stderr="")
        if argv[1:3] == ["info", "--format"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(host_info), stderr="")
        if argv[1:3] == ["image", "inspect"]:
            return SimpleNamespace(returncode=0, stdout=image_digest, stderr="")
        raise AssertionError(f"unexpected probe: {argv}")

    return run


def test_missing_runtime_is_unavailable_without_fallback() -> None:
    report = detect_podman(which=lambda _name: None)

    assert report.detected is False
    assert report.version_verified is False
    assert report.validation_outcome is ValidationOutcome.SANDBOX_UNAVAILABLE
    assert report.available is False
    assert report.reasons == ("runtime_missing",)


def test_malformed_runtime_version_is_unavailable() -> None:
    report = detect_podman(
        which=lambda _name: "/usr/bin/podman",
        run=fake_podman_runner(version="podman-five"),
    )

    assert report.detected is True
    assert report.version_verified is False
    assert report.validation_outcome is ValidationOutcome.SANDBOX_UNAVAILABLE
    assert "runtime_version_malformed" in report.reasons


def test_missing_or_unpinned_image_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="image_digest_must_be_pinned_sha256"):
        valid_spec(image_digest="latest")
    with pytest.raises(ValueError, match="image_digest_must_be_pinned_sha256"):
        valid_spec(image_digest="sha256:" + "a" * 63)

    report = detect_podman(
        which=lambda _name: "/usr/bin/podman",
        run=fake_podman_runner(),
    )
    assert report.validation_outcome is ValidationOutcome.SANDBOX_UNAVAILABLE
    assert "pinned_image_identity_missing" in report.reasons
    assert "command_manifest_identity_missing" in report.reasons


def test_verified_runtime_and_image_still_remain_unavailable_without_trusted_manifest_and_network_proof() -> None:
    report = detect_podman(
        valid_spec(),
        which=lambda _name: "/usr/bin/podman",
        run=fake_podman_runner(),
    )

    assert report.available is False
    assert report.validation_outcome is ValidationOutcome.SANDBOX_UNAVAILABLE
    assert report.version_verified is True
    assert report.rootless_verified is True
    assert report.security_controls_verified is True
    assert report.image_identity_verified is True
    assert report.command_manifest_verified is False
    assert report.command_manifest_pinned is True
    assert report.command_manifest_provenance_verified is False
    assert report.network_policy_verified is False
    assert "command_manifest_provenance_unverified" in report.reasons
    assert "network_policy_unverified" in report.reasons
    assert report.verified_controls == ("rootless", "seccomp")


def test_unsupported_security_control_is_unavailable() -> None:
    report = detect_podman(
        valid_spec(required_security_controls=("rootless", "apparmor")),
        which=lambda _name: "/usr/bin/podman",
        run=fake_podman_runner(),
    )

    assert report.available is False
    assert report.validation_outcome is ValidationOutcome.SANDBOX_UNAVAILABLE
    assert "security_control_unsupported:apparmor" in report.reasons


def test_unverified_runtime_security_fields_fail_closed() -> None:
    report = detect_podman(
        valid_spec(),
        which=lambda _name: "/usr/bin/podman",
        run=fake_podman_runner(info={"host": {"security": {"rootless": False}}}),
    )

    assert report.available is False
    assert "rootless_unverified" in report.reasons
    assert "seccomp_unverified" in report.reasons


def test_runtime_probe_rejects_oversized_output_and_uses_no_shell() -> None:
    calls = []

    def oversized_runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="x" * (64 * 1024 + 1), stderr="")

    report = detect_podman(
        which=lambda _name: "/usr/bin/podman",
        run=oversized_runner,
    )

    assert report.available is False
    assert report.reasons == ("runtime_probe_output_too_large",)
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["timeout"] == 2.0


def test_runtime_info_rejects_duplicate_keys_and_non_standard_json_constants() -> None:
    def duplicate_key_runner(argv, **kwargs):
        if argv[1:3] == ["version", "--format"]:
            return SimpleNamespace(returncode=0, stdout="5.0.0", stderr="")
        return SimpleNamespace(returncode=0, stdout='{"host": {}, "host": {}}', stderr="")

    duplicate = detect_podman(which=lambda _name: "/usr/bin/podman", run=duplicate_key_runner)
    assert duplicate.reasons == ("runtime_info_malformed",)

    def constant_runner(argv, **kwargs):
        if argv[1:3] == ["version", "--format"]:
            return SimpleNamespace(returncode=0, stdout="5.0.0", stderr="")
        return SimpleNamespace(returncode=0, stdout='{"host": NaN}', stderr="")

    constant = detect_podman(which=lambda _name: "/usr/bin/podman", run=constant_runner)
    assert constant.reasons == ("runtime_info_malformed",)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timeout_seconds", 0, "timeout_seconds_out_of_bounds"),
        ("timeout_seconds", MAX_TIMEOUT_SECONDS + 1, "timeout_seconds_out_of_bounds"),
        ("cpu_limit_millis", 0, "cpu_limit_millis_out_of_bounds"),
        ("cpu_limit_millis", MAX_CPU_LIMIT_MILLIS + 1, "cpu_limit_millis_out_of_bounds"),
        ("memory_limit_bytes", 1024, "memory_limit_bytes_out_of_bounds"),
        ("memory_limit_bytes", MAX_MEMORY_LIMIT_BYTES + 1, "memory_limit_bytes_out_of_bounds"),
        ("process_limit", 0, "process_limit_out_of_bounds"),
        ("process_limit", MAX_PROCESS_LIMIT + 1, "process_limit_out_of_bounds"),
        ("output_limit_bytes", 0, "output_limit_bytes_out_of_bounds"),
        ("output_limit_bytes", MAX_OUTPUT_LIMIT_BYTES + 1, "output_limit_bytes_out_of_bounds"),
    ],
)
def test_resource_limits_are_bounded(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        valid_spec(**{field: value})


def test_spec_identity_is_immutable_and_host_fallback_is_rejected() -> None:
    spec = valid_spec()
    with pytest.raises(FrozenInstanceError):
        spec.runtime_version = "5.0.1"  # type: ignore[misc]
    with pytest.raises(ValueError, match="runtime_unsupported"):
        valid_spec(runtime_name="host")
    with pytest.raises(ValueError, match="network_policy_unsupported"):
        valid_spec(network_policy="host")
    with pytest.raises(ValueError, match="security_controls_invalid"):
        valid_spec(required_security_controls=None)


def test_require_available_raises_structured_sandbox_unavailable() -> None:
    report = detect_podman(which=lambda _name: None)

    with pytest.raises(SandboxUnavailable) as exc_info:
        require_available(report)

    assert exc_info.value.code == "SANDBOX_UNAVAILABLE"
    assert exc_info.value.reason == "runtime_missing"
    assert exc_info.value.to_dict()["details"] == ["runtime_missing"]


def test_result_represents_exit_signal_timeout_digests_and_validation_state() -> None:
    result = SandboxResult(
        status="FAILED",
        error="timed out",
        duration_ms=12.5,
        exit_code=None,
        signal=9,
        timed_out=True,
        runtime_identity="podman@5.0.0",
        image_digest=DIGEST,
        command_manifest_digest="sha256:" + "b" * 64,
        stdout_digest=DIGEST,
        stderr_digest="sha256:" + "c" * 64,
        output_digest="sha256:" + "d" * 64,
        validation_outcome=ValidationOutcome.VALIDATION_UNKNOWN,
    )

    payload = result.to_dict()
    assert payload["signal"] == 9
    assert payload["timed_out"] is True
    assert payload["stdout_digest"] == DIGEST
    assert payload["validation_outcome"] == "VALIDATION_UNKNOWN"


def test_result_rejects_forged_identity_or_verified_result_without_trusted_runner() -> None:
    with pytest.raises(ValueError, match="sandbox_result_runtime_identity_invalid"):
        SandboxResult(status="PASSED", runtime_identity="host@1.0.0")
    with pytest.raises(ValueError, match="sandbox_result_stdout_digest_invalid"):
        SandboxResult(status="FAILED", stdout_digest=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sandbox_result_verified_identity_missing"):
        SandboxResult(status="PASSED", validation_outcome=ValidationOutcome.VALID)
    with pytest.raises(ValueError, match="sandbox_result_verified_result_requires_trusted_runner"):
        SandboxResult(
            status="PASSED",
            runtime_identity="podman@5.0.0",
            image_digest=DIGEST,
            command_manifest_digest="sha256:" + "b" * 64,
            validation_outcome=ValidationOutcome.VALID,
        )


def test_forged_valid_capability_report_is_not_available_without_detector_attestation() -> None:
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
    with pytest.raises(SandboxUnavailable, match="capability_validation_failed"):
        require_available(report)


def test_existing_skill_sandbox_remains_fail_closed_without_host_execution(tmp_path: Path) -> None:
    marker = tmp_path / "host-execution-called"

    def implementation(_inputs: dict) -> dict:
        marker.write_text("called", encoding="utf-8")
        return {"result": "unsafe"}

    candidate = SkillCandidate(
        skill_id="phase15d1a",
        name="phase15d1a",
        version=1,
        description="contract test",
        source="def run(value): return {'result': 'safe'}",
        entry_point="run",
        implementation=implementation,
    )
    result = SkillSandbox(SandboxCapabilities.detect()).execute(
        candidate,
        SkillContract.from_candidate(candidate),
        {"value": 1},
    )

    assert result.status == "QUARANTINED"
    assert result.validation_outcome is ValidationOutcome.SANDBOX_UNAVAILABLE
    assert result.error == "external_sandbox_unavailable"
    assert not marker.exists()
