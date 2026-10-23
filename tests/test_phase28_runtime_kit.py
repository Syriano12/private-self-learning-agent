from __future__ import annotations

from types import SimpleNamespace

import pytest

from private_agent.bugbounty.scanners import ScannerLimits
from private_agent.tools.scanners import HttpxReconTool
from private_agent.skills.sandbox_runtime import RuntimeActivationProof
from private_agent.skills.runtime_preflight import HostPreflightStatus, run_host_preflight


def test_current_or_missing_runtime_is_reported_without_claiming_isolation() -> None:
    report = run_host_preflight(which=lambda _name: None)

    assert report.verdict is HostPreflightStatus.FAIL
    runtime = report.check("oci_runtime")
    assert runtime.status is HostPreflightStatus.FAIL
    assert runtime.detail == "runtime_missing"
    assert report.check("filesystem_restrictions").status is not HostPreflightStatus.PASS
    assert report.host_ready is False
    assert report.can_run_acceptance is False


def test_runtime_presence_alone_is_not_a_pass() -> None:
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="5.0.0\n", stderr=""),
            SimpleNamespace(
                returncode=0,
                stdout='{"host":{"security":{"rootless":false,"seccompProfile":""}}}',
                stderr="",
            ),
        ]
    )

    def run(*_args, **_kwargs):
        return next(responses)

    report = run_host_preflight(
        which=lambda name: "/usr/bin/podman" if name == "podman" else None,
        run=run,
    )

    assert report.verdict is HostPreflightStatus.FAIL
    assert report.check("oci_runtime").status is HostPreflightStatus.FAIL
    assert "rootless" in report.check("oci_runtime").detail


def test_scanner_constructor_rejects_legacy_activation_flag(tmp_path) -> None:
    with pytest.raises(TypeError, match="external_sandbox_available_flag_not_supported"):
        HttpxReconTool(
            binary_path=str(tmp_path / "httpx"),
            limits=ScannerLimits(timeout_seconds=1.0, max_output_bytes=4096),
            external_sandbox_available=True,
        )


def test_runtime_activation_proof_has_no_public_constructor() -> None:
    with pytest.raises(TypeError, match="runtime_activation_proof_private"):
        RuntimeActivationProof()


def test_runtime_activation_proof_rejects_mutation() -> None:
    proof = object.__new__(RuntimeActivationProof)
    with pytest.raises(AttributeError, match="runtime_activation_proof_immutable"):
        proof.runtime_identity = "podman@9.9.9"


def test_scanner_without_runtime_proof_remains_quarantined(tmp_path) -> None:
    tool = HttpxReconTool(binary_path=str(tmp_path / "httpx"))
    result = tool.execute({}, None)

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "scanner_sandbox_unavailable"
