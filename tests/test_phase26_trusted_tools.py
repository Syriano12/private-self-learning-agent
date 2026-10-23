from __future__ import annotations

import os
import stat
from pathlib import Path

from private_agent.bugbounty.scanners import ProjectDiscoveryProbe
from private_agent.bugbounty.trusted_tools import (
    ToolVerificationStatus,
    TrustedToolInstallSpec,
    load_tool_spec_from_environment,
    verify_tool_installation,
)
from private_agent.tools.scanners import HttpxReconTool, NucleiScanTool


def _spec(tmp_path: Path, *, binary: Path | None = None) -> TrustedToolInstallSpec:
    binary = binary or tmp_path / "httpx"
    return TrustedToolInstallSpec(
        tool="httpx",
        version="1.12.0",
        binary_path=binary,
        archive_path=tmp_path / "httpx.zip",
        archive_member="httpx",
        release_asset_name="httpx_1.12.0_linux_amd64.zip",
        release_url="https://github.com/projectdiscovery/httpx/releases/download/v1.12.0/httpx_1.12.0_linux_amd64.zip",
        checksum_file=tmp_path / "checksums.txt",
        signature_file=tmp_path / "checksums.txt.sig",
        keyring_path=tmp_path / "projectdiscovery-release.gpg",
        signer_fingerprint="A" * 40,
    )


def test_missing_environment_provenance_is_unavailable(monkeypatch):
    for key in list(os.environ):
        if key.startswith("AGENT_HTTPX_") or key.startswith("AGENT_NUCLEI_"):
            monkeypatch.delenv(key, raising=False)
    assert load_tool_spec_from_environment("httpx", "/usr/local/bin/httpx") is None


def test_invalid_signer_fingerprint_is_rejected(tmp_path: Path):
    try:
        TrustedToolInstallSpec(
            tool="httpx",
            version="1.12.0",
            binary_path=tmp_path / "httpx",
            archive_path=tmp_path / "httpx.zip",
            archive_member="httpx",
            release_asset_name="httpx_1.12.0_linux_amd64.zip",
            release_url="https://github.com/projectdiscovery/httpx/releases/download/v1.12.0/httpx_1.12.0_linux_amd64.zip",
            checksum_file=tmp_path / "checksums.txt",
            signature_file=tmp_path / "checksums.txt.sig",
            keyring_path=tmp_path / "projectdiscovery-release.gpg",
            signer_fingerprint="not-a-fingerprint",
        )
    except ValueError as exc:
        assert str(exc) == "trusted_tool_signer_fingerprint_invalid"
    else:
        raise AssertionError("invalid signer fingerprint was accepted")


def test_missing_install_artifacts_are_not_verified(tmp_path: Path):
    result = verify_tool_installation(_spec(tmp_path))
    assert result.status is ToolVerificationStatus.UNAVAILABLE
    assert result.reason == "artifact_missing"
    assert result.verified is False


def test_symlink_binary_is_rejected_before_hashing(tmp_path: Path):
    real = tmp_path / "real-httpx"
    real.write_bytes(b"fixture")
    real.chmod(real.stat().st_mode | stat.S_IXUSR)
    link = tmp_path / "httpx"
    link.symlink_to(real)
    result = verify_tool_installation(_spec(tmp_path, binary=link))
    assert result.status is ToolVerificationStatus.UNAVAILABLE
    assert result.reason == "binary_symlink_rejected"


def test_probe_requires_trusted_provenance_without_running_binary(tmp_path: Path):
    binary = tmp_path / "httpx"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)

    report = ProjectDiscoveryProbe(require_trusted_provenance=True).probe("httpx", str(binary), required_flags=())
    assert report.available is False
    assert report.reason == "trusted_provenance_required"


def test_registered_scanners_are_quarantined_without_external_sandbox(tmp_path: Path):
    marker = tmp_path / "executed"
    binary = tmp_path / "httpx"
    binary.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    payload = {"scope": {}, "target": "http://127.0.0.1:8765/", "identity": "alice", "resource_id": "r1"}

    httpx_result = HttpxReconTool(binary_path=str(binary)).execute(payload, None)
    nuclei_result = NucleiScanTool(binary_path=str(binary)).execute(payload, None)

    assert httpx_result.status == "failed"
    assert httpx_result.error is not None and httpx_result.error.code == "scanner_sandbox_unavailable"
    assert nuclei_result.status == "failed"
    assert nuclei_result.error is not None and nuclei_result.error.code == "scanner_sandbox_unavailable"
    assert not marker.exists()
