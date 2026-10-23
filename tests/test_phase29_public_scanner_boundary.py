from __future__ import annotations

import hashlib
import json
import stat
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from private_agent.bugbounty.models import ScopeDefinition
from private_agent.bugbounty.scanners import (
    HttpxAdapter,
    NucleiAdapter,
    ProjectDiscoveryProbe,
    SafeSubprocessRunner,
    ScanStatus,
    ScannerLimits,
)


def _scope() -> ScopeDefinition:
    now = datetime.now(timezone.utc)
    return ScopeDefinition.from_dict(
        {
            "program_id": "phase29-local",
            "authorization_reference": "phase29-written-authorization",
            "issued_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
            "assets": [
                {
                    "asset_id": "local-api",
                    "host": "127.0.0.1",
                    "path_prefix": "/",
                    "methods": ["GET"],
                    "ports": [8765],
                    "active_allowed": True,
                }
            ],
            "permitted_identities": ["alice"],
            "permitted_resources": ["resource-1"],
            "permitted_methods": ["GET"],
            "prohibited_methods": ["POST", "PUT", "PATCH", "DELETE"],
            "active_testing": True,
        }
    )


def _limits() -> ScannerLimits:
    return ScannerLimits(timeout_seconds=1.0, rate_limit_per_second=1, concurrency=1, max_output_bytes=4096, max_records=4)


def _fake_executable(path: Path, marker: Path, output: str = "") -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n" + textwrap.dedent(
            f"""
            import pathlib
            pathlib.Path({str(marker)!r}).write_text('called')
            print({output!r})
            """
        ),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _templates(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "templates"
    root.mkdir()
    template = root / "safe.yaml"
    template.write_text("id: phase29-safe\ninfo:\n  name: safe\n", encoding="utf-8")
    manifest = {
        "templates": [
            {
                "id": "phase29-safe",
                "path": "safe.yaml",
                "sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
                "tags": ["safe"],
            }
        ]
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_direct_httpx_adapter_is_quarantined_before_host_execution(tmp_path: Path) -> None:
    marker = tmp_path / "httpx-called"
    binary = _fake_executable(tmp_path / "httpx", marker, json.dumps({"status-code": 200}))

    result = HttpxAdapter(binary_path=str(binary), limits=_limits()).scan(
        _scope(), "http://127.0.0.1:8765/health", identity="alice", resource_id="resource-1"
    )

    assert result.status is ScanStatus.UNAVAILABLE
    assert "external_sandbox_executor_unavailable" in result.errors
    assert not marker.exists()


def test_direct_nuclei_adapter_is_quarantined_before_host_execution(tmp_path: Path) -> None:
    marker = tmp_path / "nuclei-called"
    binary = _fake_executable(tmp_path / "nuclei", marker, json.dumps({"template-id": "phase29-safe"}))
    templates, digest = _templates(tmp_path)

    result = NucleiAdapter(
        binary_path=str(binary),
        templates_root=templates,
        expected_manifest_sha256=digest,
        limits=_limits(),
    ).scan(_scope(), "http://127.0.0.1:8765/health", identity="alice", resource_id="resource-1")

    assert result.status is ScanStatus.UNAVAILABLE
    assert "external_sandbox_executor_unavailable" in result.errors
    assert not marker.exists()


def test_public_safe_subprocess_runner_is_quarantined(tmp_path: Path) -> None:
    marker = tmp_path / "runner-called"
    binary = _fake_executable(tmp_path / "runner", marker)

    result = SafeSubprocessRunner().run([str(binary)], timeout_seconds=0.1, max_output_bytes=4096)

    assert result.status is ScanStatus.UNAVAILABLE
    assert not marker.exists()


def test_probe_rejects_caller_supplied_runner(tmp_path: Path) -> None:
    class CountingRunner:
        calls = 0

        def run(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("probe must not execute a caller-provided runner")

    binary = _fake_executable(tmp_path / "httpx", tmp_path / "probe-called")
    with pytest.raises(TypeError):
        ProjectDiscoveryProbe(CountingRunner()).probe("httpx", str(binary), required_flags=())
