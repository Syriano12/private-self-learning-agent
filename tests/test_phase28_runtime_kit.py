from __future__ import annotations

from types import SimpleNamespace

import pytest

from private_agent.bugbounty.scanners import ScannerLimits
from private_agent.tools.scanners import HttpxReconTool
from private_agent.skills.sandbox_runtime import RuntimeActivationProof
from private_agent.skills.runtime_preflight import HostPreflightStatus, _check_cgroups, run_host_preflight


_CGROUP2_MOUNTINFO = "36 25 0:32 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n"


def test_current_or_missing_runtime_is_reported_without_claiming_isolation() -> None:
    report = run_host_preflight(which=lambda _name: None)

    assert report.verdict is HostPreflightStatus.FAIL
    runtime = report.check("oci_runtime")
    assert runtime.status is HostPreflightStatus.FAIL
    assert runtime.detail == "runtime_missing"
    assert report.check("filesystem_restrictions").status is not HostPreflightStatus.PASS
    assert report.host_ready is False
    assert report.can_run_acceptance is False


def test_cgroup_preflight_checks_current_scope_and_writable_delegation() -> None:
    scope = "/sys/fs/cgroup/test.scope"
    files = {
        "/proc/self/cgroup": "0::/test.scope\n",
        "/proc/self/mountinfo": _CGROUP2_MOUNTINFO,
        f"{scope}/cgroup.controllers": "cpu memory pids\n",
        f"{scope}/cgroup.subtree_control": "",
    }
    writable_paths: set[str] = set()

    def read(path: str) -> str:
        return files.get(path, "")

    def access(path: str, mode: int, *, effective_ids: bool) -> bool:
        assert effective_ids is True
        return mode != 0 and path in writable_paths

    unproven = _check_cgroups(read, access)
    assert unproven.status is HostPreflightStatus.INCONCLUSIVE
    assert unproven.detail == "cgroup_delegation_unverified"
    assert unproven.evidence["scope_writable"] is False
    assert unproven.evidence["missing_enabled_controllers"] == ["cpu", "memory", "pids"]

    writable_paths.update({scope, f"{scope}/cgroup.subtree_control"})
    ready = _check_cgroups(read, access)
    assert ready.status is HostPreflightStatus.INCONCLUSIVE
    assert ready.detail == "cgroup_runtime_enforcement_unverified"
    assert ready.evidence["runtime_enforcement_verified"] is False

    files[f"{scope}/cgroup.subtree_control"] = "cpu memory pids\n"
    writable_paths.clear()
    writable_paths.add(scope)
    already_enabled = _check_cgroups(read, access)
    assert already_enabled.status is HostPreflightStatus.INCONCLUSIVE
    assert already_enabled.evidence["scope_writable"] is True


def test_cgroup_preflight_does_not_use_root_controllers_as_scope_proof() -> None:
    scope = "/sys/fs/cgroup/test.scope"
    files = {
        "/proc/self/cgroup": "0::/test.scope\n",
        "/proc/self/mountinfo": _CGROUP2_MOUNTINFO,
        "/sys/fs/cgroup/cgroup.controllers": "cpu memory pids\n",
        f"{scope}/cgroup.controllers": "memory pids\n",
        f"{scope}/cgroup.subtree_control": "",
    }

    result = _check_cgroups(
        lambda path: files.get(path, ""),
        lambda _path, _mode, **_kwargs: True,
    )

    assert result.status is HostPreflightStatus.INCONCLUSIVE
    assert result.detail == "resource_controllers_unverified_in_current_cgroup"
    assert result.evidence["available_controllers"] == ["memory", "pids"]
    assert result.evidence["missing_controllers"] == ["cpu"]


def test_cgroup_preflight_rejects_unverified_mount_root() -> None:
    files = {
        "/proc/self/cgroup": "0::/test.scope\n",
        "/proc/self/mountinfo": "36 25 0:32 /user.slice /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
    }

    result = _check_cgroups(lambda path: files.get(path, ""))

    assert result.status is HostPreflightStatus.INCONCLUSIVE
    assert result.detail == "cgroup_v2_mount_mapping_unverified"


def test_cgroup_preflight_rejects_duplicate_matching_mounts() -> None:
    files = {
        "/proc/self/cgroup": "0::/test.scope\n",
        "/proc/self/mountinfo": _CGROUP2_MOUNTINFO
        + "37 25 0:33 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
    }

    result = _check_cgroups(lambda path: files.get(path, ""))

    assert result.status is HostPreflightStatus.INCONCLUSIVE
    assert result.detail == "cgroup_v2_mount_mapping_unverified"
    assert result.evidence["matching_mount_count"] == 2


def test_cgroup_preflight_rejects_multiple_unified_memberships() -> None:
    result = _check_cgroups(lambda path: "0::/one\n0::/two\n" if path == "/proc/self/cgroup" else "")

    assert result.status is HostPreflightStatus.INCONCLUSIVE
    assert result.detail == "cgroup_membership_unverified"


def test_cgroup_preflight_rejects_untrusted_membership_paths() -> None:
    result = _check_cgroups(lambda _path: "0::/../../etc\n")

    assert result.status is HostPreflightStatus.INCONCLUSIVE
    assert result.detail == "cgroup_membership_unverified"


def test_cgroup_preflight_access_errors_remain_inconclusive() -> None:
    scope = "/sys/fs/cgroup/test.scope"
    files = {
        "/proc/self/cgroup": "0::/test.scope\n",
        "/proc/self/mountinfo": _CGROUP2_MOUNTINFO,
        f"{scope}/cgroup.controllers": "cpu memory pids\n",
        f"{scope}/cgroup.subtree_control": "cpu memory pids\n",
    }

    def fail_access(*_args, **_kwargs):
        raise OSError("permission metadata unavailable")

    result = _check_cgroups(lambda path: files.get(path, ""), fail_access)

    assert result.status is HostPreflightStatus.INCONCLUSIVE
    assert result.detail == "cgroup_delegation_unverified"


def test_android_termux_host_is_not_accepted_for_scanner_isolation() -> None:
    report = run_host_preflight(
        which=lambda _name: None,
        environment={"PREFIX": "/data/data/com.termux/files/usr", "TERMUX_VERSION": "0.118"},
    )

    assert report.check("host_platform").status is HostPreflightStatus.FAIL
    assert report.check("host_platform").detail == "android_termux_scanner_runtime_unverified"
    assert report.host_ready is False
    assert report.can_run_acceptance is False


def test_zero_disk_stat_on_termux_is_inconclusive_not_proof_of_full_storage() -> None:
    report = run_host_preflight(
        which=lambda _name: None,
        environment={"PREFIX": "/data/data/com.termux/files/usr", "TERMUX_VERSION": "0.118"},
        statvfs=lambda _path: SimpleNamespace(f_bavail=0, f_frsize=4096),
    )

    disk = report.check("disk_space")
    assert disk.status is HostPreflightStatus.INCONCLUSIVE
    assert disk.detail == "disk_stat_zero_on_android_termux"
    assert disk.evidence["available_bytes"] == 0
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
