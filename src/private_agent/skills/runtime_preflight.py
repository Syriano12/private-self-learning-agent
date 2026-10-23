"""Read-only host preflight for the portable scanner runtime kit.

This module never installs packages, changes host configuration, starts a container,
or executes a scanner. A successful executable lookup is deliberately insufficient:
resource, isolation, provenance, and acceptance evidence remain separate gates.
"""
from __future__ import annotations

import json
import os
import platform
import pwd
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


class HostPreflightStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    REQUIRED_ACCEPTANCE = "REQUIRED_ACCEPTANCE"


_ACCEPTANCE_ONLY_CHECKS = frozenset(
    {"seccomp", "filesystem_restrictions", "network_isolation", "process_cleanup"}
)
_ACCEPTANCE_RUNNABLE_STATUSES = frozenset(
    {HostPreflightStatus.PASS, HostPreflightStatus.REQUIRED_ACCEPTANCE}
)
_HOST_REQUIRED_CHECKS = frozenset(
    {
        "os_kernel",
        "oci_runtime",
        "user_namespaces",
        "subordinate_ids",
        "rootless_helpers",
        "cgroups",
        "disk_space",
        "memory",
    }
)


@dataclass(frozen=True, slots=True)
class HostPreflightCheck:
    name: str
    status: HostPreflightStatus
    detail: str
    evidence: Mapping[str, Any] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "evidence": dict(self.evidence) if isinstance(self.evidence, Mapping) else {},
        }


@dataclass(frozen=True, slots=True)
class HostPreflightReport:
    checks: tuple[HostPreflightCheck, ...]

    @property
    def verdict(self) -> HostPreflightStatus:
        statuses = {check.status for check in self.checks}
        if HostPreflightStatus.FAIL in statuses:
            return HostPreflightStatus.FAIL
        if HostPreflightStatus.INCONCLUSIVE in statuses:
            return HostPreflightStatus.INCONCLUSIVE
        if HostPreflightStatus.REQUIRED_ACCEPTANCE in statuses:
            return HostPreflightStatus.REQUIRED_ACCEPTANCE
        return HostPreflightStatus.PASS

    @property
    def host_ready(self) -> bool:
        """Whether host capabilities are sufficient to start acceptance probes."""
        checks = {check.name: check for check in self.checks}
        return all(
            checks.get(name) is not None and checks[name].status is HostPreflightStatus.PASS
            for name in _HOST_REQUIRED_CHECKS
        )

    @property
    def can_run_acceptance(self) -> bool:
        """Allow acceptance to prove runtime-enforced controls, never activation."""
        if not self.host_ready:
            return False
        checks = {check.name: check for check in self.checks}
        return all(
            checks.get(name) is not None and checks[name].status in _ACCEPTANCE_RUNNABLE_STATUSES
            for name in _ACCEPTANCE_ONLY_CHECKS
        )

    @property
    def acceptance_required(self) -> bool:
        checks = {check.name: check for check in self.checks}
        return any(
            checks.get(name) is not None and checks[name].status is not HostPreflightStatus.PASS
            for name in _ACCEPTANCE_ONLY_CHECKS
        )

    @property
    def runtime_identity(self) -> str | None:
        runtime = next((check for check in self.checks if check.name == "oci_runtime"), None)
        version = runtime.evidence.get("version") if runtime is not None and isinstance(runtime.evidence, Mapping) else None
        if not isinstance(version, str) or not version:
            return None
        return f"podman@{version}"

    def check(self, name: str) -> HostPreflightCheck:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "host_ready": self.host_ready,
            "can_run_acceptance": self.can_run_acceptance,
            "acceptance_required": self.acceptance_required,
            "runtime_identity": self.runtime_identity,
            "checks": [check.to_dict() for check in self.checks],
        }


_RUNTIME_VERSION_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?(?:[-+][0-9A-Za-z.-]+)?$")
_MIN_DISK_BYTES = 2 * 1024 * 1024 * 1024
_MIN_MEMORY_BYTES = 512 * 1024 * 1024
_RUNTIME_PROBE_TIMEOUT = 3.0
_MAX_PROBE_OUTPUT = 128 * 1024


def run_host_preflight(
    *,
    runtime_name: str = "podman",
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., Any] = subprocess.run,
    read_text: Callable[[str], str] | None = None,
    root_path: str = "/",
) -> HostPreflightReport:
    """Collect bounded, read-only host checks without provisioning anything."""
    read = read_text or _read_text
    checks: list[HostPreflightCheck] = []
    checks.append(_check_os_kernel())

    runtime_path = which(runtime_name)
    runtime_check, runtime_info = _check_runtime(runtime_name, runtime_path, run)
    checks.append(runtime_check)
    checks.append(_check_user_namespaces(which, run, read))
    checks.append(_check_subordinate_ids(read, _current_username()))
    checks.append(_check_rootless_helpers(which))
    checks.append(_check_cgroups(read))
    checks.append(_check_seccomp(read))
    checks.append(_check_filesystem(runtime_path, read))
    checks.append(_check_network_isolation(runtime_path, runtime_info))
    checks.append(_check_process_cleanup())
    checks.append(_check_disk(root_path))
    checks.append(_check_memory(read))
    return HostPreflightReport(tuple(checks))


def _check_os_kernel() -> HostPreflightCheck:
    system = platform.system()
    release = platform.release()
    if system != "Linux":
        return HostPreflightCheck("os_kernel", HostPreflightStatus.FAIL, "linux_required", {"system": system, "release": release})
    if not release:
        return HostPreflightCheck("os_kernel", HostPreflightStatus.INCONCLUSIVE, "kernel_release_unavailable")
    return HostPreflightCheck("os_kernel", HostPreflightStatus.PASS, "linux_detected", {"system": system, "release": release})


def _check_runtime(runtime_name: str, executable: str | None, run: Callable[..., Any]) -> tuple[HostPreflightCheck, dict[str, Any] | None]:
    if not executable:
        return HostPreflightCheck("oci_runtime", HostPreflightStatus.FAIL, "runtime_missing", {"runtime": runtime_name}), None
    version_result, version_error = _probe(run, [executable, "version", "--format", "{{.Client.Version}}"])
    if version_result is None:
        return HostPreflightCheck("oci_runtime", HostPreflightStatus.FAIL, version_error or "runtime_version_probe_failed"), None
    version = str(getattr(version_result, "stdout", "")).strip()
    if version_result.returncode != 0 or not _RUNTIME_VERSION_RE.fullmatch(version):
        return HostPreflightCheck("oci_runtime", HostPreflightStatus.FAIL, "runtime_version_unverified", {"version": version[:128]}), None
    info_result, info_error = _probe(run, [executable, "info", "--format", "json"])
    if info_result is None or info_result.returncode != 0:
        return HostPreflightCheck("oci_runtime", HostPreflightStatus.FAIL, info_error or "runtime_info_probe_failed", {"version": version}), None
    try:
        info = json.loads(str(getattr(info_result, "stdout", "")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return HostPreflightCheck("oci_runtime", HostPreflightStatus.FAIL, "runtime_info_malformed", {"version": version}), None
    if not isinstance(info, dict):
        return HostPreflightCheck("oci_runtime", HostPreflightStatus.FAIL, "runtime_info_malformed", {"version": version}), None
    host = info.get("host") if isinstance(info.get("host"), dict) else {}
    security = host.get("security") if isinstance(host.get("security"), dict) else {}
    rootless = security.get("rootless") is True
    seccomp = isinstance(security.get("seccompProfile"), str) and bool(security.get("seccompProfile").strip())
    if not rootless:
        return HostPreflightCheck("oci_runtime", HostPreflightStatus.FAIL, "rootless_unverified", {"version": version, "seccomp": seccomp}), info
    if not seccomp:
        return HostPreflightCheck("oci_runtime", HostPreflightStatus.FAIL, "seccomp_unverified", {"version": version, "rootless": rootless}), info
    return HostPreflightCheck("oci_runtime", HostPreflightStatus.PASS, "runtime_metadata_verified", {"version": version, "rootless": rootless, "seccomp": seccomp}), info


def _check_user_namespaces(which: Callable[[str], str | None], run: Callable[..., Any], read: Callable[[str], str]) -> HostPreflightCheck:
    raw = read("/proc/sys/user/max_user_namespaces")
    try:
        maximum = int(raw.strip())
    except (TypeError, ValueError):
        return HostPreflightCheck("user_namespaces", HostPreflightStatus.INCONCLUSIVE, "user_namespace_limit_unreadable")
    if maximum <= 0:
        return HostPreflightCheck("user_namespaces", HostPreflightStatus.FAIL, "user_namespaces_disabled", {"max_user_namespaces": maximum})
    unshare = which("unshare")
    if not unshare:
        return HostPreflightCheck("user_namespaces", HostPreflightStatus.INCONCLUSIVE, "unshare_probe_unavailable", {"max_user_namespaces": maximum})
    result, error = _probe(run, [unshare, "--user", "--map-root-user", "id"])
    if result is None or result.returncode != 0:
        return HostPreflightCheck("user_namespaces", HostPreflightStatus.FAIL, error or "user_namespace_probe_failed", {"max_user_namespaces": maximum})
    return HostPreflightCheck("user_namespaces", HostPreflightStatus.PASS, "user_namespace_probe_passed", {"max_user_namespaces": maximum})


def _check_subordinate_ids(read: Callable[[str], str], username: str) -> HostPreflightCheck:
    values: dict[str, int] = {}
    for name in ("/etc/subuid", "/etc/subgid"):
        text = read(name)
        if not text:
            return HostPreflightCheck("subordinate_ids", HostPreflightStatus.INCONCLUSIVE, "mapping_file_unreadable", {"file": name})
        found = False
        for line in text.splitlines():
            parts = line.split(":")
            if len(parts) != 3 or parts[0] != username:
                continue
            try:
                start, length = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if start >= 0 and length > 0:
                values[name] = length
                found = True
                break
        if not found:
            return HostPreflightCheck("subordinate_ids", HostPreflightStatus.FAIL, "mapping_missing", {"file": name, "user": username})
    return HostPreflightCheck("subordinate_ids", HostPreflightStatus.PASS, "subordinate_ranges_present", values)


def _current_username() -> str:
    """Resolve the account from the kernel UID, never from USER/LOGNAME."""
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError):
        return ""


def _check_rootless_helpers(which: Callable[[str], str | None]) -> HostPreflightCheck:
    required = ("newuidmap", "newgidmap")
    missing = [name for name in required if not which(name)]
    if missing:
        return HostPreflightCheck("rootless_helpers", HostPreflightStatus.FAIL, "uid_gid_helpers_missing", {"missing": missing})
    network = [name for name in ("slirp4netns", "pasta") if which(name)]
    if not network:
        return HostPreflightCheck("rootless_helpers", HostPreflightStatus.FAIL, "rootless_network_helper_missing", {"missing": ["slirp4netns", "pasta"]})
    return HostPreflightCheck("rootless_helpers", HostPreflightStatus.PASS, "rootless_helpers_present", {"network_helpers": network})


def _check_cgroups(read: Callable[[str], str]) -> HostPreflightCheck:
    controllers = set(read("/sys/fs/cgroup/cgroup.controllers").split())
    required = {"cpu", "memory", "pids"}
    missing = sorted(required - controllers)
    if missing:
        return HostPreflightCheck("cgroups", HostPreflightStatus.FAIL, "resource_controllers_missing", {"missing": missing})
    return HostPreflightCheck("cgroups", HostPreflightStatus.PASS, "resource_controllers_available", {"controllers": sorted(controllers)})


def _check_seccomp(read: Callable[[str], str]) -> HostPreflightCheck:
    status = ""
    for line in read("/proc/self/status").splitlines():
        if line.startswith("Seccomp:"):
            status = line.split(":", 1)[1].strip()
            break
    actions = read("/proc/sys/kernel/seccomp/actions_avail").strip()
    if status in {"1", "2"}:
        return HostPreflightCheck("seccomp", HostPreflightStatus.PASS, "current_process_seccomp_confined", {"mode": status})
    if status == "0":
        if actions:
            return HostPreflightCheck(
                "seccomp",
                HostPreflightStatus.REQUIRED_ACCEPTANCE,
                "runtime_seccomp_profile_acceptance_required",
                {"mode": status, "actions": actions.split()},
            )
        return HostPreflightCheck("seccomp", HostPreflightStatus.INCONCLUSIVE, "seccomp_capability_unverified", {"mode": status})
    return HostPreflightCheck("seccomp", HostPreflightStatus.INCONCLUSIVE, "seccomp_status_unreadable")


def _check_filesystem(runtime_path: str | None, read: Callable[[str], str]) -> HostPreflightCheck:
    mounts = read("/proc/mounts")
    root_options = ""
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[1] == "/":
            root_options = fields[3]
            break
    if not runtime_path:
        return HostPreflightCheck("filesystem_restrictions", HostPreflightStatus.FAIL, "runtime_required_for_readonly_root_and_mount_policy", {"root_options": root_options})
    return HostPreflightCheck("filesystem_restrictions", HostPreflightStatus.REQUIRED_ACCEPTANCE, "runtime_mount_policy_requires_acceptance_test", {"root_options": root_options})


def _check_network_isolation(runtime_path: str | None, runtime_info: dict[str, Any] | None) -> HostPreflightCheck:
    if not runtime_path or runtime_info is None:
        return HostPreflightCheck("network_isolation", HostPreflightStatus.FAIL, "runtime_required_for_network_policy")
    return HostPreflightCheck("network_isolation", HostPreflightStatus.REQUIRED_ACCEPTANCE, "runtime_network_policy_requires_acceptance_test")


def _check_process_cleanup() -> HostPreflightCheck:
    if os.name == "posix" and hasattr(os, "killpg"):
        return HostPreflightCheck("process_cleanup", HostPreflightStatus.REQUIRED_ACCEPTANCE, "process_group_api_available_acceptance_required")
    return HostPreflightCheck("process_cleanup", HostPreflightStatus.FAIL, "process_group_termination_unavailable")


def _check_disk(root_path: str) -> HostPreflightCheck:
    try:
        available = int(os.statvfs(root_path).f_bavail * os.statvfs(root_path).f_frsize)
    except OSError:
        return HostPreflightCheck("disk_space", HostPreflightStatus.INCONCLUSIVE, "disk_stat_unavailable")
    status = HostPreflightStatus.PASS if available >= _MIN_DISK_BYTES else HostPreflightStatus.FAIL
    return HostPreflightCheck("disk_space", status, "disk_space_sufficient" if status is HostPreflightStatus.PASS else "disk_space_low", {"available_bytes": available})


def _check_memory(read: Callable[[str], str]) -> HostPreflightCheck:
    available_kib = None
    for line in read("/proc/meminfo").splitlines():
        if line.startswith("MemAvailable:"):
            try:
                available_kib = int(line.split()[1])
            except (IndexError, ValueError):
                pass
            break
    if available_kib is None:
        return HostPreflightCheck("memory", HostPreflightStatus.INCONCLUSIVE, "memory_stat_unavailable")
    available = available_kib * 1024
    status = HostPreflightStatus.PASS if available >= _MIN_MEMORY_BYTES else HostPreflightStatus.FAIL
    return HostPreflightCheck("memory", status, "memory_sufficient" if status is HostPreflightStatus.PASS else "memory_low", {"available_bytes": available})


def _probe(run: Callable[..., Any], argv: list[str]) -> tuple[Any | None, str | None]:
    try:
        result = run(argv, check=False, capture_output=True, text=True, timeout=_RUNTIME_PROBE_TIMEOUT, shell=False)
    except (OSError, subprocess.TimeoutExpired, TypeError):
        return None, "probe_failed"
    stdout = getattr(result, "stdout", "")
    stderr = getattr(result, "stderr", "")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return None, "probe_output_invalid"
    if len(stdout.encode("utf-8")) > _MAX_PROBE_OUTPUT or len(stderr.encode("utf-8")) > _MAX_PROBE_OUTPUT:
        return None, "probe_output_too_large"
    return result, None


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


__all__ = ["HostPreflightCheck", "HostPreflightReport", "HostPreflightStatus", "run_host_preflight"]
