#!/usr/bin/env python3
"""Acceptance harness for an administrator-provisioned local Podman host.

This file is intentionally not run by the normal test suite. It requires an
explicit opt-in and a locally available, administrator-approved image pinned by
sha256. It never pulls images, contacts production targets, or accepts a shell
command from the environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Mapping

from private_agent.skills.runtime_preflight import HostPreflightStatus, run_host_preflight


_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_OUTPUT = 128 * 1024

# This policy is part of the acceptance identity. Changing any value changes
# the policy digest and invalidates a previously captured acceptance result.
_ACCEPTANCE_POLICY: dict[str, Any] = {
    "network": "none",
    "read_only": True,
    "cap_drop": "all",
    "no_new_privileges": True,
    "pids_limit": "32",
    "memory": "128m",
    "cpus": "0.5",
    "tmpfs": "/tmp:rw,noexec,nosuid,nodev,size=16m",
    "enforced_probe": (
        "set -eu; test ! -w /; test -w /tmp; test ! -e /run/secrets/host; "
        "test \"$(cat /sys/fs/cgroup/memory.max)\" = 134217728; "
        "test \"$(cat /sys/fs/cgroup/pids.max)\" = 32; "
        "test \"$(awk 'NR > 1 {print}' /proc/net/route | wc -l)\" = 0; "
        "printf acceptance-ok"
    ),
    "isolation_probe_first": "set -eu; test ! -e /tmp/vulnora-task-marker; touch /tmp/vulnora-task-marker",
    "isolation_probe_second": "set -eu; test ! -e /tmp/vulnora-task-marker; printf isolation-ok",
}


@dataclass(frozen=True)
class AcceptanceResult:
    name: str
    status: str
    detail: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def _policy_digest() -> str:
    payload = json.dumps(_ACCEPTANCE_POLICY, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def acceptance_binding_matches(
    binding: Mapping[str, Any],
    *,
    runtime_identity: str,
    image_digest: str,
    policy_digest: str,
) -> bool:
    """Return true only when all immutable acceptance inputs match exactly."""
    if not isinstance(binding, Mapping):
        return False
    if not _DIGEST_RE.fullmatch(image_digest) or not _DIGEST_RE.fullmatch(policy_digest):
        return False
    return (
        binding.get("runtime_identity") == runtime_identity
        and binding.get("image_digest") == image_digest
        and binding.get("policy_digest") == policy_digest
    )


def _run(argv: list[str], *, timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        env={"PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C"},
    )


def _result(name: str, status: str, detail: str, **evidence: Any) -> AcceptanceResult:
    return AcceptanceResult(name, status, detail, evidence)


def _bounded(value: str) -> str:
    return value[:_MAX_OUTPUT]


def _podman_run_prefix(image: str) -> list[str]:
    return [
        "podman",
        "run",
        "--rm",
        "--pull=never",
        "--network",
        str(_ACCEPTANCE_POLICY["network"]),
        "--read-only" if _ACCEPTANCE_POLICY["read_only"] else "--read-write",
        "--cap-drop",
        str(_ACCEPTANCE_POLICY["cap_drop"]),
        "--security-opt=no-new-privileges" if _ACCEPTANCE_POLICY["no_new_privileges"] else "--security-opt=no-new-privileges=false",
        "--pids-limit",
        str(_ACCEPTANCE_POLICY["pids_limit"]),
        "--memory",
        str(_ACCEPTANCE_POLICY["memory"]),
        "--cpus",
        str(_ACCEPTANCE_POLICY["cpus"]),
        "--tmpfs",
        str(_ACCEPTANCE_POLICY["tmpfs"]),
        image,
        "/bin/sh",
        "-c",
    ]


def run_acceptance(image: str) -> tuple[int, list[AcceptanceResult], dict[str, str] | None]:
    if os.getenv("VULNORA_ENABLE_RUNTIME_ACCEPTANCE") != "1":
        return 2, [_result("activation", "NOT_RUN", "explicit_opt_in_required")], None
    if not _IMAGE_RE.fullmatch(image):
        return 2, [_result("image", "FAIL", "image_must_be_local_and_digest_pinned")], None

    preflight = run_host_preflight()
    if not preflight.can_run_acceptance:
        return 2, [
            _result(
                "host_preflight",
                "NOT_RUN",
                "host_capabilities_not_ready",
                verdict=preflight.verdict.value,
                host_ready=preflight.host_ready,
                can_run_acceptance=preflight.can_run_acceptance,
                acceptance_required=preflight.acceptance_required,
                runtime_identity=preflight.runtime_identity,
            )
        ], None

    results: list[AcceptanceResult] = []
    image_digest = image.rsplit("@", 1)[1]
    runtime_identity = preflight.runtime_identity
    if runtime_identity is None:
        return 2, [_result("host_preflight", "FAIL", "runtime_identity_missing")], None
    binding = {
        "runtime_identity": runtime_identity,
        "image_digest": image_digest,
        "policy_digest": _policy_digest(),
    }

    inspected = _run(["podman", "image", "inspect", "--format", "json", image])
    try:
        records = json.loads(inspected.stdout) if inspected.returncode == 0 else []
        record = records[0] if isinstance(records, list) and records and isinstance(records[0], dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        record = {}
    repo_digests = record.get("RepoDigests", []) if isinstance(record, dict) else []
    declared_digest = record.get("Digest") if isinstance(record, dict) else None
    observed_digests = [str(value).rsplit("@", 1)[-1] for value in repo_digests if isinstance(value, str)]
    if isinstance(declared_digest, str):
        observed_digests.append(declared_digest)
    if inspected.returncode != 0 or image_digest not in observed_digests:
        return 2, [
            _result(
                "image_identity",
                "FAIL",
                "local_image_digest_mismatch",
                observed_digests=observed_digests,
                stdout=_bounded(inspected.stdout),
                stderr=_bounded(inspected.stderr),
            )
        ], binding
    results.append(_result("image_identity", "PASS", "local_image_digest_matches", digest=image_digest))

    command = _podman_run_prefix(image) + [_ACCEPTANCE_POLICY["enforced_probe"]]
    probe = _run(command, timeout=45.0)
    if probe.returncode != 0 or probe.stdout.strip() != "acceptance-ok":
        return 1, results + [
            _result(
                "enforced_policy_probe",
                "FAIL",
                "container_policy_probe_failed",
                stdout=_bounded(probe.stdout),
                stderr=_bounded(probe.stderr),
                returncode=probe.returncode,
            )
        ], binding
    results.append(_result("enforced_policy_probe", "PASS", "readonly_root_tmpfs_limits_and_network_none_verified"))

    isolation_prefix = _podman_run_prefix(image)
    first = _run(isolation_prefix + [_ACCEPTANCE_POLICY["isolation_probe_first"]], timeout=30.0)
    second = _run(isolation_prefix + [_ACCEPTANCE_POLICY["isolation_probe_second"]], timeout=30.0)
    if first.returncode != 0 or second.returncode != 0 or second.stdout.strip() != "isolation-ok":
        return 1, results + [
            _result(
                "task_filesystem_isolation",
                "FAIL",
                "independent_task_filesystem_probe_failed",
                first_stderr=_bounded(first.stderr),
                second_stderr=_bounded(second.stderr),
            )
        ], binding
    results.append(_result("task_filesystem_isolation", "PASS", "temporary_task_storage_not_shared"))
    return 0, results, binding


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled local Podman acceptance tests")
    parser.add_argument("--image", default=os.getenv("VULNORA_ACCEPTANCE_IMAGE", ""), help="administrator-approved local image@sha256:digest")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    code, results, binding = run_acceptance(args.image)
    payload = {
        "verdict": "PASS" if code == 0 else "BLOCKED" if code == 2 else "FAIL",
        "binding": binding,
        "results": [item.to_dict() for item in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
