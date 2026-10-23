from __future__ import annotations

import json
import re
from types import SimpleNamespace

from scripts import phase28_runtime_acceptance as acceptance
from private_agent.skills.runtime_preflight import (
    HostPreflightCheck,
    HostPreflightReport,
    HostPreflightStatus,
)


_IMAGE = "localhost/vulnora-acceptance@sha256:" + "a" * 64


def _ready_report() -> HostPreflightReport:
    host_names = (
        "os_kernel",
        "oci_runtime",
        "user_namespaces",
        "subordinate_ids",
        "rootless_helpers",
        "cgroups",
        "disk_space",
        "memory",
    )
    checks = [
        HostPreflightCheck(name, HostPreflightStatus.PASS, "ok", {"version": "5.2.1"})
        for name in host_names
    ]
    checks.extend(
        HostPreflightCheck(
            name,
            HostPreflightStatus.REQUIRED_ACCEPTANCE,
            "acceptance_required",
        )
        for name in ("seccomp", "filesystem_restrictions", "network_isolation", "process_cleanup")
    )
    return HostPreflightReport(tuple(checks))


def test_required_acceptance_is_runnable_but_not_pass(monkeypatch):
    report = _ready_report()

    assert report.verdict is HostPreflightStatus.REQUIRED_ACCEPTANCE
    assert report.host_ready is True
    assert report.can_run_acceptance is True
    assert report.acceptance_required is True


def test_inconclusive_acceptance_check_is_not_runnable():
    checks = list(_ready_report().checks)
    checks[8] = HostPreflightCheck("seccomp", HostPreflightStatus.INCONCLUSIVE, "unverified")
    report = HostPreflightReport(tuple(checks))

    assert report.verdict is HostPreflightStatus.INCONCLUSIVE
    assert report.host_ready is True
    assert report.can_run_acceptance is False


def test_acceptance_runs_after_host_gate_and_returns_bound_identity(monkeypatch):
    monkeypatch.setenv("VULNORA_ENABLE_RUNTIME_ACCEPTANCE", "1")
    monkeypatch.setattr(acceptance, "run_host_preflight", lambda: _ready_report())
    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, timeout: float = 20.0):
        calls.append(argv)
        if argv[1:4] == ["image", "inspect", "--format"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"RepoDigests": [_IMAGE], "Digest": _IMAGE.rsplit("@", 1)[1]}]),
                stderr="",
            )
        if "acceptance-ok" in argv[-1]:
            return SimpleNamespace(returncode=0, stdout="acceptance-ok", stderr="")
        return SimpleNamespace(returncode=0, stdout="isolation-ok", stderr="")

    monkeypatch.setattr(acceptance, "_run", fake_run)
    code, results, binding = acceptance.run_acceptance(_IMAGE)

    assert code == 0
    assert [item.status for item in results] == ["PASS", "PASS", "PASS"]
    assert binding == {
        "runtime_identity": "podman@5.2.1",
        "image_digest": "sha256:" + "a" * 64,
        "policy_digest": binding["policy_digest"],
    }
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", binding["policy_digest"])
    assert acceptance.acceptance_binding_matches(
        binding,
        runtime_identity="podman@5.2.1",
        image_digest="sha256:" + "a" * 64,
        policy_digest=binding["policy_digest"],
    )
    assert len(calls) == 4


def test_acceptance_binding_changes_are_rejected():
    binding = {
        "runtime_identity": "podman@5.2.1",
        "image_digest": "sha256:" + "a" * 64,
        "policy_digest": "sha256:" + "b" * 64,
    }

    assert not acceptance.acceptance_binding_matches(
        binding,
        runtime_identity="podman@5.2.2",
        image_digest=binding["image_digest"],
        policy_digest=binding["policy_digest"],
    )
    assert not acceptance.acceptance_binding_matches(
        binding,
        runtime_identity=binding["runtime_identity"],
        image_digest="sha256:" + "c" * 64,
        policy_digest=binding["policy_digest"],
    )
    assert not acceptance.acceptance_binding_matches(
        binding,
        runtime_identity=binding["runtime_identity"],
        image_digest=binding["image_digest"],
        policy_digest="sha256:" + "d" * 64,
    )


def test_opt_in_missing_does_not_run_acceptance(monkeypatch):
    monkeypatch.delenv("VULNORA_ENABLE_RUNTIME_ACCEPTANCE", raising=False)
    code, results, binding = acceptance.run_acceptance(_IMAGE)

    assert code == 2
    assert results[0].detail == "explicit_opt_in_required"
    assert binding is None
