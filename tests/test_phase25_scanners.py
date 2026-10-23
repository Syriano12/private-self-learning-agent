from __future__ import annotations

import hashlib
import json
import os
import stat
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from private_agent.bugbounty.models import ScopeDefinition
from private_agent.core.execution import ExecutionContext, GenericExecutor
from private_agent.core.planner import PlanStep
from private_agent.security import SecurityController
from private_agent.storage import Store
from private_agent.tools.research import ToolRegistry
from private_agent.tools.scanners import HttpxReconTool
from private_agent.bugbounty.scanners import (
    HttpxAdapter,
    NucleiAdapter,
    SafeSubprocessRunner,
    ScanStatus,
    ScannerLimits,
)


def scope_payload(*, host: str = "127.0.0.1", port: int = 8765) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "program_id": "phase25-local",
        "authorization_reference": "phase25-written-authorization",
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "assets": [{
            "asset_id": "local-api",
            "host": host,
            "path_prefix": "/",
            "methods": ["GET"],
            "ports": [port],
            "active_allowed": True,
        }],
        "permitted_identities": ["alice"],
        "permitted_resources": ["resource-1"],
        "permitted_methods": ["GET"],
        "prohibited_methods": ["POST", "PUT", "PATCH", "DELETE"],
        "active_testing": True,
    }


def write_executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def httpx_fake(path: Path, output: str = "") -> Path:
    return write_executable(path, f"""
        import sys
        if '-version' in sys.argv:
            print('httpx version v1.7.0')
        elif '-h' in sys.argv:
            print('-u -json -silent -follow-redirects -rate-limit -threads -timeout -retries')
        else:
            print({output!r})
    """)


def nuclei_fake(path: Path, output: str = "") -> Path:
    return write_executable(path, f"""
        import sys
        if '-version' in sys.argv:
            print('nuclei version v3.3.7')
        elif '-h' in sys.argv:
            print('-u -jsonl -silent -t -exclude-tags -rate-limit -c -timeout')
        else:
            print({output!r})
    """)


def local_limits(**overrides) -> ScannerLimits:
    values = {
        "timeout_seconds": 2.0,
        "rate_limit_per_second": 5,
        "concurrency": 1,
        "max_output_bytes": 32_000,
        "max_records": 10,
    }
    values.update(overrides)
    return ScannerLimits(**values)


def trusted_templates(tmp_path: Path, *, tags: list[str] | None = None) -> tuple[Path, str]:
    root = tmp_path / "templates"
    root.mkdir()
    template = root / "safe.yaml"
    template.write_text("id: phase25-safe\ninfo:\n  name: safe\n", encoding="utf-8")
    manifest = {"templates": [{"id": "phase25-safe", "path": "safe.yaml", "sha256": hashlib.sha256(template.read_bytes()).hexdigest(), "tags": tags or ["safe"]}]}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_httpx_adapter_is_quarantined_without_external_executor(tmp_path: Path):
    output = json.dumps({
        "url": "http://127.0.0.1:8765/health",
        "status-code": 200,
        "title": "Local health",
        "content-type": "application/json",
        "webserver": "controlled-fixture",
        "a": ["127.0.0.1"],
    })
    binary = httpx_fake(tmp_path / "httpx", output)
    scope = ScopeDefinition.from_dict(scope_payload())

    result = HttpxAdapter(binary_path=str(binary), limits=local_limits()).scan(
        scope, "http://127.0.0.1:8765/health", identity="alice", resource_id="resource-1"
    )

    assert result.status is ScanStatus.UNAVAILABLE
    assert "external_sandbox_executor_unavailable" in result.errors
    assert result.observations == []
    assert result.candidate_findings == []


def test_scanner_rejects_query_parameters_before_subprocess(tmp_path: Path):
    output = json.dumps({
        "url": "http://127.0.0.1:8765/health?token=must-not-leak",
        "status-code": 200,
        "a": ["127.0.0.1"],
    })
    binary = httpx_fake(tmp_path / "httpx-query", output)
    scope = ScopeDefinition.from_dict(scope_payload())
    result = HttpxAdapter(binary_path=str(binary), limits=local_limits()).scan(
        scope, "http://127.0.0.1:8765/health?token=must-not-leak", identity="alice", resource_id="resource-1"
    )
    assert result.status is ScanStatus.SCOPE_DENIED
    assert "query_not_permitted" in result.errors


def test_proof_without_sandbox_executor_cannot_enable_host_subprocess(tmp_path: Path, monkeypatch):
    marker = tmp_path / "proof-bypass"
    binary = write_executable(tmp_path / "httpx-proof", f"""
        import pathlib
        pathlib.Path({str(marker)!r}).write_text('called')
    """)
    tool = HttpxReconTool(binary_path=str(binary), limits=local_limits())
    from private_agent.skills.sandbox_runtime import RuntimeActivationProof

    monkeypatch.setattr(RuntimeActivationProof, "ready", property(lambda _self: True))
    tool.runtime_proof = object.__new__(RuntimeActivationProof)
    result = tool.execute(
        {
            "scope": scope_payload(),
            "target": "http://127.0.0.1:8765/health",
            "identity": "alice",
            "resource_id": "resource-1",
        },
        None,
    )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "scanner_sandbox_executor_unavailable"
    assert not marker.exists()


def test_nuclei_adapter_is_quarantined_after_template_validation(tmp_path: Path):
    templates, digest = trusted_templates(tmp_path)
    output = json.dumps({
        "template-id": "phase25-safe",
        "info": {"name": "controlled observation", "severity": "medium", "tags": ["safe"]},
        "matched-at": "http://127.0.0.1:8765/health",
        "type": "http",
    })
    binary = nuclei_fake(tmp_path / "nuclei", output)
    scope = ScopeDefinition.from_dict(scope_payload())

    result = NucleiAdapter(
        binary_path=str(binary),
        templates_root=templates,
        expected_manifest_sha256=digest,
        limits=local_limits(),
    ).scan(scope, "http://127.0.0.1:8765/health", identity="alice", resource_id="resource-1")

    assert result.status is ScanStatus.UNAVAILABLE
    assert "external_sandbox_executor_unavailable" in result.errors
    assert result.observations == []
    assert result.candidate_findings == []


def test_missing_binary_is_distinct_from_scan_failure(tmp_path: Path):
    scope = ScopeDefinition.from_dict(scope_payload())
    result = HttpxAdapter(binary_path=str(tmp_path / "missing-httpx"), limits=local_limits()).scan(
        scope, "http://127.0.0.1:8765/health", identity="alice", resource_id="resource-1"
    )
    assert result.status is ScanStatus.UNAVAILABLE
    assert "external_sandbox_executor_unavailable" in result.errors


def test_hostname_requires_pinned_resolution_in_active_subprocess_boundary(tmp_path: Path):
    binary = httpx_fake(tmp_path / "httpx", json.dumps({"url": "https://allowed.example/", "status-code": 200}))
    scope = ScopeDefinition.from_dict(scope_payload(host="allowed.example", port=443))
    result = HttpxAdapter(binary_path=str(binary), limits=local_limits()).scan(
        scope, "https://allowed.example/", identity="alice", resource_id="resource-1"
    )
    assert result.status is ScanStatus.UNAVAILABLE
    assert "dns_pinning_unavailable" in result.errors


def test_alternate_port_and_unauthorized_redirect_are_not_accepted(tmp_path: Path):
    binary = httpx_fake(tmp_path / "httpx", json.dumps({
        "url": "http://127.0.0.1:8765/health",
        "status-code": 302,
        "chain": ["http://127.0.0.1:8765/health", "http://other.example/"],
        "final-url": "http://other.example/",
    }))
    scope = ScopeDefinition.from_dict(scope_payload())
    adapter = HttpxAdapter(binary_path=str(binary), limits=local_limits())

    wrong_port = adapter.scan(scope, "http://127.0.0.1:9999/health", identity="alice", resource_id="resource-1")
    redirect = adapter.scan(scope, "http://127.0.0.1:8765/health", identity="alice", resource_id="resource-1")

    assert wrong_port.status is ScanStatus.SCOPE_DENIED
    assert "port_not_authorized" in wrong_port.errors
    assert redirect.status is ScanStatus.UNAVAILABLE
    assert "external_sandbox_executor_unavailable" in redirect.errors


def test_malformed_and_empty_jsonl_are_not_clean_results(tmp_path: Path):
    scope = ScopeDefinition.from_dict(scope_payload())
    malformed = httpx_fake(tmp_path / "httpx-malformed", "not-json")
    empty = httpx_fake(tmp_path / "httpx-empty", "")

    malformed_result = HttpxAdapter(binary_path=str(malformed), limits=local_limits()).scan(
        scope, "http://127.0.0.1:8765/health", identity="alice", resource_id="resource-1"
    )
    empty_result = HttpxAdapter(binary_path=str(empty), limits=local_limits()).scan(
        scope, "http://127.0.0.1:8765/health", identity="alice", resource_id="resource-1"
    )

    assert malformed_result.status is ScanStatus.UNAVAILABLE
    assert empty_result.status is ScanStatus.UNAVAILABLE
    assert "external_sandbox_executor_unavailable" in malformed_result.errors
    assert "external_sandbox_executor_unavailable" in empty_result.errors


def test_untrusted_or_prohibited_templates_fail_closed(tmp_path: Path):
    templates, digest = trusted_templates(tmp_path, tags=["dos"])
    binary = nuclei_fake(tmp_path / "nuclei", "")
    scope = ScopeDefinition.from_dict(scope_payload())
    result = NucleiAdapter(
        binary_path=str(binary),
        templates_root=templates,
        expected_manifest_sha256=digest,
        limits=local_limits(),
    ).scan(scope, "http://127.0.0.1:8765/health", identity="alice", resource_id="resource-1")
    assert result.status is ScanStatus.UNAVAILABLE
    assert "prohibited_template_category" in result.errors


def test_template_content_tampering_fails_closed(tmp_path: Path):
    templates, digest = trusted_templates(tmp_path)
    (templates / "safe.yaml").write_text("id: phase25-safe\ninfo:\n  name: tampered\n", encoding="utf-8")
    binary = nuclei_fake(tmp_path / "nuclei", "")
    scope = ScopeDefinition.from_dict(scope_payload())
    result = NucleiAdapter(
        binary_path=str(binary), templates_root=templates, expected_manifest_sha256=digest, limits=local_limits()
    ).scan(scope, "http://127.0.0.1:8765/health", identity="alice", resource_id="resource-1")
    assert result.status is ScanStatus.UNAVAILABLE
    assert "template_digest_mismatch" in result.errors


def test_safe_subprocess_runner_is_quarantined_without_external_executor(tmp_path: Path):
    slow = write_executable(tmp_path / "slow", """
        import time
        time.sleep(5)
    """)
    noisy = write_executable(tmp_path / "noisy", """
        print('x' * 100000)
    """)

    runner = SafeSubprocessRunner()
    timed = runner.run([str(slow)], timeout_seconds=0.05, max_output_bytes=4096)
    capped = runner.run([str(noisy)], timeout_seconds=2.0, max_output_bytes=1024)

    assert timed.status is ScanStatus.UNAVAILABLE
    assert capped.status is ScanStatus.UNAVAILABLE


def test_target_argument_is_never_interpreted_as_shell(tmp_path: Path):
    marker = tmp_path / "marker"
    output = json.dumps({"url": "http://127.0.0.1:8765/$(touch%20" + str(marker) + ")", "status-code": 200})
    binary = httpx_fake(tmp_path / "httpx", output)
    scope = ScopeDefinition.from_dict(scope_payload())
    result = HttpxAdapter(binary_path=str(binary), limits=local_limits()).scan(
        scope, "http://127.0.0.1:8765/$(touch%20" + str(marker) + ")", identity="alice", resource_id="resource-1"
    )
    assert not marker.exists()
    assert result.status is ScanStatus.UNAVAILABLE


def test_registered_httpx_tool_requires_owner_approval_and_remains_quarantined_without_sandbox(tmp_path: Path):
    marker = tmp_path / "executed"
    binary = write_executable(tmp_path / "httpx", f"""
        import json, pathlib, sys
        pathlib.Path({str(marker)!r}).write_text('called')
        if '-version' in sys.argv:
            print('httpx version v1.7.0')
        elif '-h' in sys.argv:
            print('-u -json -silent -follow-redirects -rate-limit -threads -timeout -retries')
        else:
            print(json.dumps({{"url": "http://127.0.0.1:8765/health", "status-code": 200}}))
    """)
    store = Store(str(tmp_path / "executor.sqlite3"))
    controller = SecurityController(store=store)
    registry = ToolRegistry()
    registry.register(HttpxReconTool(binary_path=str(binary), limits=local_limits()))
    executor = GenericExecutor(registry, controller)
    inputs = {
        "scope": scope_payload(),
        "target": "http://127.0.0.1:8765/health",
        "identity": "alice",
        "resource_id": "resource-1",
    }
    step = PlanStep("scan", "recon", "httpx_recon", input=inputs, reason="authorized local validation")
    context = ExecutionContext(task_id="phase25-executor", metadata={"goal": "authorized reconnaissance"})

    blocked = executor.execute_step(step, context)
    assert blocked.status == "blocked"
    assert blocked.error is not None and blocked.error.code == "permission_not_approved"
    assert not marker.exists()

    action = controller.action_for_tool("phase25-executor", "scan", registry.get("httpx_recon"), inputs, goal="authorized reconnaissance")
    pending = controller.authorize(action)
    assert pending.decision == "REQUIRE_APPROVAL"
    controller.approval_gate.approve(pending.approval_id, actor="owner")
    approved_context = ExecutionContext(
        task_id="phase25-executor",
        metadata={"goal": "authorized reconnaissance"},
        approval_ids={"scan": pending.approval_id},
    )
    completed = executor.execute_step(step, approved_context)
    assert completed.status == "failed"
    assert completed.error is not None and completed.error.code == "scanner_sandbox_unavailable"
    assert not marker.exists()

    marker.unlink(missing_ok=True)
    out_of_scope = dict(inputs, target="http://127.0.0.1:9999/health")
    out_of_scope_step = PlanStep("scan-out", "recon", "httpx_recon", input=out_of_scope, reason="out of scope")
    out_of_scope_context = ExecutionContext(
        task_id="phase25-executor-out",
        metadata={"goal": "authorized reconnaissance"},
        approval_ids={"scan-out": pending.approval_id},
    )
    denied = executor.execute_step(out_of_scope_step, out_of_scope_context)
    assert denied.status == "blocked"
    assert denied.error is not None
    assert not marker.exists()
    store.close()
