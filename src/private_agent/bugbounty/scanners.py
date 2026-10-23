"""Fail-closed, non-executing adapters for ProjectDiscovery reconnaissance tools.

These public adapters validate scope and trusted template metadata, then remain
quarantined. They deliberately contain no host subprocess path. A future adapter
may parse results supplied by a verified external executor, but this module cannot
activate a scanner, regardless of caller flags, proofs, or injected objects.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

from private_agent.bugbounty.models import ScopeDecision, ScopeDefinition, public_reference, safe_json
from private_agent.bugbounty.trusted_tools import (
    TrustedToolInstallSpec,
)


class ScanStatus(str, Enum):
    COMPLETED = "COMPLETED"
    EMPTY = "EMPTY"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    UNAVAILABLE = "UNAVAILABLE"
    OUTPUT_LIMIT_EXCEEDED = "OUTPUT_LIMIT_EXCEEDED"
    SCOPE_DENIED = "SCOPE_DENIED"
    INVALID_OUTPUT = "INVALID_OUTPUT"


@dataclass(frozen=True)
class ScannerLimits:
    timeout_seconds: float = 60.0
    rate_limit_per_second: int = 10
    concurrency: int = 1
    max_output_bytes: int = 1_000_000
    max_records: int = 500

    def __post_init__(self) -> None:
        if not isinstance(self.timeout_seconds, (int, float)) or isinstance(self.timeout_seconds, bool) or not 0.1 <= float(self.timeout_seconds) <= 900:
            raise ValueError("scanner_timeout_invalid")
        if type(self.rate_limit_per_second) is not int or not 1 <= self.rate_limit_per_second <= 1000:
            raise ValueError("scanner_rate_limit_invalid")
        if type(self.concurrency) is not int or not 1 <= self.concurrency <= 32:
            raise ValueError("scanner_concurrency_invalid")
        if type(self.max_output_bytes) is not int or not 1024 <= self.max_output_bytes <= 16 * 1024 * 1024:
            raise ValueError("scanner_output_limit_invalid")
        if type(self.max_records) is not int or not 1 <= self.max_records <= 10_000:
            raise ValueError("scanner_record_limit_invalid")


@dataclass(frozen=True)
class ProcessResult:
    status: ScanStatus
    stdout: bytes = b""
    stderr: bytes = b""
    exit_code: int | None = None
    duration_ms: float = 0.0
    timed_out: bool = False
    cancelled: bool = False
    output_bytes: int = 0


@dataclass(frozen=True)
class ToolAvailability:
    tool: str
    available: bool
    executable: str = ""
    version: str = ""
    reason: str = ""
    supported_flags: tuple[str, ...] = ()
    provenance_verified: bool = False
    verification_method: str = ""
    artifact_sha256: str = ""


@dataclass(frozen=True)
class TargetBinding:
    target: str
    host: str
    port: int
    scope_reference: str


@dataclass
class ActiveScanResult:
    tool: str
    status: ScanStatus
    target: str
    scope_reference: str
    tool_version: str = ""
    observations: list[dict[str, Any]] = field(default_factory=list)
    candidate_findings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    execution: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status.value,
            "target": _redacted_url(self.target),
            "scope_reference": public_reference(self.scope_reference),
            "tool_version": self.tool_version,
            "observations": safe_json(self.observations)[:500],
            "candidate_findings": safe_json(self.candidate_findings)[:100],
            "errors": sorted(set(str(item)[:256] for item in self.errors))[:50],
            "limitations": sorted(set(str(item)[:512] for item in self.limitations))[:50],
            "execution": safe_json(self.execution),
            "created_at": self.created_at,
        }


class SafeSubprocessRunner:
    """Quarantined compatibility facade; it never starts a host process.

    A future OCI-backed executor may replace this seam, but no caller-provided
    argv, flag, proof, or adapter may turn this class into host execution.
    """

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
        cancel_event: Any | None = None,
    ) -> ProcessResult:
        del argv, timeout_seconds, max_output_bytes, cancel_event
        return ProcessResult(status=ScanStatus.UNAVAILABLE)


class ProjectDiscoveryProbe:
    """Report quarantined availability without probing or executing a binary."""

    _VERSION_PATTERNS = {
        "httpx": re.compile(r"(?:current\s+)?httpx(?:\s+version)?\s+v?(\d+\.\d+\.\d+)", re.I),
        "nuclei": re.compile(r"(?:nuclei(?:\s+engine)?(?:\s+version)?|nuclei\s+version)\s*:??\s*v?(\d+\.\d+\.\d+)", re.I),
    }

    def __init__(
        self,
        *,
        trusted_installation: TrustedToolInstallSpec | None = None,
        require_trusted_provenance: bool = False,
    ) -> None:
        self.trusted_installation = trusted_installation
        self.require_trusted_provenance = require_trusted_provenance

    def probe(self, tool: str, binary_path: str, required_flags: Sequence[str]) -> ToolAvailability:
        del binary_path, required_flags
        if tool not in self._VERSION_PATTERNS:
            return ToolAvailability(tool, False, reason="tool_identity_unsupported")
        if self.require_trusted_provenance and self.trusted_installation is None:
            return ToolAvailability(tool, False, reason="trusted_provenance_required")
        return ToolAvailability(tool, False, reason="external_sandbox_executor_unavailable")


@dataclass(frozen=True)
class TrustedTemplateSet:
    root: Path
    expected_manifest_sha256: str

    _PROHIBITED_TAGS = frozenset({"dos", "ddos", "fuzz", "fuzzing", "bruteforce", "brute-force", "headless", "destructive", "exploit"})

    def validate(self) -> tuple[bool, str, tuple[str, ...]]:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.expected_manifest_sha256 or ""):
            return False, "template_manifest_digest_invalid", ()
        try:
            root = self.root.expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return False, "template_directory_unavailable", ()
        if not root.is_dir() or root.is_symlink():
            return False, "template_directory_invalid", ()
        manifest_path = root / "manifest.json"
        try:
            if not manifest_path.is_file() or manifest_path.is_symlink():
                return False, "template_manifest_missing", ()
            manifest_bytes = manifest_path.read_bytes()
        except OSError:
            return False, "template_manifest_unreadable", ()
        if hashlib.sha256(manifest_bytes).hexdigest().lower() != self.expected_manifest_sha256.lower():
            return False, "template_manifest_digest_mismatch", ()
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False, "template_manifest_invalid_json", ()
        entries = manifest.get("templates") if isinstance(manifest, dict) else None
        if not isinstance(entries, list) or not entries or len(entries) > 500:
            return False, "template_manifest_entries_invalid", ()
        ids: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                return False, "template_manifest_entry_invalid", ()
            relative = entry.get("path")
            template_id = entry.get("id") or (Path(relative).stem if isinstance(relative, str) else "")
            tags = entry.get("tags", [])
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                return False, "template_path_invalid", ()
            if not isinstance(template_id, str) or not template_id or len(template_id) > 128:
                return False, "template_id_invalid", ()
            if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                return False, "template_tags_invalid", ()
            if self._PROHIBITED_TAGS.intersection(tag.lower() for tag in tags):
                return False, "prohibited_template_category", ()
            candidate = (root / relative).resolve()
            if root not in candidate.parents or candidate.is_symlink() or not candidate.is_file():
                return False, "template_path_untrusted", ()
            expected_template_sha256 = entry.get("sha256")
            if not isinstance(expected_template_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_template_sha256):
                return False, "template_digest_missing_or_invalid", ()
            try:
                actual_template_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError:
                return False, "template_unreadable", ()
            if actual_template_sha256.lower() != expected_template_sha256.lower():
                return False, "template_digest_mismatch", ()
            ids.append(template_id)
        return True, "", tuple(sorted(set(ids)))


class _AdapterBase:
    tool_name = ""
    required_flags: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        binary_path: str,
        limits: ScannerLimits,
        trusted_installation: TrustedToolInstallSpec | None = None,
        require_trusted_provenance: bool = False,
    ) -> None:
        self.binary_path = binary_path
        self.limits = limits
        self.probe = ProjectDiscoveryProbe(
            trusted_installation=trusted_installation,
            require_trusted_provenance=require_trusted_provenance,
        )

    def availability(self) -> ToolAvailability:
        return self.probe.probe(self.tool_name, self.binary_path, self.required_flags)

    def _base(self, status: ScanStatus, target: str, scope: ScopeDefinition, *, version: str = "", errors: Sequence[str] = (), limitations: Sequence[str] = (), process: ProcessResult | None = None) -> ActiveScanResult:
        execution: dict[str, Any] = {
            "exit_code": process.exit_code if process else None,
            "duration_ms": round(process.duration_ms, 3) if process else 0.0,
            "timed_out": bool(process.timed_out) if process else False,
            "cancelled": bool(process.cancelled) if process else False,
            "output_bytes": int(process.output_bytes) if process else 0,
            "stderr_sha256": hashlib.sha256(process.stderr).hexdigest() if process else "",
        }
        return ActiveScanResult(self.tool_name, status, target, scope.authorization_reference, version, errors=list(errors), limitations=list(limitations), execution=execution)

    @staticmethod
    def _target_binding(scope: ScopeDefinition, target: str, identity: str, resource_id: str) -> tuple[TargetBinding | None, str]:
        if not isinstance(target, str) or not target or len(target) > 2048 or any(ord(char) < 32 or char.isspace() for char in target):
            return None, "target_invalid"
        try:
            parsed = urlsplit(target)
            host = parsed.hostname or ""
            port = parsed.port
        except ValueError:
            return None, "target_invalid"
        if parsed.scheme.lower() not in {"http", "https"}:
            return None, "scheme_not_permitted"
        if parsed.username is not None or parsed.password is not None:
            return None, "userinfo_not_permitted"
        if parsed.fragment:
            return None, "fragment_not_permitted"
        if parsed.query:
            return None, "query_not_permitted"
        if not host:
            return None, "host_required"
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return None, "dns_pinning_unavailable"
        if address.version != 4:
            return None, "ipv6_transport_not_verified"
        effective_port = port or (443 if parsed.scheme.lower() == "https" else 80)
        decision, reason = scope.authorize(
            target,
            method="GET",
            mode="active",
            identity=identity,
            resource_id=resource_id,
        )
        if decision is not ScopeDecision.ALLOWED:
            return None, reason
        normalized_host = str(address)
        netloc = normalized_host if port is None else f"{normalized_host}:{effective_port}"
        canonical = urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))
        return TargetBinding(canonical, normalized_host, effective_port, scope.authorization_reference), ""

def _redacted_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = host
        if parsed.port is not None:
            netloc = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path or "/", "", ""))[:2048]
    except (TypeError, ValueError):
        return "[INVALID_URL]"


class HttpxAdapter(_AdapterBase):
    tool_name = "httpx"
    required_flags = ("-u", "-json", "-silent", "-follow-redirects", "-rate-limit", "-threads", "-timeout", "-retries")

    def scan(self, scope: ScopeDefinition, target: str, *, identity: str, resource_id: str, cancel_event: Any | None = None) -> ActiveScanResult:
        del cancel_event
        binding, reason = self._target_binding(scope, target, identity, resource_id)
        if binding is None:
            status = ScanStatus.UNAVAILABLE if reason == "dns_pinning_unavailable" else ScanStatus.SCOPE_DENIED
            return self._base(status, target, scope, errors=[reason], limitations=["active_target_boundary_not_satisfied"])
        return self._base(
            ScanStatus.UNAVAILABLE,
            binding.target,
            scope,
            errors=["external_sandbox_executor_unavailable"],
            limitations=["host_subprocess_execution_removed", "active_scan_requires_verified_oci_executor"],
        )


class NucleiAdapter(_AdapterBase):
    tool_name = "nuclei"
    required_flags = ("-u", "-jsonl", "-silent", "-t", "-exclude-tags", "-rate-limit", "-c", "-timeout")

    _PROHIBITED_TAGS = ("dos", "ddos", "fuzz", "fuzzing", "bruteforce", "headless", "destructive", "exploit")

    def __init__(
        self,
        *,
        binary_path: str,
        templates_root: Path,
        expected_manifest_sha256: str,
        limits: ScannerLimits,
        trusted_installation: TrustedToolInstallSpec | None = None,
        require_trusted_provenance: bool = False,
    ) -> None:
        super().__init__(
            binary_path=binary_path,
            limits=limits,
            trusted_installation=trusted_installation,
            require_trusted_provenance=require_trusted_provenance,
        )
        self.template_set = TrustedTemplateSet(Path(templates_root), expected_manifest_sha256)

    def scan(self, scope: ScopeDefinition, target: str, *, identity: str, resource_id: str, cancel_event: Any | None = None) -> ActiveScanResult:
        del cancel_event
        binding, reason = self._target_binding(scope, target, identity, resource_id)
        if binding is None:
            status = ScanStatus.UNAVAILABLE if reason == "dns_pinning_unavailable" else ScanStatus.SCOPE_DENIED
            return self._base(status, target, scope, errors=[reason], limitations=["active_target_boundary_not_satisfied"])
        templates_ok, template_reason, template_ids = self.template_set.validate()
        if not templates_ok:
            return self._base(ScanStatus.UNAVAILABLE, binding.target, scope, errors=[template_reason], limitations=["trusted_template_set_not_verified"])
        del template_ids
        return self._base(
            ScanStatus.UNAVAILABLE,
            binding.target,
            scope,
            errors=["external_sandbox_executor_unavailable"],
            limitations=["host_subprocess_execution_removed", "active_scan_requires_verified_oci_executor"],
        )


__all__ = [
    "ActiveScanResult",
    "HttpxAdapter",
    "NucleiAdapter",
    "ProcessResult",
    "ScanStatus",
    "ScannerLimits",
    "TargetBinding",
    "ToolAvailability",
    "TrustedTemplateSet",
]
