from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from private_agent.core.experience import sanitize_value


SUPPORTED_RUNTIME = "podman"
SUPPORTED_SECURITY_CONTROLS = frozenset({"rootless", "seccomp"})
SUPPORTED_NETWORK_POLICIES = frozenset({"disabled"})
MAX_TIMEOUT_SECONDS = 300.0
MAX_CPU_LIMIT_MILLIS = 600_000
MIN_MEMORY_LIMIT_BYTES = 1 * 1024 * 1024
MAX_MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
MAX_PROCESS_LIMIT = 256
MAX_OUTPUT_LIMIT_BYTES = 16 * 1024 * 1024
MAX_ERROR_LENGTH = 4096
MAX_IDENTITY_LENGTH = 256
MAX_SECURITY_CONTROLS = 8
MAX_DIGEST_LENGTH = 71
MAX_RUNTIME_PROBE_STDOUT_BYTES = 64 * 1024
MAX_RUNTIME_PROBE_STDERR_BYTES = 8 * 1024
_DETECTOR_ATTESTATION = object()
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RUNTIME_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:[-+][0-9A-Za-z.-]+)?$")
_SECURITY_CONTROL_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ValidationOutcome(str, Enum):
    VALID = "VALID"
    SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"
    VALIDATION_UNKNOWN = "VALIDATION_UNKNOWN"
    INVALID = "INVALID"


class SandboxUnavailable(RuntimeError):
    """Raised when the trusted external sandbox cannot be established."""

    code = ValidationOutcome.SANDBOX_UNAVAILABLE.value

    def __init__(self, reason: str, *, details: tuple[str, ...] = ()) -> None:
        self.reason = str(reason)
        self.details = tuple(str(item) for item in details)
        super().__init__(f"{self.code}:{self.reason}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "reason": self.reason,
            "details": list(self.details),
        }


@dataclass(frozen=True, slots=True)
class SandboxSpec:
    """Immutable, bounded identity and resource contract for a future runner."""

    runtime_name: str
    runtime_version: str
    image_digest: str
    command_manifest_digest: str
    timeout_seconds: float
    cpu_limit_millis: int
    memory_limit_bytes: int
    process_limit: int
    output_limit_bytes: int
    network_policy: str = "disabled"
    required_security_controls: tuple[str, ...] = ("rootless", "seccomp")

    def __post_init__(self) -> None:
        if self.runtime_name != SUPPORTED_RUNTIME:
            raise ValueError("runtime_unsupported")
        if not isinstance(self.runtime_version, str) or not _RUNTIME_VERSION_RE.fullmatch(self.runtime_version):
            raise ValueError("runtime_version_invalid")
        _validate_digest(self.image_digest, "image_digest")
        _validate_digest(self.command_manifest_digest, "command_manifest_digest")
        _validate_positive_finite(self.timeout_seconds, MAX_TIMEOUT_SECONDS, "timeout_seconds")
        _validate_int_limit(self.cpu_limit_millis, 1, MAX_CPU_LIMIT_MILLIS, "cpu_limit_millis")
        _validate_int_limit(
            self.memory_limit_bytes,
            MIN_MEMORY_LIMIT_BYTES,
            MAX_MEMORY_LIMIT_BYTES,
            "memory_limit_bytes",
        )
        _validate_int_limit(self.process_limit, 1, MAX_PROCESS_LIMIT, "process_limit")
        _validate_int_limit(self.output_limit_bytes, 1, MAX_OUTPUT_LIMIT_BYTES, "output_limit_bytes")
        if self.network_policy not in SUPPORTED_NETWORK_POLICIES:
            raise ValueError("network_policy_unsupported")
        if not isinstance(self.required_security_controls, (tuple, list)):
            raise ValueError("security_controls_invalid")
        controls = tuple(self.required_security_controls)
        if not controls or len(controls) > MAX_SECURITY_CONTROLS:
            raise ValueError("security_controls_invalid")
        if len(set(controls)) != len(controls):
            raise ValueError("security_controls_duplicate")
        for control in controls:
            if not isinstance(control, str) or not _SECURITY_CONTROL_RE.fullmatch(control):
                raise ValueError("security_control_invalid")
        object.__setattr__(self, "required_security_controls", controls)

    @property
    def runtime_identity(self) -> str:
        return f"{self.runtime_name}@{self.runtime_version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_name": self.runtime_name,
            "runtime_version": self.runtime_version,
            "runtime_identity": self.runtime_identity,
            "image_digest": self.image_digest,
            "command_manifest_digest": self.command_manifest_digest,
            "timeout_seconds": self.timeout_seconds,
            "cpu_limit_millis": self.cpu_limit_millis,
            "memory_limit_bytes": self.memory_limit_bytes,
            "process_limit": self.process_limit,
            "output_limit_bytes": self.output_limit_bytes,
            "network_policy": self.network_policy,
            "required_security_controls": list(self.required_security_controls),
        }


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Immutable result envelope shared with the existing quarantined lifecycle."""

    status: str
    output: Any = None
    error: str = ""
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    applied_restrictions: tuple[str, ...] = field(default_factory=tuple)
    duration_ms: float = 0.0
    exit_code: int | None = None
    signal: int | None = None
    timed_out: bool = False
    runtime_identity: str = ""
    image_digest: str = ""
    command_manifest_digest: str = ""
    stdout_digest: str = ""
    stderr_digest: str = ""
    output_digest: str = ""
    validation_outcome: ValidationOutcome = ValidationOutcome.VALIDATION_UNKNOWN

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or not self.status or len(self.status) > 64:
            raise ValueError("sandbox_result_status_invalid")
        if not isinstance(self.error, str) or len(self.error) > MAX_ERROR_LENGTH:
            raise ValueError("sandbox_result_error_invalid")
        _validate_nonnegative_finite(self.duration_ms, "duration_ms")
        if self.exit_code is not None:
            _validate_int_limit(self.exit_code, -255, 255, "exit_code")
        if self.signal is not None:
            _validate_int_limit(self.signal, 1, 255, "signal")
        if not isinstance(self.timed_out, bool):
            raise ValueError("sandbox_result_timeout_flag_invalid")
        for name, value in (
            ("runtime_identity", self.runtime_identity),
            ("image_digest", self.image_digest),
            ("command_manifest_digest", self.command_manifest_digest),
        ):
            if not isinstance(value, str) or len(value) > MAX_IDENTITY_LENGTH:
                raise ValueError(f"sandbox_result_{name}_invalid")
        if self.runtime_identity:
            runtime_name, separator, runtime_version = self.runtime_identity.partition("@")
            if (
                not separator
                or runtime_name != SUPPORTED_RUNTIME
                or not _RUNTIME_VERSION_RE.fullmatch(runtime_version)
            ):
                raise ValueError("sandbox_result_runtime_identity_invalid")
        if self.image_digest:
            _validate_digest(self.image_digest, "image_digest")
        if self.command_manifest_digest:
            _validate_digest(self.command_manifest_digest, "command_manifest_digest")
        for name, value in (
            ("stdout_digest", self.stdout_digest),
            ("stderr_digest", self.stderr_digest),
            ("output_digest", self.output_digest),
        ):
            if not isinstance(value, str):
                raise ValueError(f"sandbox_result_{name}_invalid")
            if value and (len(value) > MAX_DIGEST_LENGTH or not _DIGEST_RE.fullmatch(value)):
                raise ValueError(f"sandbox_result_{name}_invalid")
        outcome = self.validation_outcome
        if isinstance(outcome, str):
            try:
                outcome = ValidationOutcome(outcome)
            except ValueError as exc:
                raise ValueError("sandbox_result_validation_outcome_invalid") from exc
            object.__setattr__(self, "validation_outcome", outcome)
        elif not isinstance(outcome, ValidationOutcome):
            raise ValueError("sandbox_result_validation_outcome_invalid")
        if outcome is ValidationOutcome.VALID:
            if not (self.runtime_identity and self.image_digest and self.command_manifest_digest):
                raise ValueError("sandbox_result_verified_identity_missing")
            raise ValueError("sandbox_result_verified_result_requires_trusted_runner")

    @classmethod
    def unavailable(cls, reason: str, *, capabilities: Mapping[str, Any] | None = None) -> "SandboxResult":
        return cls(
            status="QUARANTINED",
            error=str(reason),
            capabilities=capabilities or {},
            validation_outcome=ValidationOutcome.SANDBOX_UNAVAILABLE,
        )

    def to_dict(self) -> dict[str, Any]:
        # Keep serialization explicit: future result payloads must not invoke
        # arbitrary object-copy hooks while preserving the legacy output field.
        return sanitize_value({
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "capabilities": dict(self.capabilities),
            "applied_restrictions": list(self.applied_restrictions),
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
            "signal": self.signal,
            "timed_out": self.timed_out,
            "runtime_identity": self.runtime_identity,
            "image_digest": self.image_digest,
            "command_manifest_digest": self.command_manifest_digest,
            "stdout_digest": self.stdout_digest,
            "stderr_digest": self.stderr_digest,
            "output_digest": self.output_digest,
            "validation_outcome": self.validation_outcome.value,
        })


@dataclass(frozen=True, slots=True)
class RuntimeCapabilityReport:
    runtime_name: str
    executable: str | None
    version: str | None
    detected: bool
    version_verified: bool
    rootless_verified: bool
    security_controls_verified: bool
    image_identity_verified: bool
    command_manifest_verified: bool
    validation_outcome: ValidationOutcome
    command_manifest_pinned: bool = False
    command_manifest_provenance_verified: bool = False
    network_policy_verified: bool = False
    reasons: tuple[str, ...] = ()
    verified_controls: tuple[str, ...] = ()
    _attestation: object = field(default=None, init=False, repr=False, compare=False)

    @property
    def available(self) -> bool:
        return self.validation_outcome is ValidationOutcome.VALID and self._attestation is _DETECTOR_ATTESTATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_name": self.runtime_name,
            "executable": self.executable,
            "version": self.version,
            "detected": self.detected,
            "version_verified": self.version_verified,
            "rootless_verified": self.rootless_verified,
            "security_controls_verified": self.security_controls_verified,
            "image_identity_verified": self.image_identity_verified,
            "command_manifest_verified": self.command_manifest_verified,
            "command_manifest_pinned": self.command_manifest_pinned,
            "command_manifest_provenance_verified": self.command_manifest_provenance_verified,
            "network_policy_verified": self.network_policy_verified,
            "validation_outcome": self.validation_outcome.value,
            "reasons": list(self.reasons),
            "verified_controls": list(self.verified_controls),
        }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"runtime_info_non_standard_constant:{value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("runtime_info_duplicate_key")
        result[key] = value
    return result


def detect_podman(
    spec: SandboxSpec | None = None,
    *,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., Any] = subprocess.run,
) -> RuntimeCapabilityReport:
    """Probe Podman metadata only; never starts a container or target command."""

    executable = which(SUPPORTED_RUNTIME)
    if not executable:
        return _unavailable_report(None, None, False, "runtime_missing")

    version_result, probe_error = _probe(run, [executable, "version", "--format", "{{.Client.Version}}"])
    if version_result is None or version_result.returncode != 0:
        return _unavailable_report(executable, None, True, probe_error or "runtime_version_probe_failed")
    version = str(getattr(version_result, "stdout", "")).strip()
    if not _RUNTIME_VERSION_RE.fullmatch(version):
        return _unavailable_report(executable, version or None, True, "runtime_version_malformed")

    info_result, probe_error = _probe(run, [executable, "info", "--format", "json"])
    if info_result is None or info_result.returncode != 0:
        return _unavailable_report(executable, version, True, probe_error or "runtime_info_probe_failed")
    try:
        info = json.loads(
            str(getattr(info_result, "stdout", "")),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return _unavailable_report(executable, version, True, "runtime_info_malformed")
    if not isinstance(info, dict):
        return _unavailable_report(executable, version, True, "runtime_info_malformed")

    host = info.get("host") if isinstance(info.get("host"), dict) else {}
    security = host.get("security") if isinstance(host.get("security"), dict) else {}
    rootless_verified = security.get("rootless") is True
    seccomp_profile = security.get("seccompProfile")
    seccomp_verified = isinstance(seccomp_profile, str) and bool(seccomp_profile.strip())
    verified_controls = tuple(
        control for control, verified in (("rootless", rootless_verified), ("seccomp", seccomp_verified)) if verified
    )
    security_controls_verified = rootless_verified and seccomp_verified
    reasons: list[str] = []
    if not rootless_verified:
        reasons.append("rootless_unverified")
    if not seccomp_verified:
        reasons.append("seccomp_unverified")

    image_identity_verified = False
    command_manifest_verified = False
    command_manifest_pinned = False
    command_manifest_provenance_verified = False
    network_policy_verified = False
    runtime_identity_verified = True
    if spec is None:
        reasons.extend(("pinned_image_identity_missing", "command_manifest_identity_missing"))
    else:
        runtime_identity_verified = spec.runtime_version == version
        if not runtime_identity_verified:
            reasons.append("runtime_version_mismatch")
        image_result, probe_error = _probe(
            run,
            [executable, "image", "inspect", "--format", "{{.Id}}", spec.image_digest],
        )
        actual_image = str(getattr(image_result, "stdout", "")).strip() if image_result is not None else ""
        image_identity_verified = (
            image_result is not None
            and image_result.returncode == 0
            and actual_image == spec.image_digest
        )
        if not image_identity_verified:
            reasons.append(probe_error or "pinned_image_identity_unverified")
        # The digest is structurally pinned, but this phase has no trusted
        # manifest registry and does not prove that the digest came from one.
        command_manifest_pinned = bool(spec.command_manifest_digest)
        reasons.append("command_manifest_provenance_unverified")
        reasons.append("network_policy_unverified")
        unsupported = sorted(set(spec.required_security_controls) - SUPPORTED_SECURITY_CONTROLS)
        if unsupported:
            reasons.extend(f"security_control_unsupported:{item}" for item in unsupported)
            security_controls_verified = False
        if not command_manifest_pinned:
            reasons.append("command_manifest_identity_unverified")

    if not runtime_identity_verified:
        reasons.append("runtime_identity_unverified")
    available = (
        runtime_identity_verified
        and security_controls_verified
        and image_identity_verified
        and command_manifest_verified
        and command_manifest_provenance_verified
        and network_policy_verified
        and not reasons
    )
    report = RuntimeCapabilityReport(
        runtime_name=SUPPORTED_RUNTIME,
        executable=executable,
        version=version,
        detected=True,
        version_verified=True,
        rootless_verified=rootless_verified,
        security_controls_verified=security_controls_verified,
        image_identity_verified=image_identity_verified,
        command_manifest_verified=command_manifest_verified,
        validation_outcome=ValidationOutcome.VALID if available else ValidationOutcome.SANDBOX_UNAVAILABLE,
        command_manifest_pinned=command_manifest_pinned,
        command_manifest_provenance_verified=command_manifest_provenance_verified,
        network_policy_verified=network_policy_verified,
        reasons=tuple(reasons),
        verified_controls=verified_controls,
    )
    if available:
        object.__setattr__(report, "_attestation", _DETECTOR_ATTESTATION)
    return report


def require_available(report: RuntimeCapabilityReport) -> RuntimeCapabilityReport:
    if not report.available:
        raise SandboxUnavailable(report.reasons[0] if report.reasons else "capability_validation_failed", details=report.reasons)
    return report


def _probe(run: Callable[..., Any], argv: list[str]) -> tuple[Any | None, str | None]:
    try:
        result = run(argv, check=False, capture_output=True, text=True, timeout=2.0, shell=False)
    except (OSError, subprocess.TimeoutExpired, TypeError):
        return None, "runtime_probe_failed"
    stdout = getattr(result, "stdout", "")
    stderr = getattr(result, "stderr", "")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return None, "runtime_probe_output_invalid"
    if (
        len(stdout.encode("utf-8")) > MAX_RUNTIME_PROBE_STDOUT_BYTES
        or len(stderr.encode("utf-8")) > MAX_RUNTIME_PROBE_STDERR_BYTES
    ):
        return None, "runtime_probe_output_too_large"
    return result, None


def _unavailable_report(
    executable: str | None,
    version: str | None,
    detected: bool,
    reason: str,
) -> RuntimeCapabilityReport:
    return RuntimeCapabilityReport(
        runtime_name=SUPPORTED_RUNTIME,
        executable=executable,
        version=version,
        detected=detected,
        version_verified=False,
        rootless_verified=False,
        security_controls_verified=False,
        image_identity_verified=False,
        command_manifest_verified=False,
        validation_outcome=ValidationOutcome.SANDBOX_UNAVAILABLE,
        reasons=(reason,),
    )


def _validate_digest(value: str, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != MAX_DIGEST_LENGTH or not _DIGEST_RE.fullmatch(value):
        raise ValueError(f"{field_name}_must_be_pinned_sha256")


def _validate_positive_finite(value: float, maximum: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name}_invalid")
    if value <= 0 or value > maximum:
        raise ValueError(f"{field_name}_out_of_bounds")


def _validate_nonnegative_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{field_name}_invalid")


def _validate_int_limit(value: int, minimum: int, maximum: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise ValueError(f"{field_name}_out_of_bounds")
