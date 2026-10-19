from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
import re

from private_agent.skills.sandbox_contract import (
    RuntimeCapabilityReport,
    SandboxResult,
    SandboxSpec,
    SandboxUnavailable,
    ValidationOutcome,
    detect_podman,
)


REQUIRED_RUNTIME_CONTROLS = frozenset(
    {
        "runtime_identity",
        "image_identity",
        "command_manifest_provenance",
        "rootless",
        "resource_limits",
        "process_cleanup",
        "filesystem_restrictions",
        "network_policy",
        "output_limits",
    }
)
MAX_ATTESTATION_EVIDENCE_ITEMS = 16
MAX_ATTESTATION_EVIDENCE_LENGTH = 256
_RUNTIME_IDENTITY_RE = re.compile(r"^podman@[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:[-+][0-9A-Za-z.-]+)?$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SandboxAttestation:
    """Evidence envelope produced only by a trusted runtime adapter.

    The public constructor intentionally does not create a trusted attestation.
    A future built-in adapter must use the private factory after independently
    verifying every control in ``REQUIRED_RUNTIME_CONTROLS``.
    """

    runtime_identity: str
    image_digest: str
    command_manifest_digest: str
    verified_controls: tuple[str, ...] = ()
    evidence: Mapping[str, str] = field(default_factory=dict)
    validation_outcome: ValidationOutcome = ValidationOutcome.VALIDATION_UNKNOWN
    _adapter_attestation: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_identity, str) or not _RUNTIME_IDENTITY_RE.fullmatch(self.runtime_identity):
            raise ValueError("attestation_runtime_identity_invalid")
        if not isinstance(self.image_digest, str) or not _DIGEST_RE.fullmatch(self.image_digest):
            raise ValueError("attestation_image_digest_invalid")
        if not isinstance(self.command_manifest_digest, str) or not _DIGEST_RE.fullmatch(self.command_manifest_digest):
            raise ValueError("attestation_command_manifest_digest_invalid")
        if not isinstance(self.verified_controls, (tuple, list)):
            raise ValueError("attestation_controls_invalid")
        controls = tuple(self.verified_controls)
        if len(set(controls)) != len(controls):
            raise ValueError("attestation_controls_duplicate")
        if any(not isinstance(control, str) or control not in REQUIRED_RUNTIME_CONTROLS for control in controls):
            raise ValueError("attestation_control_unknown")
        object.__setattr__(self, "verified_controls", controls)
        if not isinstance(self.evidence, Mapping):
            raise ValueError("attestation_evidence_invalid")
        evidence = dict(self.evidence)
        if len(evidence) > MAX_ATTESTATION_EVIDENCE_ITEMS:
            raise ValueError("attestation_evidence_too_large")
        for key, value in evidence.items():
            if (
                not isinstance(key, str)
                or key not in REQUIRED_RUNTIME_CONTROLS
                or not isinstance(value, str)
                or not value
                or len(value) > MAX_ATTESTATION_EVIDENCE_LENGTH
            ):
                raise ValueError("attestation_evidence_invalid")
        object.__setattr__(self, "evidence", MappingProxyType(evidence))
        outcome = self.validation_outcome
        if isinstance(outcome, str):
            try:
                outcome = ValidationOutcome(outcome)
            except ValueError as exc:
                raise ValueError("attestation_validation_outcome_invalid") from exc
            object.__setattr__(self, "validation_outcome", outcome)
        elif not isinstance(outcome, ValidationOutcome):
            raise ValueError("attestation_validation_outcome_invalid")

    @property
    def trusted(self) -> bool:
        return (
            self.validation_outcome is ValidationOutcome.VALID
            and self._adapter_attestation is _ADAPTER_ATTESTATION
            and REQUIRED_RUNTIME_CONTROLS.issubset(self.verified_controls)
            and REQUIRED_RUNTIME_CONTROLS.issubset(self.evidence)
        )

    @classmethod
    def _from_trusted_adapter(
        cls,
        *,
        runtime_identity: str,
        image_digest: str,
        command_manifest_digest: str,
        verified_controls: tuple[str, ...],
        evidence: Mapping[str, str],
        token: object,
    ) -> "SandboxAttestation":
        if token is not _ADAPTER_ATTESTATION:
            raise ValueError("attestation_factory_forbidden")
        attestation = cls(
            runtime_identity=runtime_identity,
            image_digest=image_digest,
            command_manifest_digest=command_manifest_digest,
            verified_controls=verified_controls,
            evidence=evidence,
            validation_outcome=ValidationOutcome.VALID,
        )
        object.__setattr__(attestation, "_adapter_attestation", _ADAPTER_ATTESTATION)
        return attestation

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_identity": self.runtime_identity,
            "image_digest": self.image_digest,
            "command_manifest_digest": self.command_manifest_digest,
            "verified_controls": list(self.verified_controls),
            "evidence": dict(self.evidence),
            "validation_outcome": self.validation_outcome.value,
            "trusted": self.trusted,
        }


_ADAPTER_ATTESTATION = object()


class SandboxRuntimeAdapter(ABC):
    """Strict adapter seam; it accepts no arbitrary argv, callable, or host path."""

    @abstractmethod
    def inspect(self, spec: SandboxSpec) -> RuntimeCapabilityReport:
        """Read runtime metadata without starting a container."""

    @abstractmethod
    def attest(self, spec: SandboxSpec, report: RuntimeCapabilityReport) -> SandboxAttestation:
        """Return only independently verified controls or raise unavailable."""

    @abstractmethod
    def execute(self, spec: SandboxSpec, attestation: SandboxAttestation) -> SandboxResult:
        """Execute only through a future built-in external runtime adapter."""


class PodmanSandboxAdapter(SandboxRuntimeAdapter):
    """Read-only Podman adapter; execution remains deliberately disabled here."""

    def inspect(self, spec: SandboxSpec) -> RuntimeCapabilityReport:
        if type(spec) is not SandboxSpec:
            raise SandboxUnavailable("invalid_sandbox_spec")
        return detect_podman(spec)

    def attest(self, spec: SandboxSpec, report: RuntimeCapabilityReport) -> SandboxAttestation:
        if type(spec) is not SandboxSpec:
            raise SandboxUnavailable("invalid_sandbox_spec")
        require_runtime_capability(report)
        # Phase 15D.1A does not independently verify manifest provenance,
        # resource enforcement, cleanup, filesystem policy, or networking.
        raise SandboxUnavailable(
            "full_runtime_attestation_unavailable",
            details=tuple(sorted(REQUIRED_RUNTIME_CONTROLS)),
        )

    def execute(self, spec: SandboxSpec, attestation: SandboxAttestation) -> SandboxResult:
        del spec, attestation
        raise SandboxUnavailable("podman_execution_not_enabled")


def require_runtime_capability(report: RuntimeCapabilityReport) -> RuntimeCapabilityReport:
    if type(report) is not RuntimeCapabilityReport or not report.available:
        reasons = () if not isinstance(report, RuntimeCapabilityReport) else report.reasons
        raise SandboxUnavailable(
            reasons[0] if reasons else "runtime_capability_unavailable",
            details=tuple(reasons),
        )
    return report


def validate_attestation(
    spec: SandboxSpec,
    report: RuntimeCapabilityReport,
    attestation: SandboxAttestation,
) -> SandboxAttestation:
    """Validate exact identity, controls, provenance gate, and runtime proof."""

    if type(spec) is not SandboxSpec:
        raise SandboxUnavailable("invalid_sandbox_spec")
    if type(attestation) is not SandboxAttestation:
        raise SandboxUnavailable("attestation_not_trusted")
    if attestation.runtime_identity != spec.runtime_identity:
        raise SandboxUnavailable("attestation_runtime_mismatch")
    if attestation.image_digest != spec.image_digest:
        raise SandboxUnavailable("attestation_image_mismatch")
    if attestation.command_manifest_digest != spec.command_manifest_digest:
        raise SandboxUnavailable("attestation_manifest_mismatch")
    missing = sorted(REQUIRED_RUNTIME_CONTROLS - set(attestation.verified_controls))
    if missing:
        raise SandboxUnavailable("attestation_controls_missing", details=tuple(missing))
    if not attestation.trusted:
        raise SandboxUnavailable("attestation_not_trusted")
    require_runtime_capability(report)
    if not report.detected or not report.version_verified:
        raise SandboxUnavailable("runtime_identity_unverified")
    if report.runtime_name != spec.runtime_name or report.version != spec.runtime_version:
        raise SandboxUnavailable("runtime_identity_mismatch")
    if not report.rootless_verified:
        raise SandboxUnavailable("rootless_unverified")
    if not report.image_identity_verified:
        raise SandboxUnavailable("image_identity_unverified")
    if (
        not report.command_manifest_verified
        or not report.command_manifest_pinned
        or not report.command_manifest_provenance_verified
    ):
        raise SandboxUnavailable("command_manifest_provenance_unverified")
    if not set(spec.required_security_controls).issubset(report.verified_controls):
        raise SandboxUnavailable("required_security_controls_unverified")
    if not report.security_controls_verified or not report.network_policy_verified:
        raise SandboxUnavailable("runtime_security_controls_unverified")
    return attestation


def validate_sandbox_result(
    spec: SandboxSpec,
    attestation: SandboxAttestation,
    result: SandboxResult,
) -> SandboxResult:
    """Reject malformed or unvalidated adapter results before lifecycle use."""

    if type(result) is not SandboxResult:
        raise SandboxUnavailable("sandbox_result_malformed")
    if result.validation_outcome is not ValidationOutcome.VALID:
        raise SandboxUnavailable("sandbox_result_not_validated")
    if type(attestation) is not SandboxAttestation or not attestation.trusted:
        raise SandboxUnavailable("attestation_not_trusted")
    if attestation.runtime_identity != spec.runtime_identity:
        raise SandboxUnavailable("attestation_runtime_mismatch")
    if attestation.image_digest != spec.image_digest:
        raise SandboxUnavailable("attestation_image_mismatch")
    if attestation.command_manifest_digest != spec.command_manifest_digest:
        raise SandboxUnavailable("attestation_manifest_mismatch")
    if result.runtime_identity != spec.runtime_identity:
        raise SandboxUnavailable("sandbox_result_runtime_mismatch")
    if result.image_digest != spec.image_digest:
        raise SandboxUnavailable("sandbox_result_image_mismatch")
    if result.command_manifest_digest != spec.command_manifest_digest:
        raise SandboxUnavailable("sandbox_result_manifest_mismatch")
    return result


class ExternalSandboxRunner:
    """Fail-closed facade; no runtime execution is enabled in this subphase."""

    __slots__ = ()

    def __init__(self, *, adapter: SandboxRuntimeAdapter | None = None) -> None:
        if adapter is not None:
            raise TypeError("custom_adapter_injection_disabled")

    def execute(self, spec: SandboxSpec) -> SandboxResult:
        if type(spec) is not SandboxSpec:
            return SandboxResult.unavailable("invalid_sandbox_spec")
        adapter = PodmanSandboxAdapter()
        try:
            report = adapter.inspect(spec)
            attestation = adapter.attest(spec, report)
            validate_attestation(spec, report, attestation)
            result = adapter.execute(spec, attestation)
            return validate_sandbox_result(spec, attestation, result)
        except SandboxUnavailable as exc:
            return SandboxResult.unavailable(exc.reason, capabilities={"details": list(exc.details)})
        except (TypeError, ValueError):
            return SandboxResult.unavailable("sandbox_adapter_contract_invalid")
