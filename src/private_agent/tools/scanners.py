from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from private_agent.bugbounty.models import ScopeDefinition
from private_agent.bugbounty.scanners import HttpxAdapter, NucleiAdapter, ScanStatus, ScannerLimits
from private_agent.bugbounty.trusted_tools import TrustedToolInstallSpec
from private_agent.skills.sandbox_runtime import RuntimeActivationProof
from private_agent.tools.contracts import ToolResult


class _ScannerToolBase:
    required_capabilities = ["NETWORK_ACCESS"]
    declared_capabilities = ["NETWORK_ACCESS"]
    permission_level = "public_read"
    risk_level = "HIGH"
    network_required = True
    filesystem_required = False

    def __init__(
        self,
        *,
        limits: ScannerLimits | None = None,
        runtime_proof: RuntimeActivationProof | None = None,
        external_sandbox_available: bool | None = None,
    ) -> None:
        if external_sandbox_available is not None:
            raise TypeError("external_sandbox_available_flag_not_supported")
        if runtime_proof is not None and type(runtime_proof) is not RuntimeActivationProof:
            raise TypeError("runtime_activation_proof_invalid")
        self.limits = limits or ScannerLimits()
        self.runtime_proof = runtime_proof

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scope": {"type": "object"},
                "target": {"type": "string"},
                "identity": {"type": "string"},
                "resource_id": {"type": "string"},
            },
            "required": ["scope", "target", "identity", "resource_id"],
            "additionalProperties": False,
        }

    @staticmethod
    def _scope(payload: Any) -> ScopeDefinition:
        if not isinstance(payload, Mapping):
            raise ValueError("scope_object_required")
        raw = dict(payload)
        if "authorization_reference" not in raw and isinstance(raw.get("scope_reference"), str):
            raw["authorization_reference"] = raw.pop("scope_reference")
        return ScopeDefinition.from_dict(raw)

    @staticmethod
    def _common_inputs(inputs: Mapping[str, Any]) -> tuple[ScopeDefinition, str, str, str]:
        scope = _ScannerToolBase._scope(inputs.get("scope"))
        target = inputs.get("target")
        identity = inputs.get("identity")
        resource_id = inputs.get("resource_id")
        if not all(isinstance(value, str) and value for value in (target, identity, resource_id)):
            raise ValueError("scanner_target_binding_required")
        return scope, target, identity, resource_id

    def _sandbox_unavailable(self) -> ToolResult:
        return ToolResult.failed(
            "scanner_sandbox_unavailable",
            "External sandbox execution is unavailable; scanner execution is quarantined",
            details={"status": ScanStatus.UNAVAILABLE.value},
        )

    @staticmethod
    def _sandbox_executor_unavailable() -> ToolResult:
        return ToolResult.failed(
            "scanner_sandbox_executor_unavailable",
            "A runtime proof without a bound external executor cannot authorize host subprocess execution",
            details={"status": ScanStatus.UNAVAILABLE.value},
        )


class HttpxReconTool(_ScannerToolBase):
    name = "httpx_recon"
    planned_effect = "Run bounded ProjectDiscovery httpx reconnaissance against one explicitly authorized IP-literal target"

    def __init__(
        self,
        *,
        binary_path: str = "httpx",
        limits: ScannerLimits | None = None,
        runtime_proof: RuntimeActivationProof | None = None,
        external_sandbox_available: bool | None = None,
        trusted_installation: TrustedToolInstallSpec | None = None,
    ) -> None:
        super().__init__(
            limits=limits,
            runtime_proof=runtime_proof,
            external_sandbox_available=external_sandbox_available,
        )
        self.adapter = HttpxAdapter(
            binary_path=binary_path,
            limits=self.limits,
            trusted_installation=trusted_installation,
            require_trusted_provenance=True,
        )

    def description(self) -> str:
        return "Controlled ProjectDiscovery httpx metadata reconnaissance; no vulnerability confirmation"

    def output_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"scan": {"type": "object"}}, "required": ["scan"], "additionalProperties": False}

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        del context
        if self.runtime_proof is None or not self.runtime_proof.ready:
            return self._sandbox_unavailable()
        # A proof alone is not an execution boundary.  Until an OCI-backed
        # executor is wired in, never fall back to the host adapter.
        return self._sandbox_executor_unavailable()


class NucleiScanTool(_ScannerToolBase):
    name = "nuclei_scan"
    planned_effect = "Run bounded non-destructive ProjectDiscovery Nuclei templates against one explicitly authorized IP-literal target"

    def __init__(
        self,
        *,
        binary_path: str = "nuclei",
        templates_root: str | Path | None = None,
        expected_manifest_sha256: str = "",
        limits: ScannerLimits | None = None,
        runtime_proof: RuntimeActivationProof | None = None,
        external_sandbox_available: bool | None = None,
        trusted_installation: TrustedToolInstallSpec | None = None,
    ) -> None:
        super().__init__(
            limits=limits,
            runtime_proof=runtime_proof,
            external_sandbox_available=external_sandbox_available,
        )
        self.adapter = NucleiAdapter(
            binary_path=binary_path,
            templates_root=Path(templates_root or "/__unconfigured_nuclei_templates__"),
            expected_manifest_sha256=expected_manifest_sha256,
            limits=self.limits,
            trusted_installation=trusted_installation,
            require_trusted_provenance=True,
        )

    def description(self) -> str:
        return "Controlled non-destructive ProjectDiscovery Nuclei candidate scan; every match remains a hypothesis"

    def output_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"scan": {"type": "object"}}, "required": ["scan"], "additionalProperties": False}

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        del context
        if self.runtime_proof is None or not self.runtime_proof.ready:
            return self._sandbox_unavailable()
        return self._sandbox_executor_unavailable()


def build_scanner_limits() -> ScannerLimits:
    """Return the conservative process-wide scanner budget."""
    return ScannerLimits(timeout_seconds=60.0, rate_limit_per_second=5, concurrency=1, max_output_bytes=1_000_000, max_records=500)


__all__ = ["HttpxReconTool", "NucleiScanTool", "build_scanner_limits"]
