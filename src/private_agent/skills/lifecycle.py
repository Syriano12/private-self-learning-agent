from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from private_agent.core.experience import ExperienceAction, ExperienceRecord, sanitize_value
from private_agent.core.learning import LearningEngine
from private_agent.core.memory import ExperienceMemory
from private_agent.core.reflection import ReflectionEngine
from private_agent.security import SecurityController
from private_agent.skills.sandbox_contract import SandboxResult, ValidationOutcome, detect_podman
from private_agent.tools.contracts import validate_schema_value
from private_agent.skills.registry import SkillRegistry


SKILL_STATUSES = {
    "CANDIDATE",
    "ANALYZING",
    "TESTING",
    "QUARANTINED",
    "APPROVED",
    "ACTIVE",
    "DEGRADED",
    "ROLLED_BACK",
    "REJECTED",
}


@dataclass
class SkillContract:
    max_execution_time: float = 2.0
    max_output_size: int = 64_000
    max_memory_bytes: int = 128 * 1024 * 1024
    allowed_imports: list[str] = field(default_factory=list)
    denied_imports: list[str] = field(
        default_factory=lambda: [
            "os",
            "sys",
            "subprocess",
            "pty",
            "commands",
            "ctypes",
            "pickle",
        ]
    )
    network_allowed: bool = False
    allowed_network_domains: list[str] = field(default_factory=list)
    filesystem_allowed: bool = False
    workspace_root: str = ""
    declared_capabilities: list[str] = field(default_factory=list)
    require_strong_isolation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return sanitize_value(asdict(self))

    @classmethod
    def from_candidate(cls, candidate: "SkillCandidate") -> "SkillContract":
        resources = candidate.required_resources if isinstance(candidate.required_resources, dict) else {}
        return cls(
            max_execution_time=float(resources.get("max_execution_time", 2.0)),
            max_output_size=int(resources.get("max_output_size", 64_000)),
            max_memory_bytes=int(resources.get("max_memory_bytes", 128 * 1024 * 1024)),
            allowed_imports=list(candidate.allowed_imports),
            network_allowed="network" in candidate.declared_capabilities,
            allowed_network_domains=list(candidate.allowed_network_domains),
            filesystem_allowed="filesystem" in candidate.declared_capabilities,
            workspace_root=str(resources.get("workspace_root", "")),
            declared_capabilities=list(candidate.declared_capabilities),
            require_strong_isolation=bool(resources.get("require_strong_isolation", False)),
        )


@dataclass
class SkillCandidate:
    skill_id: str
    name: str
    version: int
    description: str
    source: str
    entry_point: str
    declared_capabilities: list[str] = field(default_factory=list)
    required_resources: dict[str, Any] = field(default_factory=dict)
    allowed_imports: list[str] = field(default_factory=list)
    allowed_network_domains: list[str] = field(default_factory=list)
    expected_inputs: dict[str, Any] = field(default_factory=dict)
    expected_outputs: dict[str, Any] = field(default_factory=dict)
    tests: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    status: str = "CANDIDATE"
    parent_version: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    implementation: Callable[[dict[str, Any]], Any] | None = field(default=None, repr=False, compare=False)
    mutation_test_runner: Callable[[str, "MutationCase"], bool] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.status not in SKILL_STATUSES:
            raise ValueError(f"invalid_skill_status:{self.status}")
        if self.version < 1:
            raise ValueError("skill_version_must_be_positive")

    def to_dict(self) -> dict[str, Any]:
        # Build the persisted metadata explicitly.  Calling ``asdict`` would
        # deepcopy arbitrary callable fields before they could be removed.
        payload = {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "source": self.source,
            "entry_point": self.entry_point,
            "declared_capabilities": self.declared_capabilities,
            "required_resources": self.required_resources,
            "allowed_imports": self.allowed_imports,
            "allowed_network_domains": self.allowed_network_domains,
            "expected_inputs": self.expected_inputs,
            "expected_outputs": self.expected_outputs,
            "tests": self.tests,
            "metadata": self.metadata,
            "provenance": self.provenance,
            "status": self.status,
            "parent_version": self.parent_version,
            "created_at": self.created_at,
        }
        return sanitize_value(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillCandidate":
        return cls(
            skill_id=str(payload["skill_id"]),
            name=str(payload["name"]),
            version=int(payload["version"]),
            description=str(payload.get("description", "")),
            source=str(payload.get("source", "")),
            entry_point=str(payload.get("entry_point", "")),
            declared_capabilities=list(payload.get("declared_capabilities", [])),
            required_resources=dict(payload.get("required_resources", {})),
            allowed_imports=list(payload.get("allowed_imports", [])),
            allowed_network_domains=list(payload.get("allowed_network_domains", [])),
            expected_inputs=dict(payload.get("expected_inputs", {})),
            expected_outputs=dict(payload.get("expected_outputs", {})),
            tests=list(payload.get("tests", [])),
            metadata=dict(payload.get("metadata", {})),
            provenance=dict(payload.get("provenance", {})),
            status=str(payload.get("status", "CANDIDATE")),
            parent_version=payload.get("parent_version"),
            created_at=str(payload.get("created_at", "")),
        )


@dataclass
class SkillFinding:
    severity: str
    rule_id: str
    location: dict[str, int]
    message: str
    evidence: str
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return sanitize_value(asdict(self))


@dataclass
class ContractReport:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StaticAnalysisReport:
    passed: bool
    findings: list[SkillFinding] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "findings": [finding.to_dict() for finding in self.findings],
            "imports": self.imports,
            "calls": self.calls,
        }


class SkillContractValidator:
    SUPPORTED_CAPABILITIES = {"network", "filesystem", "environment", "subprocess"}

    def validate(self, candidate: SkillCandidate, contract: SkillContract) -> ContractReport:
        errors: list[str] = []
        warnings: list[str] = []
        if not candidate.name.strip():
            errors.append("skill_name_required")
        if not candidate.entry_point.strip():
            errors.append("entry_point_required")
        if not isinstance(candidate.expected_inputs, dict) or not isinstance(candidate.expected_outputs, dict):
            errors.append("input_output_contracts_must_be_objects")
        if contract.max_execution_time <= 0:
            errors.append("max_execution_time_must_be_positive")
        if contract.max_output_size <= 0 or contract.max_memory_bytes <= 0:
            errors.append("resource_limits_must_be_positive")
        unknown = sorted(set(candidate.declared_capabilities) - self.SUPPORTED_CAPABILITIES)
        errors.extend(f"unsupported_capability:{capability}" for capability in unknown)
        if candidate.allowed_network_domains and "network" not in candidate.declared_capabilities:
            errors.append("network_domains_without_network_capability")
        if "network" in candidate.declared_capabilities and not contract.network_allowed:
            errors.append("network_capability_not_allowed_by_contract")
        if "filesystem" in candidate.declared_capabilities:
            if not contract.filesystem_allowed:
                errors.append("filesystem_capability_not_allowed_by_contract")
            if not contract.workspace_root:
                errors.append("filesystem_workspace_root_required")
        if "environment" in candidate.declared_capabilities:
            warnings.append("environment_capability_requires_runtime_restrictions")
        if "subprocess" in candidate.declared_capabilities:
            errors.append("subprocess_capability_not_supported_in_phase8")
        if contract.require_strong_isolation:
            warnings.append("strong_isolation_must_be_confirmed_by_sandbox")
        return ContractReport(not errors, errors, warnings)


class _StaticVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: list[tuple[str, ast.AST]] = []
        self.calls: list[tuple[str, ast.Call]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append((alias.name, node))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append((node.module or "", node))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append((_dotted_name(node.func), node))
        self.generic_visit(node)


class SkillStaticAnalyzer:
    NETWORK_MODULES = {"socket", "requests", "httpx", "urllib", "urllib3", "aiohttp"}
    FILESYSTEM_MODULES = {"pathlib", "shutil", "tempfile"}
    CREDENTIAL_MODULES = {"keyring", "boto3", "google.auth"}

    def analyze(self, candidate: SkillCandidate, contract: SkillContract) -> StaticAnalysisReport:
        findings: list[SkillFinding] = []
        try:
            tree = ast.parse(candidate.source or "", filename=f"{candidate.name}.py", mode="exec")
        except SyntaxError as exc:
            findings.append(
                SkillFinding(
                    "ERROR",
                    "syntax_error",
                    {"line": int(exc.lineno or 0), "column": int(exc.offset or 0)},
                    "Candidate source is not valid Python syntax",
                    str(exc),
                    True,
                )
            )
            return StaticAnalysisReport(False, findings)

        visitor = _StaticVisitor()
        visitor.visit(tree)
        imported_names = sorted({name for name, _ in visitor.imports if name})
        called_names = sorted({name for name, _ in visitor.calls if name})
        allowed_imports = set(contract.allowed_imports) | set(candidate.allowed_imports)
        denied_imports = set(contract.denied_imports)

        for module, node in visitor.imports:
            root = module.split(".")[0]
            location = _location(node)
            if root in denied_imports or module in denied_imports:
                findings.append(
                    SkillFinding(
                        "ERROR",
                        "dangerous_import",
                        location,
                        f"Denied import detected: {module}",
                        module,
                        True,
                    )
                )
            elif module not in allowed_imports and root not in allowed_imports:
                findings.append(
                    SkillFinding(
                        "ERROR",
                        "undeclared_import",
                        location,
                        f"Import is not declared in the capability contract: {module}",
                        module,
                        True,
                    )
                )
            if root in self.NETWORK_MODULES:
                if "network" not in candidate.declared_capabilities or not contract.network_allowed:
                    findings.append(
                        SkillFinding(
                            "ERROR",
                            "undeclared_network",
                            location,
                            "Network library used without an allowed network capability",
                            module,
                            True,
                        )
                    )
            if root in self.CREDENTIAL_MODULES:
                findings.append(
                    SkillFinding(
                        "ERROR",
                        "credential_access",
                        location,
                        "Credential-oriented import is not allowed for a learned skill",
                        module,
                        True,
                    )
                )
            if root in self.FILESYSTEM_MODULES and "filesystem" not in candidate.declared_capabilities:
                findings.append(
                    SkillFinding(
                        "ERROR",
                        "undeclared_filesystem",
                        location,
                        "Filesystem library used without a filesystem capability",
                        module,
                        True,
                    )
                )

        for dotted, node in visitor.calls:
            simple = dotted.rsplit(".", 1)[-1]
            location = _location(node)
            if simple in {"eval", "exec", "compile", "__import__"}:
                findings.append(
                    SkillFinding(
                        "ERROR",
                        "dynamic_code_execution",
                        location,
                        "Dynamic code execution is forbidden",
                        dotted,
                        True,
                    )
                )
            elif dotted in {"os.system", "os.popen", "subprocess.run", "subprocess.Popen", "subprocess.call"} or dotted.startswith("subprocess."):
                findings.append(
                    SkillFinding(
                        "ERROR",
                        "shell_or_subprocess",
                        location,
                        "Shell or unrestricted subprocess execution is forbidden",
                        dotted,
                        True,
                    )
                )
            elif simple == "open":
                if "filesystem" not in candidate.declared_capabilities or not contract.filesystem_allowed:
                    findings.append(
                        SkillFinding(
                            "ERROR",
                            "undeclared_filesystem_access",
                            location,
                            "Filesystem access is not allowed by the contract",
                            dotted,
                            True,
                        )
                    )
                elif not contract.workspace_root:
                    findings.append(
                        SkillFinding(
                            "ERROR",
                            "unrestricted_filesystem_path",
                            location,
                            "Filesystem capability lacks a restricted workspace root",
                            dotted,
                            True,
                        )
                    )
            elif dotted in {"os.getenv", "os.environ.get", "os.environ.__getitem__"} or dotted.startswith("os.environ"):
                findings.append(
                    SkillFinding(
                        "ERROR",
                        "environment_or_credential_access",
                        location,
                        "Environment and credential access is forbidden for learned skills",
                        dotted,
                        True,
                    )
                )
            elif dotted.startswith("socket.") or dotted.startswith("requests.") or dotted.startswith("httpx."):
                if "network" not in candidate.declared_capabilities or not contract.network_allowed:
                    findings.append(
                        SkillFinding(
                            "ERROR",
                            "undeclared_network_call",
                            location,
                            "Network call is not allowed by the contract",
                            dotted,
                            True,
                        )
                    )

        passed = not any(finding.blocking for finding in findings)
        return StaticAnalysisReport(passed, findings, imported_names, called_names)


@dataclass
class MutationCase:
    mutant_id: str
    category: str
    description: str
    mutated_source: str
    killed: bool = False
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_value(
            {
                "mutant_id": self.mutant_id,
                "category": self.category,
                "description": self.description,
                "source_sha256": hashlib.sha256(self.mutated_source.encode("utf-8")).hexdigest(),
                "killed": self.killed,
                "evidence": self.evidence,
            }
        )


@dataclass
class MutationReport:
    passed: bool
    total_mutants: int
    killed_mutants: int
    survived_mutants: int
    mutation_score: float
    threshold: float
    mutants: list[MutationCase] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_mutants": self.total_mutants,
            "killed_mutants": self.killed_mutants,
            "survived_mutants": self.survived_mutants,
            "mutation_score": self.mutation_score,
            "threshold": self.threshold,
            "mutants": [mutant.to_dict() for mutant in self.mutants],
            "reason": self.reason,
        }


class _MutationTransformer(ast.NodeTransformer):
    def __init__(self, category: str, target: int) -> None:
        self.category = category
        self.target = target
        self.index = 0
        self.changed = False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if self.category == "comparison_inversion" and node.ops:
            for index, operator in enumerate(node.ops):
                if self.index == self.target and type(operator) in {ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq}:
                    replacements = {
                        ast.Gt: ast.LtE,
                        ast.Lt: ast.GtE,
                        ast.GtE: ast.Lt,
                        ast.LtE: ast.Gt,
                        ast.Eq: ast.NotEq,
                        ast.NotEq: ast.Eq,
                    }
                    node.ops[index] = replacements[type(operator)]()
                    self.changed = True
                    return node
                self.index += 1
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.category == "boolean_inversion" and isinstance(node.value, bool):
            if self.index == self.target:
                node.value = not node.value
                self.changed = True
                return node
            self.index += 1
        return node


def _generate_mutants(source: str, max_mutants: int) -> list[MutationCase]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    candidates: list[tuple[str, int, str]] = []
    compare_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Compare) for _ in node.ops)
    bool_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, bool))
    candidates.extend(("comparison_inversion", index, "invert a comparison operator") for index in range(compare_count))
    candidates.extend(("boolean_inversion", index, "invert a boolean constant") for index in range(bool_count))
    mutants: list[MutationCase] = []
    for category, target, description in candidates[: max(0, max_mutants)]:
        mutated_tree = copy.deepcopy(tree)
        transformer = _MutationTransformer(category, target)
        transformer.visit(mutated_tree)
        if not transformer.changed:
            continue
        mutated_source = ast.unparse(ast.fix_missing_locations(mutated_tree))
        mutant_id = "mutant-" + hashlib.sha256(
            f"{category}:{target}:{mutated_source}".encode("utf-8")
        ).hexdigest()[:16]
        mutants.append(MutationCase(mutant_id, category, description, mutated_source))
    return mutants


class MutationTestingEngine:
    def __init__(self, *, max_mutants: int = 8, threshold: float = 1.0) -> None:
        self.max_mutants = max(1, max_mutants)
        self.threshold = max(0.0, min(1.0, float(threshold)))

    def run(
        self,
        candidate: SkillCandidate,
        test_runner: Callable[[str, MutationCase], bool] | None = None,
    ) -> MutationReport:
        mutants = _generate_mutants(candidate.source, self.max_mutants)
        runner = test_runner if test_runner is not None else candidate.mutation_test_runner
        if not mutants:
            return MutationReport(False, 0, 0, 0, 0.0, self.threshold, reason="no_supported_mutants_generated")
        if runner is None:
            return MutationReport(
                False,
                len(mutants),
                0,
                len(mutants),
                0.0,
                self.threshold,
                mutants,
                "mutation_test_runner_required",
            )
        return MutationReport(
            False,
            len(mutants),
            0,
            len(mutants),
            0.0,
            self.threshold,
            mutants,
            "external_sandbox_required",
        )


@dataclass
class SandboxCapabilities:
    process_isolation: bool
    timeout: bool
    resource_limits: bool
    environment_restrictions: bool
    filesystem_restrictions: bool
    network_restrictions: bool
    strong_isolation: bool
    isolation_level: str
    unavailable: list[str] = field(default_factory=list)
    external_sandbox: bool = False
    runtime_name: str = ""
    runtime_version: str = ""
    runtime_executable: str = ""
    runtime_detected: bool = False
    runtime_verified: bool = False
    security_controls_verified: bool = False
    validation_outcome: str = ValidationOutcome.SANDBOX_UNAVAILABLE.value

    @classmethod
    def detect(cls) -> "SandboxCapabilities":
        report = detect_podman()
        legacy_unavailable = [
            "process_isolation",
            "resource_limits",
            "filesystem_restrictions",
            "network_restrictions",
            "strong_isolation",
            "external_sandbox",
        ]
        unavailable = sorted(set(legacy_unavailable).union(report.reasons))
        return cls(
            process_isolation=False,
            timeout=False,
            resource_limits=False,
            environment_restrictions=False,
            filesystem_restrictions=False,
            network_restrictions=False,
            strong_isolation=False,
            external_sandbox=report.available,
            isolation_level="verified" if report.available else "unavailable",
            unavailable=unavailable,
            runtime_name=report.runtime_name,
            runtime_version=report.version or "",
            runtime_executable=report.executable or "",
            runtime_detected=report.detected,
            runtime_verified=report.version_verified,
            security_controls_verified=report.security_controls_verified,
            validation_outcome=report.validation_outcome.value,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MAX_SANDBOX_MESSAGE_BYTES = 64_000
_SANDBOX_RESULT_KEYS = {"status", "output", "error", "duration_ms"}
_SANDBOX_RESULT_STATUSES = {"success", "failed"}


def _validate_json_primitives(value: Any, *, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError("sandbox_result_nesting_limit_exceeded")
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_primitives(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("sandbox_result_non_string_key")
        for item in value.values():
            _validate_json_primitives(item, depth=depth + 1)
        return
    raise ValueError("sandbox_result_non_primitive_value")


def _reject_json_constant(value: str) -> None:
    raise ValueError(value)


def decode_sandbox_result(payload: bytes | bytearray, *, max_bytes: int = MAX_SANDBOX_MESSAGE_BYTES) -> dict[str, Any]:
    """Decode the future external-runner protocol without Python object loading.

    The current project intentionally has no external runner, so this decoder is
    not a fallback execution path. It defines the only acceptable result shape
    for a future runner: bounded UTF-8 JSON containing primitives only.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("sandbox_result_must_be_bytes")
    if max_bytes <= 0 or len(payload) > max_bytes:
        raise ValueError("sandbox_result_too_large")
    try:
        decoded = bytes(payload).decode("utf-8")
        message = json.loads(
            decoded,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, TypeError):
        raise ValueError("sandbox_result_malformed") from None
    if not isinstance(message, dict):
        raise ValueError("sandbox_result_malformed")
    if set(message) - _SANDBOX_RESULT_KEYS:
        raise ValueError("sandbox_result_unexpected_fields")
    status = message.get("status")
    if status not in _SANDBOX_RESULT_STATUSES:
        raise ValueError("sandbox_result_invalid_status")
    if status == "success" and "output" not in message:
        raise ValueError("sandbox_result_missing_output")
    if status == "failed" and "error" not in message:
        raise ValueError("sandbox_result_missing_error")
    try:
        _validate_json_primitives(message)
    except ValueError:
        raise
    if "error" in message and not isinstance(message["error"], str):
        raise ValueError("sandbox_result_invalid_error")
    if "duration_ms" in message and (
        isinstance(message["duration_ms"], bool)
        or not isinstance(message["duration_ms"], (int, float))
        or message["duration_ms"] < 0
    ):
        raise ValueError("sandbox_result_invalid_duration")
    return message


class SkillSandbox:
    def __init__(self, capabilities: SandboxCapabilities | None = None) -> None:
        if capabilities is not None and type(capabilities) is not SandboxCapabilities:
            raise TypeError("untrusted_sandbox_capabilities_not_supported")
        self.capabilities = capabilities if capabilities is not None else SandboxCapabilities.detect()

    def execute(self, candidate: SkillCandidate, contract: SkillContract, inputs: dict[str, Any]) -> SandboxResult:
        # A local fork/pipe is not a security sandbox and its old result
        # channel previously deserialized arbitrary Python objects. Never
        # execute an untrusted implementation until an external sandbox is
        # configured. This guard also makes direct calls fail closed.
        return SandboxResult(
            "QUARANTINED",
            error="external_sandbox_unavailable",
            capabilities=self.capabilities.to_dict(),
            applied_restrictions=["no_untrusted_host_execution"],
            validation_outcome=ValidationOutcome.SANDBOX_UNAVAILABLE,
        )


@dataclass
class RuntimeVerificationReport:
    passed: bool
    status: str
    errors: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkillRuntimeVerifier:
    def verify(self, candidate: SkillCandidate, contract: SkillContract, inputs: dict[str, Any], sandbox: SandboxResult) -> RuntimeVerificationReport:
        errors: list[str] = []
        evidence = [{"sandbox_status": sandbox.status, "duration_ms": sandbox.duration_ms}]
        input_errors = validate_schema_value(inputs, candidate.expected_inputs, path="skill_input")
        errors.extend(input_errors)
        if sandbox.status != "PASSED":
            errors.append(f"sandbox_status:{sandbox.status}:{sandbox.error}")
        if sandbox.status == "PASSED":
            errors.extend(validate_schema_value(sandbox.output, candidate.expected_outputs, path="skill_output"))
            serialized = json.dumps(sanitize_value(sandbox.output), ensure_ascii=False, default=str)
            if len(serialized) > contract.max_output_size:
                errors.append("output_size_limit_exceeded")
        return RuntimeVerificationReport(not errors, sandbox.status, errors, evidence)


@dataclass
class CrossVerificationReport:
    passed: bool
    errors: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkillCrossVerifier:
    def verify(self, candidate: SkillCandidate, output: Any) -> CrossVerificationReport:
        errors: list[str] = []
        evidence: list[dict[str, Any]] = []
        sanitized = sanitize_value(output)
        if sanitized != output:
            errors.append("secret_like_output_detected")
        invariants = candidate.metadata.get("invariants", []) if isinstance(candidate.metadata, dict) else []
        for invariant in invariants:
            if invariant == "non_empty_result":
                if not isinstance(output, dict) or not output.get("result"):
                    errors.append("invariant_failed:non_empty_result")
                else:
                    evidence.append({"invariant": invariant, "passed": True})
            elif isinstance(invariant, dict) and invariant.get("field"):
                value = output.get(invariant["field"]) if isinstance(output, dict) else None
                if invariant.get("non_empty") and not value:
                    errors.append(f"invariant_failed:{invariant['field']}")
                else:
                    evidence.append({"invariant": invariant, "passed": True})
        if not errors:
            evidence.append({"independent_check": "sanitized_output_and_invariants", "passed": True})
        return CrossVerificationReport(not errors, errors, evidence)


@dataclass
class ApprovalDecision:
    status: str
    approved: bool
    reasons: list[str] = field(default_factory=list)
    required_checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkillApprovalEngine:
    def decide(
        self,
        static: StaticAnalysisReport,
        contract: ContractReport,
        mutation: MutationReport,
        sandbox: SandboxResult,
        runtime: RuntimeVerificationReport,
        cross: CrossVerificationReport,
    ) -> ApprovalDecision:
        checks = {
            "static_analysis": static.passed,
            "contract_validation": contract.passed,
            "mutation_threshold": mutation.passed,
            "sandbox_execution": sandbox.status == "PASSED",
            "runtime_verification": runtime.passed,
            "cross_verification": cross.passed,
            "no_blocking_findings": not any(item.blocking for item in static.findings),
        }
        reasons: list[str] = []
        for name, passed in checks.items():
            if not passed:
                reasons.append(f"check_failed:{name}")
        approved = all(checks.values())
        if approved:
            return ApprovalDecision("APPROVED", True, reasons, checks)
        quarantine_reasons = {"sandbox_execution", "runtime_verification"}
        status = "QUARANTINED" if any(not checks[name] for name in quarantine_reasons) and not static.findings else "REJECTED"
        return ApprovalDecision(status, False, reasons, checks)


@dataclass
class LifecycleResult:
    candidate: SkillCandidate
    contract: ContractReport
    analysis: StaticAnalysisReport
    mutation: MutationReport
    sandbox: SandboxResult
    runtime: RuntimeVerificationReport
    cross_verification: CrossVerificationReport
    approval: ApprovalDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "contract": self.contract.to_dict(),
            "analysis": self.analysis.to_dict(),
            "mutation": self.mutation.to_dict(),
            "sandbox": self.sandbox.to_dict(),
            "runtime": self.runtime.to_dict(),
            "cross_verification": self.cross_verification.to_dict(),
            "approval": self.approval.to_dict(),
        }


@dataclass
class ActiveExecutionResult:
    status: str
    skill_id: str
    version: int
    sandbox: SandboxResult
    runtime: RuntimeVerificationReport
    cross_verification: CrossVerificationReport
    metrics: dict[str, Any]
    rollback: dict[str, Any] | None = None
    experience_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "skill_id": self.skill_id,
            "version": self.version,
            "sandbox": self.sandbox.to_dict(),
            "runtime": self.runtime.to_dict(),
            "cross_verification": self.cross_verification.to_dict(),
            "metrics": self.metrics,
            "rollback": self.rollback,
            "experience_id": self.experience_id,
        }


class SkillRuntimeMonitor:
    def __init__(self, *, min_executions: int = 3, max_failure_rate: float = 0.5) -> None:
        self.min_executions = max(1, min_executions)
        self.max_failure_rate = max(0.0, min(1.0, max_failure_rate))

    def degraded(self, metrics: dict[str, Any]) -> bool:
        executions = int(metrics.get("executions", 0))
        if executions < self.min_executions:
            return False
        return 1.0 - float(metrics.get("success_rate", 0.0)) > self.max_failure_rate


class SkillLifecycleManager:
    """Coordinates the zero-trust skill lifecycle without granting learned code authority."""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        analyzer: SkillStaticAnalyzer | None = None,
        contract_validator: SkillContractValidator | None = None,
        mutation_engine: MutationTestingEngine | None = None,
        sandbox: SkillSandbox | None = None,
        runtime_verifier: SkillRuntimeVerifier | None = None,
        cross_verifier: SkillCrossVerifier | None = None,
        approval_engine: SkillApprovalEngine | None = None,
        monitor: SkillRuntimeMonitor | None = None,
        experience_memory: ExperienceMemory | None = None,
        reflection_engine: ReflectionEngine | None = None,
        learning_engine: LearningEngine | None = None,
        security_controller: SecurityController | None = None,
    ) -> None:
        if mutation_engine is not None and type(mutation_engine) is not MutationTestingEngine:
            raise TypeError("untrusted_mutation_engine_not_supported")
        if sandbox is not None and type(sandbox) is not SkillSandbox:
            raise TypeError("untrusted_sandbox_provider_not_supported")
        self.registry = registry
        self.analyzer = analyzer or SkillStaticAnalyzer()
        self.contract_validator = contract_validator or SkillContractValidator()
        self.mutation_engine = mutation_engine or MutationTestingEngine()
        self.sandbox = sandbox or SkillSandbox()
        self.runtime_verifier = runtime_verifier or SkillRuntimeVerifier()
        self.cross_verifier = cross_verifier or SkillCrossVerifier()
        self.approval_engine = approval_engine or SkillApprovalEngine()
        self.monitor = monitor or SkillRuntimeMonitor()
        self.experience_memory = experience_memory
        self.reflection_engine = reflection_engine
        self.learning_engine = learning_engine
        self.security = security_controller
        self._candidates: dict[str, SkillCandidate] = {}
        self._contracts: dict[str, SkillContract] = {}

    def submit(
        self,
        candidate: SkillCandidate,
        *,
        contract: SkillContract | None = None,
        inputs: dict[str, Any] | None = None,
        goal: str = "",
        mutation_test_runner: Callable[[str, MutationCase], bool] | None = None,
        approval_id: str = "",
    ) -> LifecycleResult:
        contract = contract or SkillContract.from_candidate(candidate)
        inputs = inputs or {}
        active = self.registry.active(candidate.name)
        if active and int(candidate.version) <= int(active["version"]):
            candidate.status = "REJECTED"
            empty_analysis = StaticAnalysisReport(False, [SkillFinding("ERROR", "version_conflict", {}, "Active version must not be overwritten", candidate.name, True)])
            empty_contract = ContractReport(False, ["active_skill_version_must_increase"], [])
            empty_mutation = MutationReport(False, 0, 0, 0, 0.0, self.mutation_engine.threshold, reason="version_conflict")
            sandbox = SandboxResult("QUARANTINED", error="version_conflict")
            runtime = RuntimeVerificationReport(False, sandbox.status, [sandbox.error])
            cross = CrossVerificationReport(False, ["version_conflict"])
            approval = ApprovalDecision("REJECTED", False, ["active_skill_version_must_increase"], {})
            self.registry.persist_candidate(
                candidate,
                contract=contract,
                analysis=empty_analysis,
                mutation=empty_mutation,
                runtime=runtime,
                sandbox=sandbox,
                cross_verification=cross,
                approval=approval,
                status=candidate.status,
            )
            return LifecycleResult(candidate, empty_contract, empty_analysis, empty_mutation, sandbox, runtime, cross, approval)

        # Keep only the serializable candidate metadata.  Learned callables are
        # never retained as an execution authority in this process.
        self._candidates[candidate.skill_id] = SkillCandidate.from_dict(candidate.to_dict())
        self._contracts[candidate.skill_id] = contract
        candidate.status = "ANALYZING"
        self.registry.persist_candidate(candidate, contract=contract, status=candidate.status)
        contract_report = self.contract_validator.validate(candidate, contract)
        analysis = self.analyzer.analyze(candidate, contract)
        if not contract_report.passed or not analysis.passed:
            candidate.status = "REJECTED"
            mutation = MutationReport(False, 0, 0, 0, 0.0, self.mutation_engine.threshold, reason="analysis_or_contract_failed")
            sandbox = SandboxResult("QUARANTINED", error="not_executed_after_rejection")
            runtime = RuntimeVerificationReport(False, sandbox.status, [sandbox.error])
            cross = CrossVerificationReport(False, ["not_executed_after_rejection"])
            approval = ApprovalDecision("REJECTED", False, ["analysis_or_contract_failed"], {})
            self.registry.persist_candidate(
                candidate,
                contract=contract,
                analysis=analysis,
                mutation=mutation,
                runtime=runtime,
                sandbox=sandbox,
                cross_verification=cross,
                approval=approval,
                status=candidate.status,
            )
            return LifecycleResult(candidate, contract_report, analysis, mutation, sandbox, runtime, cross, approval)

        candidate.status = "TESTING"
        self.registry.persist_candidate(candidate, contract=contract, analysis=analysis, status=candidate.status)
        mutation = self.mutation_engine.run(candidate, mutation_test_runner)
        if not mutation.passed:
            candidate.status = "QUARANTINED"
            quarantine_reason = mutation.reason or "mutation_threshold_not_met"
            self._audit_skill_event(candidate, "SKILL_QUARANTINED", quarantine_reason)
            sandbox = SandboxResult("QUARANTINED", error=quarantine_reason)
            runtime = RuntimeVerificationReport(False, sandbox.status, [sandbox.error])
            cross = CrossVerificationReport(False, [quarantine_reason])
            approval = ApprovalDecision("QUARANTINED", False, [quarantine_reason], {"mutation_threshold": False})
            self.registry.persist_candidate(
                candidate,
                contract=contract,
                analysis=analysis,
                mutation=mutation,
                runtime=runtime,
                sandbox=sandbox,
                cross_verification=cross,
                approval=approval,
                status=candidate.status,
            )
            return LifecycleResult(candidate, contract_report, analysis, mutation, sandbox, runtime, cross, approval)

        if self.security is None:
            candidate.status = "QUARANTINED"
            sandbox = SandboxResult("BLOCKED", error="security_not_configured")
            runtime = RuntimeVerificationReport(False, sandbox.status, [sandbox.error])
            cross = CrossVerificationReport(False, ["security_not_configured"])
            approval = ApprovalDecision("QUARANTINED", False, ["security_not_configured"], {"security_controller": False})
            self.registry.persist_candidate(
                candidate,
                contract=contract,
                analysis=analysis,
                mutation=mutation,
                runtime=runtime,
                sandbox=sandbox,
                cross_verification=cross,
                approval=approval,
                status=candidate.status,
            )
            return LifecycleResult(candidate, contract_report, analysis, mutation, sandbox, runtime, cross, approval)

        security_action = self.security.action_for_skill(
            f"skill-submit:{candidate.name}:{candidate.version}",
            f"activate:{candidate.name}:{candidate.version}",
            candidate,
            inputs,
            goal=goal,
        )
        security_action.action_type = "skill_activation"
        activation_decision = self.security.authorize(
            security_action,
            approval_id=approval_id,
            boundary=True,
        )
        if not activation_decision.allowed:
            candidate.status = "QUARANTINED"
            sandbox = SandboxResult("BLOCKED", error=f"security_policy_blocked:{activation_decision.decision}")
            runtime = RuntimeVerificationReport(False, sandbox.status, [sandbox.error])
            cross = CrossVerificationReport(False, ["security_policy_blocked"])
            approval = ApprovalDecision(
                "QUARANTINED",
                False,
                ["security_policy_blocked", *activation_decision.reasons],
                {"security_authorization": False},
            )
            self.registry.persist_candidate(
                candidate,
                contract=contract,
                analysis=analysis,
                mutation=mutation,
                runtime=runtime,
                sandbox=sandbox,
                cross_verification=cross,
                approval=approval,
                status=candidate.status,
            )
            return LifecycleResult(candidate, contract_report, analysis, mutation, sandbox, runtime, cross, approval)

        execution_state: dict[str, Any] = {}

        def run_activation() -> SandboxResult:
            if type(self.sandbox) is not SkillSandbox:
                sandbox_result = SandboxResult(
                    "QUARANTINED",
                    error="untrusted_sandbox_provider_not_supported",
                )
                runtime_result = self.runtime_verifier.verify(candidate, contract, inputs, sandbox_result)
                cross_result = CrossVerificationReport(False, [sandbox_result.error])
                approval_result = ApprovalDecision(
                    "QUARANTINED",
                    False,
                    [sandbox_result.error],
                    {"external_sandbox": False},
                )
                execution_state.update(
                    sandbox=sandbox_result,
                    runtime=runtime_result,
                    cross=cross_result,
                    approval=approval_result,
                )
                return sandbox_result
            sandbox_result = self.sandbox.execute(candidate, contract, inputs)
            runtime_result = self.runtime_verifier.verify(candidate, contract, inputs, sandbox_result)
            cross_result = (
                self.cross_verifier.verify(candidate, sandbox_result.output)
                if runtime_result.passed
                else CrossVerificationReport(False, ["runtime_verification_failed"])
            )
            approval_result = self.approval_engine.decide(
                analysis,
                contract_report,
                mutation,
                sandbox_result,
                runtime_result,
                cross_result,
            )
            execution_state.update(
                sandbox=sandbox_result,
                runtime=runtime_result,
                cross=cross_result,
                approval=approval_result,
            )
            return sandbox_result

        def activation_result(value: SandboxResult) -> tuple[bool, dict[str, Any]]:
            approval_result = execution_state["approval"]
            return bool(approval_result.approved), {
                "sandbox_status": value.status,
                "verification": execution_state["runtime"].status,
            }

        activation_binding = self.security._bind_operation(
            security_action,
            run_activation,
            kind="skill_activation",
        )
        execution = self.security.execute_authorized(
            security_action,
            run_activation,
            authorization=activation_decision,
            result_validator=activation_result,
            operation_binding=activation_binding,
        )
        if execution.exception is not None:
            sandbox = SandboxResult("FAILED", error=f"{type(execution.exception).__name__}: {execution.exception}")
            runtime = RuntimeVerificationReport(False, sandbox.status, [sandbox.error])
            cross = CrossVerificationReport(False, ["activation_execution_failed"])
            approval = ApprovalDecision("REJECTED", False, ["activation_execution_failed"], {})
        else:
            sandbox = execution_state["sandbox"]
            runtime = execution_state["runtime"]
            cross = execution_state["cross"]
            approval = execution_state["approval"]
        candidate.status = approval.status
        self.registry.persist_candidate(
            candidate,
            contract=contract,
            analysis=analysis,
            mutation=mutation,
            runtime=runtime,
            sandbox=sandbox,
            cross_verification=cross,
            approval=approval,
            status=candidate.status,
        )
        if approval.approved:
            candidate.status = "ACTIVE"
            self.registry.activate(candidate, approval=approval)
            self._audit_skill_event(candidate, "SKILL_ACTIVATED", "Phase 8 approval and activation boundary passed")
        elif candidate.status == "QUARANTINED":
            self._audit_skill_event(candidate, "SKILL_QUARANTINED", "; ".join(approval.reasons))
        return LifecycleResult(candidate, contract_report, analysis, mutation, sandbox, runtime, cross, approval)

    def execute_active(
        self,
        name: str,
        inputs: dict[str, Any],
        *,
        task_id: str | None = None,
        action_id: str | None = None,
        goal: str = "",
        approval_id: str = "",
    ) -> ActiveExecutionResult:
        record = self.registry.active(name)
        if record is None:
            return ActiveExecutionResult("QUARANTINED", "", 0, SandboxResult("QUARANTINED", error="active_skill_not_found"), RuntimeVerificationReport(False, "QUARANTINED", ["active_skill_not_found"]), CrossVerificationReport(False, ["active_skill_not_found"]), {})
        skill_id = str(record["skill_id"])
        candidate = self._candidates.get(skill_id)
        if candidate is None:
            candidate = SkillCandidate.from_dict(record["candidate"])
        contract = self._contracts.get(skill_id) or SkillContract.from_candidate(candidate)
        if self.security is None:
            sandbox = SandboxResult("BLOCKED", error="security_not_configured")
            runtime = RuntimeVerificationReport(False, sandbox.status, [sandbox.error])
            cross = CrossVerificationReport(False, ["security_not_configured"])
            metrics = self.registry.record_runtime(
                candidate,
                success=False,
                verification_failed=True,
                contract_violation=False,
            )
            return ActiveExecutionResult("BLOCKED", candidate.skill_id, candidate.version, sandbox, runtime, cross, metrics)

        security_action = None
        security_action = self.security.action_for_skill(
            task_id or f"skill-runtime:{candidate.name}:{candidate.version}",
            action_id or f"invoke:{candidate.name}:{candidate.version}",
            candidate,
            inputs,
            goal=goal,
        )
        security_decision = self.security.authorize(
            security_action,
            approval_id=approval_id,
            boundary=True,
        )
        if not security_decision.allowed:
            sandbox = SandboxResult("BLOCKED", error=f"security_policy_blocked:{security_decision.decision}")
            runtime = self.runtime_verifier.verify(candidate, contract, inputs, sandbox)
            cross = CrossVerificationReport(False, ["security_policy_blocked"])
            metrics = self.registry.record_runtime(
                candidate,
                success=False,
                verification_failed=True,
                contract_violation=False,
            )
            return ActiveExecutionResult("BLOCKED", candidate.skill_id, candidate.version, sandbox, runtime, cross, metrics)
        execution_state: dict[str, Any] = {}

        def run_active() -> SandboxResult:
            if type(self.sandbox) is not SkillSandbox:
                sandbox_result = SandboxResult("QUARANTINED", error="untrusted_sandbox_provider_not_supported")
                runtime_result = self.runtime_verifier.verify(candidate, contract, inputs, sandbox_result)
                cross_result = CrossVerificationReport(False, [sandbox_result.error])
                execution_state.update(sandbox=sandbox_result, runtime=runtime_result, cross=cross_result)
                return sandbox_result
            sandbox_result = self.sandbox.execute(candidate, contract, inputs)
            runtime_result = self.runtime_verifier.verify(candidate, contract, inputs, sandbox_result)
            cross_result = (
                self.cross_verifier.verify(candidate, sandbox_result.output)
                if runtime_result.passed
                else CrossVerificationReport(False, ["runtime_verification_failed"])
            )
            execution_state.update(sandbox=sandbox_result, runtime=runtime_result, cross=cross_result)
            return sandbox_result

        def active_result(value: SandboxResult) -> tuple[bool, dict[str, Any]]:
            runtime_result = execution_state["runtime"]
            cross_result = execution_state["cross"]
            return bool(runtime_result.passed and cross_result.passed), {
                "sandbox_status": value.status,
                "verification": runtime_result.status,
            }

        active_binding = self.security._bind_operation(
            security_action,
            run_active,
            kind="skill_runtime",
        )
        execution = self.security.execute_authorized(
            security_action,
            run_active,
            authorization=security_decision,
            result_validator=active_result,
            operation_binding=active_binding,
        )
        if execution.exception is not None:
            sandbox = SandboxResult("FAILED", error=f"{type(execution.exception).__name__}: {execution.exception}")
            runtime = RuntimeVerificationReport(False, sandbox.status, [sandbox.error])
            cross = CrossVerificationReport(False, ["active_execution_failed"])
            success = False
        else:
            sandbox = execution_state["sandbox"]
            runtime = execution_state["runtime"]
            cross = execution_state["cross"]
            success = runtime.passed and cross.passed
        metrics = self.registry.record_runtime(
            candidate,
            success=success,
            verification_failed=not runtime.passed or not cross.passed,
            timeout=sandbox.status == "TIMEOUT",
            contract_violation=any("invalid" in item for item in runtime.errors),
        )
        rollback: dict[str, Any] | None = None
        experience_id = ""
        status = "VERIFIED" if success else sandbox.status if sandbox.status != "PASSED" else "FAILED"
        if not success and self.monitor.degraded(metrics):
            self.registry.mark_status(candidate.skill_id, "DEGRADED", event={"metrics": metrics})
            previous = self.registry.rollback(candidate.name, candidate.version, reason="runtime_degradation")
            if previous is not None:
                rollback = previous
                self.registry.record_rollback(candidate)
                self._audit_skill_event(candidate, "SKILL_ROLLED_BACK", "runtime_degradation")
                experience_id = self._record_rollback_experience(candidate, sandbox, runtime, metrics)
        return ActiveExecutionResult(status, candidate.skill_id, candidate.version, sandbox, runtime, cross, metrics, rollback, experience_id)

    def _record_rollback_experience(
        self,
        candidate: SkillCandidate,
        sandbox: SandboxResult,
        runtime: RuntimeVerificationReport,
        metrics: dict[str, Any],
    ) -> str:
        experience_id = f"skill-rollback-{candidate.skill_id}-{int(time.time() * 1000)}"
        action = ExperienceAction(
            action_type="skill_runtime",
            step_id=candidate.skill_id,
            tool_name=f"skill:{candidate.name}",
            input_fingerprint=hashlib.sha256(candidate.skill_id.encode("utf-8")).hexdigest()[:16],
            execution_status="failed",
            observation=sandbox.to_dict(),
            verification=runtime.to_dict(),
            diagnosis={"failure_type": "SKILL_DEGRADED", "metrics": metrics},
            recovery={"strategy": "ROLLBACK", "skill_version": candidate.version},
        )
        experience = ExperienceRecord(
            task_id=experience_id,
            goal=f"runtime monitoring for skill {candidate.name}",
            context={"task_type": "skill_runtime", "skill_id": candidate.skill_id, "skill_version": candidate.version},
            initial_plan={"skill": candidate.name, "version": candidate.version},
            actions=[action],
            final_outcome="ROLLED_BACK",
        )
        if self.experience_memory is not None:
            self.experience_memory.store(experience)
        if self.reflection_engine is not None:
            insights = self.reflection_engine.reflect_for_experience(experience)
            if self.learning_engine is not None:
                self.learning_engine.learn(insights)
        return experience_id

    def _audit_skill_event(self, candidate: SkillCandidate, event_type: str, reason: str) -> None:
        if self.security is None:
            return
        action = self.security.action_for_skill(
            "skill-lifecycle",
            f"{event_type}:{candidate.name}:{candidate.version}",
            candidate,
            {},
        )
        decision = self.security.policy_engine.evaluate(action)
        self.security.audit.record(
            event_type,
            action,
            decision=candidate.status,
            reason=reason,
            risk_level=decision.risk_level,
            policy_version=decision.policy_version,
        )


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _location(node: ast.AST) -> dict[str, int]:
    return {"line": int(getattr(node, "lineno", 0)), "column": int(getattr(node, "col_offset", 0))}
