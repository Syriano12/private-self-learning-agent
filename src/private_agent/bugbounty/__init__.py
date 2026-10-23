"""Authorized, bounded Bug Bounty research components."""

from .engines import DeterministicPlanner, IDORBOLAEngine, PassiveRecon, StaticSecurityAnalyzer
from .fixtures import DeterministicFixtureApp, ScopeEnforcedFixtureClient, ScopeViolation, benchmark_fixture_variants
from .models import AuthorizationObservation, BugBountyRun, Evidence, Finding, FindingState, PolicyDecision, ScopeDefinition, ScopeDecision, Severity
from .workflow import BugBountyAgent, BugBountyRepository, evaluate_local_benchmark, markdown_report
from .validation import build_independent_corpus, evaluate_independent_corpus, validate_confirmed_evidence
from .cross_application import build_cross_application_corpus, evaluate_cross_application_corpus
from .scanners import ActiveScanResult, HttpxAdapter, NucleiAdapter, ScanStatus, ScannerLimits, TrustedTemplateSet
from .trusted_tools import ToolVerificationResult, ToolVerificationStatus, TrustedToolInstallSpec, load_tool_spec_from_environment, verify_tool_installation

__all__ = [
    "BugBountyAgent",
    "BugBountyRepository",
    "BugBountyRun",
    "ActiveScanResult",
    "AuthorizationObservation",
    "DeterministicFixtureApp",
    "DeterministicPlanner",
    "Evidence",
    "Finding",
    "FindingState",
    "IDORBOLAEngine",
    "HttpxAdapter",
    "NucleiAdapter",
    "PassiveRecon",
    "PolicyDecision",
    "ScopeDefinition",
    "ScopeDecision",
    "ScopeEnforcedFixtureClient",
    "ScopeViolation",
    "ScanStatus",
    "ScannerLimits",
    "Severity",
    "StaticSecurityAnalyzer",
    "TrustedTemplateSet",
    "ToolVerificationResult",
    "ToolVerificationStatus",
    "TrustedToolInstallSpec",
    "benchmark_fixture_variants",
    "build_independent_corpus",
    "build_cross_application_corpus",
    "evaluate_local_benchmark",
    "evaluate_independent_corpus",
    "evaluate_cross_application_corpus",
    "markdown_report",
    "validate_confirmed_evidence",
    "load_tool_spec_from_environment",
    "verify_tool_installation",
]
