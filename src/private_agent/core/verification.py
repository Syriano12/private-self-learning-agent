from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from private_agent.core.observation import Observation


VERIFICATION_STATUSES = {"VERIFIED", "FAILED", "INSUFFICIENT", "BLOCKED"}


@dataclass
class VerificationResult:
    status: str
    reason: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    verifier_type: str = "generic"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VERIFICATION_STATUSES:
            raise ValueError(f"invalid_verification_status:{self.status}")

    @property
    def verified(self) -> bool:
        return self.status == "VERIFIED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "verified": self.verified,
            "reason": self.reason,
            "evidence": self.evidence,
            "verifier_type": self.verifier_type,
            "details": self.details,
            # Kept for compatibility with the Phase 1/2 result shape.
            "criteria": self.details.get("criteria", {}),
            "errors": self.details.get("errors", []),
        }


class Verifier(Protocol):
    name: str

    def verify(self, observation: Observation, criteria: dict[str, Any]) -> VerificationResult:
        ...


class GenericVerifier:
    name = "generic"

    def verify(self, observation: Observation, criteria: dict[str, Any]) -> VerificationResult:
        if observation.execution_status == "blocked":
            return VerificationResult(
                "BLOCKED",
                "Tool execution was blocked before verification",
                verifier_type=self.name,
                details={"execution_status": observation.execution_status},
            )
        if observation.execution_status != "success":
            return VerificationResult(
                "FAILED",
                "Tool execution did not succeed",
                verifier_type=self.name,
                details={"execution_status": observation.execution_status, "error": observation.observed_error},
            )
        if not criteria:
            return VerificationResult(
                "INSUFFICIENT",
                "No success criteria were provided",
                verifier_type=self.name,
                details={"code": "missing_success_criteria"},
            )
        if observation.observed_output is None or observation.observed_output == {} or observation.observed_output == []:
            return VerificationResult(
                "INSUFFICIENT",
                "Tool execution succeeded but produced no observable output",
                verifier_type=self.name,
                details={"code": "missing_output"},
            )

        required_fields = criteria.get("required_fields", [])
        missing = [field for field in required_fields if not isinstance(observation.observed_output, dict) or field not in observation.observed_output]
        if missing:
            return VerificationResult(
                "FAILED",
                "Required output fields are missing",
                verifier_type=self.name,
                details={"code": "missing_required_output", "missing_fields": missing},
            )
        return VerificationResult(
            "VERIFIED",
            "Observable output satisfies the supplied generic criteria",
            evidence=[{"type": "output_present", "value": True}],
            verifier_type=self.name,
            details={"criteria": criteria},
        )


class WebResearchVerifier:
    name = "web_research"

    def verify(self, observation: Observation, criteria: dict[str, Any]) -> VerificationResult:
        if observation.execution_status == "blocked":
            return VerificationResult(
                "BLOCKED",
                "Web research execution was blocked",
                verifier_type=self.name,
                details={"execution_status": observation.execution_status},
            )
        if observation.execution_status != "success":
            return VerificationResult(
                "FAILED",
                "Web research execution failed",
                verifier_type=self.name,
                details={"execution_status": observation.execution_status, "error": observation.observed_error},
            )
        if not isinstance(observation.observed_output, dict):
            return VerificationResult(
                "FAILED",
                "Web research returned a non-object output",
                verifier_type=self.name,
                details={"code": "invalid_output_type"},
            )

        output = observation.observed_output
        sources = output.get("sources")
        errors = output.get("errors", [])
        if not isinstance(sources, list):
            return VerificationResult(
                "FAILED",
                "Web research output does not contain a sources list",
                verifier_type=self.name,
                details={"code": "sources_missing_or_invalid", "errors": errors},
            )
        if errors:
            return VerificationResult(
                "FAILED",
                "Web research reported tool or fetch errors",
                evidence=[{"type": "errors", "value": errors}],
                verifier_type=self.name,
                details={"code": "research_errors", "errors": errors},
            )
        if not sources:
            return VerificationResult(
                "INSUFFICIENT",
                "Web research produced no sources",
                verifier_type=self.name,
                details={"code": "no_sources", "criteria": criteria},
            )

        minimum_sources = int(criteria.get("minimum_sources", 2))
        require_non_empty_content = criteria.get("require_non_empty_content", True)
        valid_sources: list[dict[str, Any]] = []
        invalid_sources: list[dict[str, Any]] = []
        for source in sources:
            if not isinstance(source, dict):
                invalid_sources.append({"reason": "source_not_object"})
                continue
            text = source.get("text")
            url = source.get("url")
            if not url:
                invalid_sources.append({"reason": "url_missing", "source": source})
                continue
            if require_non_empty_content and not isinstance(text, str):
                invalid_sources.append({"reason": "text_missing", "url": url})
                continue
            if require_non_empty_content and not text.strip():
                invalid_sources.append({"reason": "text_empty", "url": url})
                continue
            valid_sources.append({"url": url, "title": source.get("title", ""), "content_length": len(text or "")})

        evidence = [
            {"type": "source_count", "value": len(sources)},
            {"type": "valid_source_count", "value": len(valid_sources)},
            {"type": "invalid_source_count", "value": len(invalid_sources)},
        ]
        if len(valid_sources) < minimum_sources:
            return VerificationResult(
                "INSUFFICIENT",
                "Web research did not meet the minimum valid source criterion",
                evidence=evidence,
                verifier_type=self.name,
                details={
                    "code": "insufficient_sources",
                    "minimum_sources": minimum_sources,
                    "criteria": criteria,
                    "invalid_sources": invalid_sources,
                },
            )

        required_terms = criteria.get("required_terms", [])
        searchable = " ".join(str(source.get("text", "")) for source in sources).lower()
        missing_terms = [term for term in required_terms if str(term).lower() not in searchable]
        if missing_terms:
            return VerificationResult(
                "FAILED",
                "Research sources do not contain all required terms",
                evidence=evidence,
                verifier_type=self.name,
                details={"code": "required_terms_missing", "missing_terms": missing_terms, "criteria": criteria},
            )

        return VerificationResult(
            "VERIFIED",
            "Web research produced sufficient readable evidence",
            evidence=evidence,
            verifier_type=self.name,
            details={
                "criteria": {
                    "minimum_sources": minimum_sources,
                    "require_non_empty_content": require_non_empty_content,
                    "required_terms": required_terms,
                },
                "valid_sources": valid_sources,
            },
        )


class VerifierRegistry:
    """Resolve verifiers by contract name; the executor remains verifier-agnostic."""

    def __init__(self, verifiers: list[Verifier] | None = None) -> None:
        self._verifiers: dict[str, Verifier] = {}
        for verifier in verifiers or [GenericVerifier(), WebResearchVerifier()]:
            self.register(verifier)

    def register(self, verifier: Verifier) -> None:
        if not isinstance(getattr(verifier, "name", None), str) or not verifier.name.strip():
            raise ValueError("verifier_name_required")
        if not callable(getattr(verifier, "verify", None)):
            raise ValueError("verifier_method_required")
        self._verifiers[verifier.name] = verifier

    def get(self, name: str) -> Verifier:
        return self._verifiers[name]

    def available(self) -> list[str]:
        return sorted(self._verifiers)

    def verify(
        self,
        name: str,
        observation: Observation,
        criteria: dict[str, Any] | None = None,
    ) -> VerificationResult:
        try:
            verifier = self.get(name)
        except KeyError:
            return VerificationResult(
                "BLOCKED",
                f"Verifier is not registered: {name}",
                verifier_type=name,
                details={"code": "unknown_verifier"},
            )
        try:
            result = verifier.verify(observation, criteria or {})
        except Exception as exc:
            return VerificationResult(
                "FAILED",
                "Verifier raised an exception",
                verifier_type=name,
                details={"code": "verification_exception", "exception_type": type(exc).__name__},
            )
        if not isinstance(result, VerificationResult):
            return VerificationResult(
                "FAILED",
                "Verifier returned a malformed result",
                verifier_type=name,
                details={"code": "malformed_verification_result"},
            )
        return result
