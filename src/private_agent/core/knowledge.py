from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from private_agent.core.experience import sanitize_value


MAX_ID_LENGTH = 256
MAX_URI_LENGTH = 2048
MAX_REVISION_LENGTH = 256
MAX_TEXT_LENGTH = 4096
MAX_LOCATOR_LENGTH = 512
MAX_EXCERPT_LENGTH = 2048
MAX_SOURCE_CONTENT_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 4096
MAX_EVIDENCE_IDS = 32
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_QUERY_PARTS = ("token", "secret", "password", "credential", "authorization", "api_key", "apikey", "signature")

SOURCE_TYPES = frozenset({"web", "repository", "document", "local", "user"})
SOURCE_CLASSES = frozenset({"primary", "official", "secondary", "local_verified", "unverified"})
SOURCE_STATUSES = frozenset({"ACTIVE", "STALE", "RETIRED", "QUARANTINED"})
CLAIM_STATUSES = frozenset({"PROPOSED", "SUPPORTED", "CONTRADICTED", "STALE", "RETIRED", "QUARANTINED"})
VERIFICATION_STATUSES = frozenset({"unverified", "needs_review", "partially_verified", "verified", "contradicted"})
CONFLICT_STATUSES = frozenset({"OPEN", "RESOLVED", "DISMISSED"})


class KnowledgeValidationError(ValueError):
    """Raised when typed provenance or knowledge data is malformed."""


def sha256_hex(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _error(code: str) -> KnowledgeValidationError:
    return KnowledgeValidationError(code)


def _text(value: Any, field_name: str, *, max_length: int = MAX_TEXT_LENGTH, required: bool = True) -> str:
    if not isinstance(value, str):
        raise _error(f"{field_name}_must_be_string")
    normalized = re.sub(r"\s+", " ", value).strip()
    if required and not normalized:
        raise _error(f"{field_name}_required")
    if len(normalized) > max_length:
        raise _error(f"{field_name}_too_large")
    return normalized


def _identifier(value: Any, field_name: str, *, required: bool = True) -> str:
    return _text(value, field_name, max_length=MAX_ID_LENGTH, required=required)


def _safe_text(value: Any, field_name: str, *, max_length: int = MAX_TEXT_LENGTH, required: bool = True) -> str:
    return _text(sanitize_value(value), field_name, max_length=max_length, required=required)


def _timestamp(value: Any, field_name: str, *, required: bool = True) -> str:
    if value in (None, "") and not required:
        return ""
    if not isinstance(value, str):
        raise _error(f"{field_name}_must_be_timestamp")
    raw = value.strip()
    if not raw:
        if required:
            raise _error(f"{field_name}_required")
        return ""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _error(f"{field_name}_invalid") from exc
    if parsed.tzinfo is None:
        raise _error("timestamp_timezone_required")
    normalized = parsed.astimezone(timezone.utc)
    if normalized > datetime.now(timezone.utc):
        raise _error(f"{field_name}_future")
    return normalized.isoformat()


def _metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _error("metadata_must_be_object")
    sanitized = sanitize_value(value)
    try:
        encoded = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise _error("metadata_must_be_json") from exc
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise _error("metadata_too_large")
    return dict(sanitized)


def _metadata_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonicalize_source_uri(value: Any) -> str:
    uri = _text(value, "source_uri", max_length=MAX_URI_LENGTH)
    if any(ord(char) < 32 or char.isspace() for char in uri):
        raise _error("source_uri_invalid")
    parsed = urlsplit(uri)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https", "repo", "local"}:
        raise _error("source_uri_scheme_unsupported")
    if parsed.username is not None or parsed.password is not None:
        raise _error("source_uri_credentials_forbidden")
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise _error("source_uri_invalid") from exc
    if scheme in {"http", "https"}:
        if not hostname:
            raise _error("source_uri_host_required")
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
            lowered_key = key.lower()
            if any(part in lowered_key for part in _SENSITIVE_QUERY_PARTS):
                raise _error("source_uri_sensitive_query")
        normalized_host = hostname.lower()
        default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        netloc = normalized_host if port is None or default_port else f"{normalized_host}:{port}"
        path = parsed.path or "/"
        return urlunsplit((scheme, netloc, path, parsed.query, ""))
    if not hostname and not parsed.path:
        raise _error("source_uri_path_required")
    if any(part == ".." for part in parsed.path.split("/")):
        raise _error("source_uri_traversal")
    netloc = hostname.lower() if hostname else parsed.netloc
    return urlunsplit((scheme, netloc, parsed.path, parsed.query, ""))


def _hash(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise _error(f"{field_name}_invalid")
    return value


def _confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise _error("confidence_invalid")
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise _error("confidence_invalid") from exc
    if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        raise _error("confidence_invalid")
    return round(confidence, 6)


def _ordered_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class KnowledgeSource:
    source_id: str
    canonical_uri: str
    source_type: str
    source_class: str
    retrieved_at: str
    published_at: str = ""
    content_sha256: str = ""
    revision: str = ""
    status: str = "ACTIVE"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    content: str | bytes | None = field(default=None, repr=False, compare=False)
    content_bound: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(self, "canonical_uri", canonicalize_source_uri(self.canonical_uri))
        object.__setattr__(self, "source_type", _text(self.source_type, "source_type", max_length=64).lower())
        object.__setattr__(self, "source_class", _text(self.source_class, "source_class", max_length=64).lower())
        if self.source_type not in SOURCE_TYPES:
            raise _error("source_type_unsupported")
        if self.source_class not in SOURCE_CLASSES:
            raise _error("source_class_unsupported")
        object.__setattr__(self, "retrieved_at", _timestamp(self.retrieved_at, "retrieved_at"))
        object.__setattr__(self, "published_at", _timestamp(self.published_at, "published_at", required=False))
        object.__setattr__(self, "content_sha256", _hash(self.content_sha256, "source_hash"))
        object.__setattr__(self, "revision", _text(self.revision, "revision", max_length=MAX_REVISION_LENGTH, required=False))
        object.__setattr__(self, "status", _text(self.status, "source_status", max_length=32).upper())
        if self.status not in SOURCE_STATUSES:
            raise _error("source_status_invalid")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "created_at", _timestamp(self.created_at or _now(), "created_at"))
        if self.content is not None:
            if not isinstance(self.content, (str, bytes)):
                raise _error("source_content_type_invalid")
            content_bytes = self.content if isinstance(self.content, bytes) else self.content.encode("utf-8")
            if len(content_bytes) > MAX_SOURCE_CONTENT_BYTES:
                raise _error("source_content_too_large")
            if sha256_hex(content_bytes) != self.content_sha256:
                raise _error("source_content_hash_mismatch")
            object.__setattr__(self, "content_bound", True)

    @property
    def identity_key(self) -> tuple[str, str, str]:
        return self.canonical_uri, self.revision, self.content_sha256

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("content", None)
        payload.pop("content_bound", None)
        payload["metadata"] = json.loads(_metadata_json(self.metadata))
        return payload


@dataclass(frozen=True)
class KnowledgeClaim:
    claim_id: str
    domain: str
    subject: str
    predicate: str
    object_text: str
    knowledge_type: str
    verification_status: str = "unverified"
    status: str = "PROPOSED"
    confidence: float = 0.0
    observed_at: str = ""
    last_verified_at: str = ""
    valid_from: str = ""
    valid_until: str = ""
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    claim_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(self, "domain", _text(self.domain, "domain", max_length=256))
        object.__setattr__(self, "subject", _safe_text(self.subject, "subject"))
        object.__setattr__(self, "predicate", _safe_text(self.predicate, "predicate", max_length=512))
        object.__setattr__(self, "object_text", _safe_text(self.object_text, "object_text"))
        object.__setattr__(self, "knowledge_type", _text(self.knowledge_type, "knowledge_type", max_length=64))
        verification_status = _text(self.verification_status, "verification_status", max_length=32).lower()
        status = _text(self.status, "claim_status", max_length=32).upper()
        if verification_status not in VERIFICATION_STATUSES:
            raise _error("verification_status_invalid")
        if status not in CLAIM_STATUSES:
            raise _error("claim_status_invalid")
        object.__setattr__(self, "verification_status", verification_status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at or _now(), "observed_at"))
        object.__setattr__(self, "last_verified_at", _timestamp(self.last_verified_at, "last_verified_at", required=False))
        object.__setattr__(self, "valid_from", _timestamp(self.valid_from, "valid_from", required=False))
        object.__setattr__(self, "valid_until", _timestamp(self.valid_until, "valid_until", required=False))
        if self.valid_from and self.valid_until and self.valid_until <= self.valid_from:
            raise _error("claim_validity_range_invalid")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise _error("claim_version_invalid")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        created_at = _timestamp(self.created_at or self.observed_at, "created_at")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at or created_at, "updated_at"))
        fingerprint_payload = {
            "domain": self.domain,
            "subject": self.subject,
            "predicate": self.predicate,
            "object_text": self.object_text,
            "knowledge_type": self.knowledge_type,
        }
        calculated = sha256_hex(_ordered_json(fingerprint_payload))
        if self.claim_fingerprint and self.claim_fingerprint != calculated:
            raise _error("claim_fingerprint_mismatch")
        object.__setattr__(self, "claim_fingerprint", calculated)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = json.loads(_metadata_json(self.metadata))
        return payload


@dataclass(frozen=True)
class KnowledgeEvidence:
    evidence_id: str
    claim_id: str
    source_id: str
    task_id: str = ""
    step_id: str = ""
    locator: str = ""
    excerpt: str = ""
    verification_status: str = "unverified"
    observed_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    excerpt_digest: str = ""
    created_at: str = ""
    evidence_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id", required=False))
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id", required=False))
        if self.step_id and not self.task_id:
            raise _error("evidence_step_requires_task")
        object.__setattr__(self, "locator", _safe_text(self.locator, "locator", max_length=MAX_LOCATOR_LENGTH, required=False))
        sanitized_excerpt = _text(sanitize_value(self.excerpt), "evidence_excerpt", max_length=MAX_EXCERPT_LENGTH)
        object.__setattr__(self, "excerpt", sanitized_excerpt)
        verification_status = _text(self.verification_status, "verification_status", max_length=32).lower()
        if verification_status not in VERIFICATION_STATUSES:
            raise _error("verification_status_invalid")
        object.__setattr__(self, "verification_status", verification_status)
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at or _now(), "observed_at"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        digest = sha256_hex(sanitized_excerpt)
        if self.excerpt_digest and self.excerpt_digest != digest:
            raise _error("evidence_digest_mismatch")
        object.__setattr__(self, "excerpt_digest", digest)
        object.__setattr__(self, "created_at", _timestamp(self.created_at or self.observed_at, "created_at"))
        fingerprint_payload = {
            "claim_id": self.claim_id,
            "source_id": self.source_id,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "locator": self.locator,
            "excerpt_digest": digest,
        }
        calculated = sha256_hex(_ordered_json(fingerprint_payload))
        if self.evidence_fingerprint and self.evidence_fingerprint != calculated:
            raise _error("evidence_fingerprint_mismatch")
        object.__setattr__(self, "evidence_fingerprint", calculated)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = json.loads(_metadata_json(self.metadata))
        return payload


@dataclass(frozen=True)
class KnowledgeConflict:
    conflict_id: str
    claim_id: str
    conflicting_claim_id: str
    conflict_type: str
    status: str = "OPEN"
    detected_at: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    resolution_note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    conflict_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "conflict_id", _identifier(self.conflict_id, "conflict_id"))
        claim_id = _identifier(self.claim_id, "claim_id")
        conflicting_claim_id = _identifier(self.conflicting_claim_id, "conflicting_claim_id")
        if claim_id == conflicting_claim_id:
            raise _error("conflict_self_reference")
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "conflicting_claim_id", conflicting_claim_id)
        object.__setattr__(self, "conflict_type", _text(self.conflict_type, "conflict_type", max_length=64))
        status = _text(self.status, "conflict_status", max_length=32).upper()
        if status not in CONFLICT_STATUSES:
            raise _error("conflict_status_invalid")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "detected_at", _timestamp(self.detected_at or _now(), "detected_at"))
        if not isinstance(self.evidence_ids, list) or len(self.evidence_ids) > MAX_EVIDENCE_IDS:
            raise _error("conflict_evidence_ids_invalid")
        evidence_ids = [_identifier(item, "evidence_id") for item in self.evidence_ids]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise _error("conflict_evidence_ids_duplicate")
        object.__setattr__(self, "evidence_ids", sorted(evidence_ids))
        object.__setattr__(self, "resolution_note", _safe_text(self.resolution_note, "resolution_note", max_length=1024, required=False))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "created_at", _timestamp(self.created_at or self.detected_at, "created_at"))
        fingerprint_payload = {
            "claim_ids": sorted((claim_id, conflicting_claim_id)),
            "conflict_type": self.conflict_type,
        }
        calculated = sha256_hex(_ordered_json(fingerprint_payload))
        if self.conflict_fingerprint and self.conflict_fingerprint != calculated:
            raise _error("conflict_fingerprint_mismatch")
        object.__setattr__(self, "conflict_fingerprint", calculated)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = json.loads(_metadata_json(self.metadata))
        return payload


__all__ = [
    "CLAIM_STATUSES",
    "CONFLICT_STATUSES",
    "KnowledgeClaim",
    "KnowledgeConflict",
    "KnowledgeEvidence",
    "KnowledgeSource",
    "KnowledgeValidationError",
    "SOURCE_CLASSES",
    "SOURCE_STATUSES",
    "SOURCE_TYPES",
    "VERIFICATION_STATUSES",
    "canonicalize_source_uri",
    "sha256_hex",
]
