"""Immutable identities and source-bound evidence references for repository reads.

Digests identify the bytes observed by the trusted repository reader. They do not
prove that a static claim is correct, that a repository is authorized, or that a
runtime behavior is reachable.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from private_agent.tools.github_access import RepositoryAccessError, parse_repository, validate_repository_path

_SCHEMA_VERSION = 1
_SNAPSHOT_FIELDS = frozenset({
    "schema_version", "repository_identity", "repository", "source", "commit", "source_revision",
    "snapshot_id", "content_digest", "tree_digest", "files", "tree_manifest", "excluded_files",
    "tree_truncated", "exclusions_truncated", "analysis_truncated",
})
_EVIDENCE_REQUIRED_FIELDS = frozenset({
    "repository_identity", "snapshot_id", "repository", "revision", "source_revision", "commit",
    "path", "file_path", "sha256", "file_sha256", "line_start", "line_end", "column_start",
    "column_end", "symbol_id", "parser_id", "parser_version", "rule_id", "rule_version",
    "evidence_type", "truncated", "uncertainty",
})
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _bounded_text(value: Any, name: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or any(ord(char) < 0x20 for char in value):
        raise ValueError(f"invalid_{name}")
    return value


def _file_record(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid_snapshot_file_record")
    path = validate_repository_path(value.get("path"))
    sha256 = value.get("sha256")
    size = value.get("size_bytes")
    lines = value.get("line_count")
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise ValueError("invalid_snapshot_file_digest")
    if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= 65536:
        raise ValueError("invalid_snapshot_file_size")
    if not isinstance(lines, int) or isinstance(lines, bool) or not 0 <= lines <= size:
        raise ValueError("invalid_snapshot_line_count")
    return {"path": path, "sha256": sha256.lower(), "size_bytes": size, "line_count": lines}


def _tree_record(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid_tree_manifest_record")
    path = validate_repository_path(value.get("path"))
    size = value.get("size_bytes")
    object_sha = value.get("object_sha", "")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("invalid_tree_file_size")
    if not isinstance(object_sha, str) or len(object_sha) > 64:
        raise ValueError("invalid_tree_object_digest")
    if object_sha and not re.fullmatch(r"[0-9a-fA-F]{40,64}", object_sha):
        raise ValueError("invalid_tree_object_digest")
    return {"path": path, "size_bytes": size, "object_sha": object_sha.lower()}


def _exclusion_record(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid_exclusion_record")
    path = value.get("path")
    reason = value.get("reason")
    if not isinstance(path, str) or len(path) > 1024 or any(ord(char) < 0x20 for char in path):
        raise ValueError("invalid_excluded_path")
    if path and path not in {".", "..", "[TREE_REMAINDER]"}:
        # Paths that were rejected for traversal are recorded only as a reason,
        # never echoed as an actionable filesystem path.
        components = path.split("/")
        safe_git_metadata_path = (
            ".git" in components and not path.startswith("/") and "\\" not in path
            and all(part not in {"", ".", ".."} for part in components)
        )
        if not safe_git_metadata_path:
            try:
                validate_repository_path(path)
            except RepositoryAccessError:
                path = "[UNSAFE_PATH]"
    reason = _bounded_text(reason, "exclusion_reason", 128)
    return {"path": path, "reason": reason}


@dataclass(frozen=True)
class RepositorySnapshot:
    repository_identity: str
    repository: str
    source: str
    commit: str
    source_revision: str
    snapshot_id: str
    content_digest: str
    tree_digest: str
    files: tuple[Mapping[str, Any], ...]
    tree_manifest: tuple[Mapping[str, Any], ...]
    excluded_files: tuple[Mapping[str, str], ...]
    tree_truncated: bool
    exclusions_truncated: bool
    analysis_truncated: bool

    @classmethod
    def create(
        cls,
        *,
        repository: str,
        source: str,
        commit: str = "",
        files: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        tree_manifest: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        excluded_files: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        tree_truncated: bool = False,
        exclusions_truncated: bool = False,
        analysis_truncated: bool = False,
    ) -> "RepositorySnapshot":
        repository = _bounded_text(repository, "repository", 256)
        try:
            canonical_repository = parse_repository(repository)
        except RepositoryAccessError:
            raise ValueError("invalid_snapshot_repository") from None
        if repository.lower() != canonical_repository.lower():
            raise ValueError("invalid_snapshot_repository")
        repository = canonical_repository.lower()
        source = _bounded_text(source, "source", 64)
        if source not in {"local_fixture", "github_api"}:
            raise ValueError("unsupported_snapshot_source")
        if not isinstance(commit, str) or len(commit) > 128 or (commit and not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit)):
            raise ValueError("invalid_snapshot_commit")
        if any(not isinstance(flag, bool) for flag in (tree_truncated, exclusions_truncated, analysis_truncated)):
            raise ValueError("invalid_snapshot_truncation_state")
        if not isinstance(files, (list, tuple)) or len(files) > 500:
            raise ValueError("snapshot_file_limit_exceeded")
        if not isinstance(tree_manifest, (list, tuple)) or len(tree_manifest) > 500:
            raise ValueError("snapshot_tree_limit_exceeded")
        if not isinstance(excluded_files, (list, tuple)):
            raise ValueError("invalid_snapshot_exclusions")

        file_records = [_file_record(value) for value in files]
        file_records.sort(key=lambda item: item["path"])
        if len({item["path"] for item in file_records}) != len(file_records):
            raise ValueError("duplicate_snapshot_file")
        tree_records = [_tree_record(value) for value in tree_manifest]
        tree_records.sort(key=lambda item: item["path"])
        if len({item["path"] for item in tree_records}) != len(tree_records):
            raise ValueError("duplicate_tree_manifest_path")
        exclusion_records = [_exclusion_record(value) for value in excluded_files[:200]]
        exclusion_records.sort(key=lambda item: (item["path"], item["reason"]))
        if len({(item["path"], item["reason"]) for item in exclusion_records}) != len(exclusion_records):
            exclusion_records = list({(item["path"], item["reason"]): item for item in exclusion_records}.values())
            exclusion_records.sort(key=lambda item: (item["path"], item["reason"]))

        exclusions_truncated = exclusions_truncated or len(excluded_files) > 200
        included_paths = {item["path"] for item in file_records}
        tree_paths = {item["path"] for item in tree_records}
        exclusion_paths = {item["path"] for item in exclusion_records}
        analysis_truncated = bool(
            analysis_truncated or tree_truncated or exclusions_truncated
            or any(path not in included_paths and path not in exclusion_paths for path in tree_paths)
        )
        repository_identity = f"{source}:{repository.lower()}"
        content_digest = _digest(file_records)
        tree_digest = _digest({"entries": tree_records, "truncated": tree_truncated})
        if commit:
            source_revision = f"commit:{commit.lower()}"
        else:
            source_revision = "working-tree"
        identity_payload = {
            "schema_version": _SCHEMA_VERSION,
            "repository_identity": repository_identity,
            "repository": repository,
            "source": source,
            "commit": commit.lower(),
            "source_revision": source_revision,
            "content_digest": content_digest,
            "tree_digest": tree_digest,
            "excluded_files": exclusion_records,
            "exclusions_truncated": exclusions_truncated,
            "analysis_truncated": analysis_truncated,
        }
        snapshot_id = "sha256:" + _digest(identity_payload)
        return cls(
            repository_identity=repository_identity,
            repository=repository,
            source=source,
            commit=commit.lower(),
            source_revision=source_revision,
            snapshot_id=snapshot_id,
            content_digest=content_digest,
            tree_digest=tree_digest,
            files=tuple(MappingProxyType(dict(item)) for item in file_records),
            tree_manifest=tuple(MappingProxyType(dict(item)) for item in tree_records),
            excluded_files=tuple(MappingProxyType(dict(item)) for item in exclusion_records),
            tree_truncated=tree_truncated,
            exclusions_truncated=exclusions_truncated,
            analysis_truncated=analysis_truncated,
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": _SCHEMA_VERSION,
            "repository_identity": self.repository_identity,
            "repository": self.repository,
            "source": self.source,
            "commit": self.commit,
            "source_revision": self.source_revision,
            "snapshot_id": self.snapshot_id,
            "content_digest": self.content_digest,
            "tree_digest": self.tree_digest,
            "files": [dict(item) for item in self.files],
            "tree_manifest": [dict(item) for item in self.tree_manifest],
            "excluded_files": [dict(item) for item in self.excluded_files],
            "tree_truncated": self.tree_truncated,
            "exclusions_truncated": self.exclusions_truncated,
            "analysis_truncated": self.analysis_truncated,
        }
        self.validate()
        return value

    def validate(self) -> None:
        rebuilt = RepositorySnapshot.create(
            repository=self.repository,
            source=self.source,
            commit=self.commit,
            files=self.files,
            tree_manifest=self.tree_manifest,
            excluded_files=self.excluded_files,
            tree_truncated=self.tree_truncated,
            exclusions_truncated=self.exclusions_truncated,
            analysis_truncated=self.analysis_truncated,
        )
        if (
            self.repository_identity != rebuilt.repository_identity
            or self.source_revision != rebuilt.source_revision
            or self.snapshot_id != rebuilt.snapshot_id
            or self.content_digest != rebuilt.content_digest
            or self.tree_digest != rebuilt.tree_digest
        ):
            raise ValueError("snapshot_digest_mismatch")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepositorySnapshot":
        if (
            not isinstance(value, Mapping)
            or not isinstance(value.get("schema_version"), int)
            or isinstance(value.get("schema_version"), bool)
            or value.get("schema_version") != _SCHEMA_VERSION
        ):
            raise ValueError("unsupported_snapshot_schema")
        if set(value) != _SNAPSHOT_FIELDS:
            raise ValueError("invalid_snapshot_schema")
        try:
            snapshot = cls.create(
                repository=value["repository"],
                source=value["source"],
                commit=value.get("commit", ""),
                files=value.get("files", []),
                tree_manifest=value.get("tree_manifest", []),
                excluded_files=value.get("excluded_files", []),
                tree_truncated=value.get("tree_truncated", False),
                exclusions_truncated=value.get("exclusions_truncated", False),
                analysis_truncated=value.get("analysis_truncated", False),
            )
        except (KeyError, TypeError, AttributeError, ValueError):
            raise ValueError("invalid_snapshot") from None
        for field_name in ("repository_identity", "source_revision", "snapshot_id", "content_digest", "tree_digest"):
            if value.get(field_name) != getattr(snapshot, field_name):
                raise ValueError("snapshot_digest_mismatch")
        return snapshot

    def file(self, path: str) -> dict[str, Any] | None:
        return next((dict(item) for item in self.files if item["path"] == path), None)

    @classmethod
    def merge(cls, snapshots: list["RepositorySnapshot"]) -> "RepositorySnapshot":
        """Combine bounded inspections only when they name one consistent source view."""
        if not snapshots:
            raise ValueError("snapshot_required")
        for snapshot in snapshots:
            snapshot.validate()
        base = snapshots[0]
        populated_trees = [snapshot for snapshot in snapshots if snapshot.tree_manifest]
        if any(
            (item.repository_identity, item.repository, item.source, item.commit, item.source_revision)
            != (base.repository_identity, base.repository, base.source, base.commit, base.source_revision)
            for item in snapshots
        ):
            raise ValueError("snapshot_source_mismatch")
        if any(item.tree_digest != populated_trees[0].tree_digest for item in populated_trees[1:]):
            raise ValueError("source_changed_during_analysis")
        files: dict[str, dict[str, Any]] = {}
        exclusions: dict[tuple[str, str], dict[str, str]] = {}
        for snapshot in snapshots:
            for record in snapshot.files:
                previous = files.get(record["path"])
                if previous is not None and previous != record:
                    raise ValueError("source_changed_during_analysis")
                files[record["path"]] = dict(record)
            for record in snapshot.excluded_files:
                exclusions[(record["path"], record["reason"])] = dict(record)
        for path in files:
            exclusions.pop((path, "not_selected_for_analysis"), None)
        tree = populated_trees[0].tree_manifest if populated_trees else base.tree_manifest
        return cls.create(
            repository=base.repository,
            source=base.source,
            commit=base.commit,
            files=list(files.values()),
            tree_manifest=list(tree),
            excluded_files=list(exclusions.values()),
            tree_truncated=any(item.tree_truncated for item in snapshots),
            exclusions_truncated=any(item.exclusions_truncated for item in snapshots),
            analysis_truncated=any(item.analysis_truncated for item in snapshots),
        )


@dataclass(frozen=True)
class CodeEvidenceRef:
    repository_identity: str
    snapshot_id: str
    repository: str
    revision: str
    source_revision: str
    commit: str
    path: str
    file_path: str
    sha256: str
    file_sha256: str
    line_start: int
    line_end: int
    column_start: int | None
    column_end: int | None
    symbol_id: str | None
    parser_id: str
    parser_version: str
    rule_id: str
    rule_version: str
    evidence_type: str
    truncated: bool
    uncertainty: tuple[str, ...]

    @classmethod
    def create(
        cls,
        snapshot: RepositorySnapshot,
        *,
        path: str,
        line_start: int,
        line_end: int,
        column_start: int | None = None,
        column_end: int | None = None,
        symbol_id: str | None = None,
        parser_id: str = "none",
        parser_version: str = "0",
        rule_id: str = "repository.read",
        rule_version: str = "1",
        evidence_type: str = "source_observation",
        truncated: bool = False,
        uncertainty: tuple[str, ...] | list[str] = (),
    ) -> "CodeEvidenceRef":
        snapshot.validate()
        path = validate_repository_path(path)
        record = snapshot.file(path)
        if record is None:
            raise ValueError("evidence_file_not_in_snapshot")
        if not isinstance(line_start, int) or isinstance(line_start, bool) or not isinstance(line_end, int) or isinstance(line_end, bool):
            raise ValueError("invalid_evidence_line_range")
        if line_start < 1 or line_end < line_start or line_end > record["line_count"]:
            raise ValueError("evidence_range_outside_snapshot")
        if (column_start is None) != (column_end is None):
            raise ValueError("invalid_evidence_column_range")
        if column_start is not None:
            if not isinstance(column_start, int) or isinstance(column_start, bool) or not isinstance(column_end, int) or isinstance(column_end, bool):
                raise ValueError("invalid_evidence_column_range")
            if column_start < 1 or column_end < 1:
                raise ValueError("invalid_evidence_column_range")
            if line_start == line_end and column_end < column_start:
                raise ValueError("invalid_evidence_column_range")
            if column_end > 65536:
                raise ValueError("invalid_evidence_column_range")
        if symbol_id is not None:
            symbol_id = _bounded_text(symbol_id, "evidence_symbol", 512)
        parser_id = _bounded_text(parser_id, "evidence_parser_id", 128)
        parser_version = _bounded_text(parser_version, "evidence_parser_version", 64)
        rule_id = _bounded_text(rule_id, "evidence_rule_id", 128)
        rule_version = _bounded_text(rule_version, "evidence_rule_version", 64)
        evidence_type = _bounded_text(evidence_type, "evidence_type", 64)
        if not isinstance(truncated, bool):
            raise ValueError("invalid_evidence_truncation_state")
        clean_uncertainty = tuple(dict.fromkeys(_bounded_text(item, "evidence_uncertainty", 128) for item in uncertainty[:12]))
        if column_start is None:
            clean_uncertainty = tuple(dict.fromkeys((*clean_uncertainty, "column_range_unavailable")))
        return cls(
            repository_identity=snapshot.repository_identity,
            snapshot_id=snapshot.snapshot_id,
            repository=snapshot.repository,
            revision=snapshot.commit,
            source_revision=snapshot.source_revision,
            commit=snapshot.commit,
            path=path,
            file_path=path,
            sha256=record["sha256"],
            file_sha256=record["sha256"],
            line_start=line_start,
            line_end=line_end,
            column_start=column_start,
            column_end=column_end,
            symbol_id=symbol_id,
            parser_id=parser_id,
            parser_version=parser_version,
            rule_id=rule_id,
            rule_version=rule_version,
            evidence_type=evidence_type,
            truncated=truncated,
            uncertainty=clean_uncertainty,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_identity": self.repository_identity,
            "snapshot_id": self.snapshot_id,
            "repository": self.repository,
            "revision": self.revision,
            "source_revision": self.source_revision,
            "commit": self.commit,
            "path": self.path,
            "file_path": self.file_path,
            "sha256": self.sha256,
            "file_sha256": self.file_sha256,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "column_start": self.column_start,
            "column_end": self.column_end,
            "symbol_id": self.symbol_id,
            "parser_id": self.parser_id,
            "parser_version": self.parser_version,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "evidence_type": self.evidence_type,
            "truncated": self.truncated,
            "uncertainty": list(self.uncertainty),
        }

    def validate_against(self, snapshot: RepositorySnapshot) -> None:
        snapshot.validate()
        try:
            expected = CodeEvidenceRef.create(
                snapshot,
                path=self.path,
                line_start=self.line_start,
                line_end=self.line_end,
                column_start=self.column_start,
                column_end=self.column_end,
                symbol_id=self.symbol_id,
                parser_id=self.parser_id,
                parser_version=self.parser_version,
                rule_id=self.rule_id,
                rule_version=self.rule_version,
                evidence_type=self.evidence_type,
                truncated=self.truncated,
                uncertainty=self.uncertainty,
            )
        except (TypeError, ValueError):
            raise ValueError("evidence_snapshot_mismatch") from None
        if self != expected:
            raise ValueError("evidence_snapshot_mismatch")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], snapshot: Mapping[str, Any] | RepositorySnapshot) -> "CodeEvidenceRef":
        if not isinstance(value, Mapping) or not _EVIDENCE_REQUIRED_FIELDS.issubset(value):
            raise ValueError("invalid_code_evidence_ref")
        if set(value) - (_EVIDENCE_REQUIRED_FIELDS | {"excerpt"}):
            raise ValueError("invalid_code_evidence_ref")
        if "excerpt" in value and (not isinstance(value["excerpt"], str) or len(value["excerpt"]) > 500):
            raise ValueError("invalid_code_evidence_ref")
        if not isinstance(value.get("uncertainty"), (list, tuple)):
            raise ValueError("invalid_code_evidence_ref")
        if not isinstance(snapshot, RepositorySnapshot):
            snapshot = RepositorySnapshot.from_dict(snapshot)
        snapshot.validate()
        try:
            ref = cls(
                repository_identity=value["repository_identity"],
                snapshot_id=value["snapshot_id"],
                repository=value["repository"],
                revision=value["revision"],
                source_revision=value["source_revision"],
                commit=value.get("commit", ""),
                path=validate_repository_path(value["path"]),
                file_path=validate_repository_path(value["file_path"]),
                sha256=value["sha256"],
                file_sha256=value["file_sha256"],
                line_start=value["line_start"],
                line_end=value["line_end"],
                column_start=value.get("column_start"),
                column_end=value.get("column_end"),
                symbol_id=value.get("symbol_id"),
                parser_id=value["parser_id"],
                parser_version=value["parser_version"],
                rule_id=value["rule_id"],
                rule_version=value["rule_version"],
                evidence_type=value["evidence_type"],
                truncated=value["truncated"],
                uncertainty=tuple(value["uncertainty"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid_code_evidence_ref") from exc
        if ref.path != ref.file_path or ref.file_sha256 != ref.sha256:
            raise ValueError("invalid_code_evidence_ref")
        ref.validate_against(snapshot)
        return ref
