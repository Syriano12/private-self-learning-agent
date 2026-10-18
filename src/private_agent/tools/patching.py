"""Bounded, reviewable repository patch lifecycle.

This module deliberately operates only on operator-configured local fixtures. It
never invokes a repository command, imports repository modules, runs tests, or
writes to the configured source tree. Preparation validates a unified diff and
records a durable patch session; application writes only to an agent-owned
isolated workspace; rollback restores that workspace only when no concurrent
change is detected.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl; the thread lock still applies.
    fcntl = None

from private_agent.tools.github_access import RepositoryAccessError, parse_repository, validate_repository_path
from private_agent.tools.local_repository import EXCLUDED_DIRECTORIES, sensitive_path
from private_agent.tools.contracts import ToolResult, validate_schema_value


PATCH_PREPARE_TOOL = "prepare_repository_patch"
PATCH_APPLY_TOOL = "apply_repository_patch"
PATCH_ROLLBACK_TOOL = "rollback_repository_patch"
PATCH_TOOL_NAMES = (PATCH_PREPARE_TOOL, PATCH_APPLY_TOOL, PATCH_ROLLBACK_TOOL)


class PatchSafetyError(Exception):
    """A bounded patch operation rejected for a deterministic safety reason."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True)
class PatchLimits:
    max_files: int = 8
    max_patch_bytes: int = 65536
    max_changed_lines: int = 240
    max_file_bytes: int = 65536
    max_tree_files: int = 4000
    max_total_tree_bytes: int = 4 * 1024 * 1024
    max_path_bytes: int = 256
    max_diagnosis_bytes: int = 65536

    def __post_init__(self) -> None:
        ceilings = {
            "max_files": 32,
            "max_patch_bytes": 512 * 1024,
            "max_changed_lines": 2000,
            "max_file_bytes": 256 * 1024,
        "max_tree_files": 10000,
        "max_total_tree_bytes": 16 * 1024 * 1024,
        "max_path_bytes": 1024,
        "max_diagnosis_bytes": 512 * 1024,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > ceiling:
                raise ValueError(f"invalid_patch_limit:{name}")


@dataclass(frozen=True)
class _Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _FilePatch:
    path: str
    hunks: tuple[_Hunk, ...]
    is_new: bool = False


@dataclass
class _PatchSession:
    patch_id: str
    repository: str
    base_revision: str
    permitted_files: list[str]
    diff: str
    bug_report: str
    diagnosis_link: dict[str, Any]
    file_patches: list[_FilePatch]
    original_bytes: dict[str, bytes]
    proposed_bytes: dict[str, bytes]
    workspace: Path | None = None
    status: str = "PREPARED"


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_SECRET_LITERAL_RE = re.compile(
    r"(?i)(?:-----BEGIN [^-]*PRIVATE KEY-----|\b(?:gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|AKIA[A-Z0-9]{16})\b|"
    r"\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|token|authorization)\s*[:=]\s*['\"][^'\"]{4,}['\"]|"
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,})"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _ensure_safe_path(path: str, *, max_path_bytes: int = 1024) -> str:
    try:
        validated = validate_repository_path(path)
    except RepositoryAccessError as exc:
        raise PatchSafetyError(exc.code) from None
    if any(part in EXCLUDED_DIRECTORIES for part in validated.split("/")):
        raise PatchSafetyError("excluded_path")
    if len(validated.encode("utf-8")) > max_path_bytes or sensitive_path(validated):
        raise PatchSafetyError("sensitive_or_unsafe_path")
    return validated


def _decode_text(data: bytes, path: str) -> str:
    if len(data) > 256 * 1024 or b"\x00" in data:
        raise PatchSafetyError(f"binary_or_large_file:{path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise PatchSafetyError(f"binary_file:{path}") from None
    if any(ord(char) < 9 or 13 < ord(char) < 32 for char in text):
        raise PatchSafetyError(f"binary_file:{path}")
    return text


def _strip_diff_path(value: str, *, max_path_bytes: int) -> str | None:
    raw = value.split("\t", 1)[0].strip()
    if raw == "/dev/null":
        return None
    if raw.startswith("a/") or raw.startswith("b/"):
        raw = raw[2:]
    return _ensure_safe_path(raw, max_path_bytes=max_path_bytes)


def _parse_unified_diff(diff: str, limits: PatchLimits) -> list[_FilePatch]:
    if not isinstance(diff, str) or not diff.strip():
        raise PatchSafetyError("patch_required")
    raw = diff.encode("utf-8", "strict")
    if len(raw) > limits.max_patch_bytes or b"\x00" in raw:
        raise PatchSafetyError("patch_size_exceeded")
    if _SECRET_LITERAL_RE.search(diff):
        raise PatchSafetyError("patch_contains_sensitive_literal")

    lines = diff.splitlines()
    index = 0
    patches: list[_FilePatch] = []
    changed_lines = 0
    while index < len(lines):
        if not lines[index].startswith("--- "):
            index += 1
            continue
        old_path = _strip_diff_path(lines[index][4:], max_path_bytes=limits.max_path_bytes)
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise PatchSafetyError("invalid_patch_headers")
        new_path = _strip_diff_path(lines[index][4:], max_path_bytes=limits.max_path_bytes)
        index += 1
        if old_path is None and new_path is None:
            raise PatchSafetyError("invalid_patch_path")
        if old_path is not None and new_path is not None and old_path != new_path:
            raise PatchSafetyError("renames_not_supported")
        path = new_path or old_path
        assert path is not None
        if new_path is None:
            raise PatchSafetyError("file_deletion_not_supported")
        hunks: list[_Hunk] = []
        while index < len(lines) and not lines[index].startswith("--- "):
            if not lines[index].startswith("@@ "):
                index += 1
                continue
            match = _HUNK_RE.match(lines[index])
            if match is None:
                raise PatchSafetyError("invalid_hunk_header")
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            index += 1
            hunk_lines: list[tuple[str, str]] = []
            old_seen = 0
            new_seen = 0
            while index < len(lines) and not lines[index].startswith(("@@ ", "--- ")):
                line = lines[index]
                index += 1
                if line == r"\ No newline at end of file":
                    raise PatchSafetyError("patch_no_newline_marker_unsupported")
                if not line or line[0] not in " +-":
                    raise PatchSafetyError("invalid_hunk_line")
                marker, content = line[0], line[1:]
                if marker in " -":
                    old_seen += 1
                if marker in " +":
                    new_seen += 1
                hunk_lines.append((marker, content))
            if old_seen != old_count or new_seen != new_count:
                raise PatchSafetyError("hunk_line_count_mismatch")
            changed_lines += sum(1 for marker, _ in hunk_lines if marker in "+-")
            if changed_lines > limits.max_changed_lines:
                raise PatchSafetyError("change_count_exceeded")
            hunks.append(_Hunk(old_start, old_count, new_start, new_count, tuple(hunk_lines)))
        if not hunks:
            raise PatchSafetyError("patch_hunk_required")
        patches.append(_FilePatch(path=path, hunks=tuple(hunks), is_new=old_path is None))
        if len(patches) > limits.max_files:
            raise PatchSafetyError("file_count_exceeded")

    if not patches:
        raise PatchSafetyError("invalid_patch_no_files")
    if len({item.path for item in patches}) != len(patches):
        raise PatchSafetyError("duplicate_patch_file")
    return patches


def _apply_file_patch(original: bytes, file_patch: _FilePatch) -> bytes:
    source_text = _decode_text(original, file_patch.path) if original else ""
    source_lines = source_text.splitlines()
    output: list[str] = []
    position = 0
    for hunk in file_patch.hunks:
        start = 0 if file_patch.is_new else max(hunk.old_start - 1, 0)
        if start < position or start > len(source_lines):
            raise PatchSafetyError("patch_context_ambiguous")
        output.extend(source_lines[position:start])
        position = start
        for marker, content in hunk.lines:
            if marker == " ":
                if position >= len(source_lines) or source_lines[position] != content:
                    raise PatchSafetyError("patch_context_mismatch")
                output.append(content)
                position += 1
            elif marker == "-":
                if position >= len(source_lines) or source_lines[position] != content:
                    raise PatchSafetyError("patch_removal_mismatch")
                position += 1
            else:
                output.append(content)
    output.extend(source_lines[position:])
    trailing_newline = source_text.endswith("\n") if source_text else True
    return ("\n".join(output) + ("\n" if trailing_newline and output else "")).encode("utf-8")


def _structural_validate(path: str, data: bytes, *, max_bytes: int = 256 * 1024) -> None:
    if len(data) > max_bytes:
        raise PatchSafetyError(f"file_too_large:{path}")
    text = _decode_text(data, path)
    suffix = Path(path).suffix.lower()
    try:
        if suffix == ".py":
            ast.parse(text, filename=path, mode="exec")
        elif suffix == ".json":
            json.loads(text)
    except (SyntaxError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise PatchSafetyError(f"structural_validation_failed:{path}") from exc


def _safe_lstat(root: Path, path: str, *, allow_missing: bool = False) -> os.stat_result | None:
    current = root
    for segment in path.split("/"):
        current = current / segment
        try:
            info = current.lstat()
        except FileNotFoundError:
            if allow_missing:
                return None
            raise PatchSafetyError(f"missing_file:{path}") from None
        if current.is_symlink():
            raise PatchSafetyError(f"symlink_path:{path}")
        if current != root / path and not info.st_mode:
            raise PatchSafetyError(f"unsafe_path:{path}")
    return current.lstat()


def _safe_read(root: Path, path: str, *, max_bytes: int) -> bytes:
    _safe_lstat(root, path)
    target = root / path
    info = target.lstat()
    if not stat_is_regular(info.st_mode):
        raise PatchSafetyError(f"unsupported_file:{path}")
    if info.st_size > max_bytes:
        raise PatchSafetyError(f"file_too_large:{path}")
    try:
        with target.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError as exc:
        raise PatchSafetyError(f"file_unavailable:{path}") from exc
    if len(data) > max_bytes:
        raise PatchSafetyError(f"file_too_large:{path}")
    return data


def stat_is_regular(mode: int) -> bool:
    return (mode & 0o170000) == 0o100000


def _safe_write(
    root: Path,
    path: str,
    data: bytes,
    *,
    overwrite: bool,
    expected_sha256: str | None = None,
) -> None:
    _ensure_safe_path(path)
    target = root / path
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        _safe_lstat(root, path, allow_missing=True)
    except PatchSafetyError as exc:
        if expected_sha256 is not None:
            raise PatchSafetyError("rollback_conflict") from exc
        raise
    if target.exists() and target.is_symlink():
        raise PatchSafetyError(f"symlink_path:{path}")
    if expected_sha256 is not None:
        try:
            current = _safe_read(root, path, max_bytes=256 * 1024)
        except PatchSafetyError as exc:
            raise PatchSafetyError("rollback_conflict") from exc
        if _sha256(current) != expected_sha256:
            raise PatchSafetyError("rollback_conflict")
    if target.exists() and not overwrite:
        raise PatchSafetyError(f"workspace_file_exists:{path}")
    if len(data) > 256 * 1024:
        raise PatchSafetyError(f"file_too_large:{path}")
    fd, temporary = tempfile.mkstemp(prefix=".patch-", dir=str(parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _safe_delete(root: Path, path: str, *, expected_sha256: str | None = None) -> None:
    try:
        _safe_lstat(root, path)
    except PatchSafetyError as exc:
        if expected_sha256 is not None:
            raise PatchSafetyError("rollback_conflict") from exc
        raise
    target = root / path
    if target.is_symlink() or not target.is_file():
        raise PatchSafetyError(f"unsafe_workspace_entry:{path}")
    if expected_sha256 is not None:
        try:
            current = _safe_read(root, path, max_bytes=256 * 1024)
        except PatchSafetyError as exc:
            raise PatchSafetyError("rollback_conflict") from exc
        if _sha256(current) != expected_sha256:
            raise PatchSafetyError("rollback_conflict")
    target.unlink()


def _workspace_snapshot(root: Path, limits: PatchLimits) -> tuple[dict[str, str], set[str]]:
    if not root.exists() or root.is_symlink() or not root.is_dir():
        raise PatchSafetyError("rollback_conflict")
    files: dict[str, str] = {}
    directories: set[str] = set()
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = child.relative_to(root).as_posix()
        if child.is_symlink():
            raise PatchSafetyError("rollback_conflict")
        _ensure_safe_path(relative, max_path_bytes=limits.max_path_bytes)
        if child.is_dir():
            directories.add(relative)
        elif child.is_file():
            if len(files) >= limits.max_tree_files:
                raise PatchSafetyError("rollback_conflict")
            files[relative] = _sha256(_safe_read(root, relative, max_bytes=limits.max_file_bytes))
        else:
            raise PatchSafetyError("rollback_conflict")
    return files, directories


def _workspace_parent_directories(paths: Iterable[str]) -> set[str]:
    directories: set[str] = set()
    for path in paths:
        parent = Path(path).parent
        while str(parent) != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _tree_revision(root: Path, limits: PatchLimits) -> str:
    if not root.exists() or root.is_symlink() or not root.is_dir():
        raise PatchSafetyError("repository_unavailable")
    entries: list[tuple[str, str, int]] = []
    total = 0
    visited = 0

    def walk(directory: Path, prefix: str) -> None:
        nonlocal total, visited
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise PatchSafetyError("repository_unavailable") from exc
        for child in children:
            visited += 1
            if visited > limits.max_tree_files:
                raise PatchSafetyError("repository_tree_limit_exceeded")
            relative = f"{prefix}{child.name}"
            if child.name in EXCLUDED_DIRECTORIES or sensitive_path(relative):
                continue
            _ensure_safe_path(relative)
            if child.is_symlink():
                raise PatchSafetyError("symlink_in_repository")
            if child.is_dir():
                walk(child, relative + "/")
                continue
            info = child.stat()
            if not stat_is_regular(info.st_mode):
                raise PatchSafetyError("unsupported_repository_entry")
            total += info.st_size
            if total > limits.max_total_tree_bytes:
                raise PatchSafetyError("repository_tree_size_exceeded")
            data = _safe_read(root, relative, max_bytes=limits.max_file_bytes)
            entries.append((relative, _sha256(data), len(data)))

    walk(root, "")
    payload = "\n".join(f"{path}\0{digest}\0{size}" for path, digest, size in entries).encode("utf-8")
    return "local-tree:" + _sha256(payload)


def compute_local_revision(root: Path, *, limits: PatchLimits | None = None) -> str:
    """Return the bounded, content-addressed revision used by local patch sessions."""
    return _tree_revision(Path(root), limits or PatchLimits())


def _safe_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _looks_like_secret(text: str) -> bool:
    return bool(_SECRET_LITERAL_RE.search(text))


class PatchWorkspaceManager:
    """Manage patch sessions rooted in trusted local-repository configuration."""

    def __init__(
        self,
        inspector: Any,
        *,
        workspace_root: Path | None = None,
        limits: PatchLimits | None = None,
    ) -> None:
        self.inspector = inspector
        self.limits = limits or PatchLimits()
        root = Path(workspace_root or Path("data") / "patch_workspaces")
        if root.exists() and root.is_symlink():
            raise ValueError("patch_workspace_root_must_not_be_symlink")
        self.workspace_root = root.resolve()
        for configured_root in (getattr(inspector, "local_repositories", {}) or {}).values():
            repository_root = Path(configured_root).resolve()
            if _paths_overlap(self.workspace_root, repository_root):
                raise ValueError("patch_workspace_root_must_be_separate")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._mutation_lock = threading.RLock()

    def _repository_root(self, repository: str) -> tuple[str, Path]:
        try:
            canonical = parse_repository(repository)
        except RepositoryAccessError as exc:
            raise PatchSafetyError(exc.code) from None
        roots = getattr(self.inspector, "local_repositories", {})
        root = roots.get(canonical.lower())
        if root is None:
            raise PatchSafetyError("modification_source_unsupported")
        root = Path(root)
        if root.is_symlink() or not root.is_dir():
            raise PatchSafetyError("repository_unavailable")
        return canonical, root

    def _session_path(self, patch_id: str) -> Path:
        if not isinstance(patch_id, str) or not re.fullmatch(r"[0-9a-f-]{36}", patch_id):
            raise PatchSafetyError("invalid_patch_id")
        path = self.workspace_root / patch_id
        if path.is_symlink():
            raise PatchSafetyError("unsafe_patch_session")
        return path

    @contextmanager
    def _session_lock(self, session_path: Path) -> Iterator[None]:
        with self._mutation_lock:
            lock_path = session_path / ".mutation.lock"
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            descriptor = os.open(lock_path, flags, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _load_metadata(self, patch_id: str) -> dict[str, Any]:
        session = self._session_path(patch_id)
        metadata_path = session / "metadata.json"
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise PatchSafetyError("patch_not_found")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise PatchSafetyError("patch_metadata_invalid") from exc
        if not isinstance(payload, dict) or payload.get("patch_id") != patch_id:
            raise PatchSafetyError("patch_metadata_invalid")
        return payload

    def _diagnosis_link(
        self,
        diagnosis: Any,
        *,
        repository: str,
        base_revision: str,
        root: Path,
        changed_paths: set[str],
    ) -> dict[str, Any]:
        if diagnosis in (None, {}):
            return {"available": False, "evidence_refs": []}
        if not isinstance(diagnosis, dict):
            raise PatchSafetyError("diagnosis_must_be_object")
        if diagnosis.get("repository") not in {repository, ""}:
            raise PatchSafetyError("diagnosis_repository_mismatch")
        diagnosis_revision = str(diagnosis.get("revision", ""))
        if diagnosis_revision and diagnosis_revision != base_revision:
            raise PatchSafetyError("diagnosis_revision_mismatch")
        raw_refs = diagnosis.get("evidence", [])
        if not isinstance(raw_refs, list) or len(raw_refs) > 60:
            raise PatchSafetyError("diagnosis_evidence_invalid")
        refs: list[dict[str, Any]] = []
        for raw in raw_refs:
            if not isinstance(raw, dict):
                continue
            path = raw.get("path")
            digest = raw.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                continue
            safe_path = _ensure_safe_path(path, max_path_bytes=self.limits.max_path_bytes)
            try:
                actual = _sha256(_safe_read(root, safe_path, max_bytes=self.limits.max_file_bytes))
            except PatchSafetyError:
                continue
            if actual != digest:
                raise PatchSafetyError("diagnosis_evidence_stale")
            refs.append({"path": safe_path, "sha256": digest, "line_start": int(raw.get("line_start", 1) or 1), "line_end": int(raw.get("line_end", 1) or 1)})
        if raw_refs and not refs:
            raise PatchSafetyError("diagnosis_evidence_unusable")
        if refs and not changed_paths.intersection({item["path"] for item in refs}):
            raise PatchSafetyError("patch_not_linked_to_diagnosis_evidence")
        root_cause = diagnosis.get("root_cause", {})
        return {
            "available": True,
            "repository": repository,
            "revision": diagnosis_revision,
            "root_cause": root_cause if isinstance(root_cause, dict) else {},
            "evidence_refs": refs,
        }

    def prepare(self, inputs: dict[str, Any]) -> dict[str, Any]:
        errors = validate_schema_value(inputs, self.prepare_schema(), path="input")
        if errors:
            raise PatchSafetyError("invalid_input")
        repository, root = self._repository_root(str(inputs["repository"]))
        base_revision = str(inputs["base_revision"])
        if not re.fullmatch(r"local-tree:[0-9a-f]{64}", base_revision):
            raise PatchSafetyError("invalid_base_revision")
        bug_report = str(inputs["bug_report"])
        if not bug_report.strip() or len(bug_report) > 4096:
            raise PatchSafetyError("invalid_bug_report")
        raw_permitted = inputs["permitted_files"]
        if not isinstance(raw_permitted, list) or not raw_permitted or len(raw_permitted) > self.limits.max_files:
            raise PatchSafetyError("file_count_exceeded")
        permitted = [
            _ensure_safe_path(str(item), max_path_bytes=self.limits.max_path_bytes)
            for item in raw_permitted
        ]
        if len(set(permitted)) != len(permitted):
            raise PatchSafetyError("duplicate_permitted_file")
        file_patches = _parse_unified_diff(str(inputs["patch"]), self.limits)
        changed_paths = {item.path for item in file_patches}
        if not changed_paths.issubset(set(permitted)):
            raise PatchSafetyError("patch_scope_exceeded")
        current_revision = compute_local_revision(root, limits=self.limits)
        if current_revision != base_revision:
            raise PatchSafetyError("base_revision_mismatch")
        original: dict[str, bytes] = {}
        proposed: dict[str, bytes] = {}
        for file_patch in file_patches:
            if file_patch.is_new:
                original[file_patch.path] = b""
            else:
                original[file_patch.path] = _safe_read(root, file_patch.path, max_bytes=self.limits.max_file_bytes)
            proposed[file_patch.path] = _apply_file_patch(original[file_patch.path], file_patch)
            _structural_validate(file_patch.path, proposed[file_patch.path], max_bytes=self.limits.max_file_bytes)
        diagnosis_input = inputs.get("diagnosis")
        if diagnosis_input not in (None, {}):
            try:
                diagnosis_size = len(json.dumps(diagnosis_input, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            except (TypeError, ValueError, RecursionError) as exc:
                raise PatchSafetyError("diagnosis_invalid") from exc
            if diagnosis_size > self.limits.max_diagnosis_bytes:
                raise PatchSafetyError("diagnosis_size_exceeded")
        diagnosis_link = self._diagnosis_link(
            diagnosis_input,
            repository=repository,
            base_revision=base_revision,
            root=root,
            changed_paths=changed_paths,
        )
        patch_id = str(uuid.uuid4())
        session = _PatchSession(
            patch_id=patch_id,
            repository=repository,
            base_revision=base_revision,
            permitted_files=permitted,
            diff=str(inputs["patch"]),
            bug_report=bug_report,
            diagnosis_link=diagnosis_link,
            file_patches=file_patches,
            original_bytes=original,
            proposed_bytes=proposed,
        )
        session_path = self._session_path(patch_id)
        session_path.mkdir(parents=True, exist_ok=False)
        metadata = {
            "patch_id": patch_id,
            "status": "PREPARED",
            "repository": repository,
            "base_revision": base_revision,
            "permitted_files": permitted,
            "bug_report": session.bug_report,
            "diagnosis": diagnosis_link,
            "diff": session.diff,
            "patch_sha256": _sha256(session.diff.encode("utf-8")),
            "changed_files": [
                {"path": path, "original_sha256": _sha256(original[path]), "proposed_sha256": _sha256(proposed[path]), "created": not bool(original[path]) and next(item for item in file_patches if item.path == path).is_new}
                for path in sorted(changed_paths)
            ],
            "prepared_at": _now(),
            "target_code_executed": False,
            "application": {"status": "NOT_APPLIED"},
            "rollback": {"status": "NOT_AVAILABLE"},
        }
        _safe_json_write(session_path / "metadata.json", metadata)
        return {
            "patch_id": patch_id,
            "status": "PREPARED",
            "repository": repository,
            "base_revision": base_revision,
            "permitted_files": permitted,
            "changed_files": sorted(changed_paths),
            "diff": session.diff,
            "patch_sha256": metadata["patch_sha256"],
            "diagnosis": diagnosis_link,
            "validation": {"structural": "passed", "target_code_executed": False, "tests_run": []},
            "application": {"status": "NOT_APPLIED", "approval_required": True},
            "rollback": {"status": "NOT_AVAILABLE"},
            "remaining_uncertainties": ["Patch preparation does not prove runtime behavior; target tests are intentionally not run."],
        }

    def apply(self, inputs: dict[str, Any]) -> dict[str, Any]:
        errors = validate_schema_value(inputs, self.apply_schema(), path="input")
        if errors:
            raise PatchSafetyError("invalid_input")
        patch_id = str(inputs["patch_id"])
        session_path = self._session_path(patch_id)
        self._load_metadata(patch_id)
        with self._session_lock(session_path):
            metadata = self._load_metadata(patch_id)
            if metadata.get("status") != "PREPARED":
                raise PatchSafetyError("patch_not_prepared")
            repository, root = self._repository_root(str(metadata["repository"]))
            if repository != metadata.get("repository"):
                raise PatchSafetyError("repository_identity_mismatch")
            if compute_local_revision(root, limits=self.limits) != metadata.get("base_revision"):
                raise PatchSafetyError("base_revision_mismatch")
            file_patches = _parse_unified_diff(str(metadata.get("diff", "")), self.limits)
            original: dict[str, bytes] = {}
            proposed: dict[str, bytes] = {}
            workspace = session_path / "workspace"
            workspace.mkdir(parents=True, exist_ok=False)
            try:
                for file_patch in file_patches:
                    if file_patch.is_new:
                        original[file_patch.path] = b""
                    else:
                        original[file_patch.path] = _safe_read(root, file_patch.path, max_bytes=self.limits.max_file_bytes)
                    _safe_write(workspace, file_patch.path, original[file_patch.path], overwrite=False)
                    proposed[file_patch.path] = _apply_file_patch(original[file_patch.path], file_patch)
                    _structural_validate(file_patch.path, proposed[file_patch.path], max_bytes=self.limits.max_file_bytes)
                    _safe_write(workspace, file_patch.path, proposed[file_patch.path], overwrite=True)
                changed = {item["path"]: item for item in metadata.get("changed_files", []) if isinstance(item, dict)}
                if set(changed) != set(proposed):
                    raise PatchSafetyError("application_integrity_mismatch")
                for path, data in proposed.items():
                    if _sha256(data) != changed[path].get("proposed_sha256"):
                        raise PatchSafetyError("application_integrity_mismatch")
                rollback_dir = session_path / "rollback"
                rollback_dir.mkdir()
                for path, data in original.items():
                    _safe_write(rollback_dir, path, data, overwrite=False)
                metadata.update({
                    "status": "APPLIED",
                    "applied_at": _now(),
                    "workspace_rel": f"{patch_id}/workspace",
                    "target_code_executed": False,
                    "application": {"status": "APPLIED", "structural_validation": "passed", "approval_state": "verified_by_execution_boundary"},
                    "rollback": {"status": "AVAILABLE"},
                })
                _safe_json_write(session_path / "metadata.json", metadata)
            except Exception:
                # Do not expose a partially applied workspace as successful.
                self._remove_tree(workspace)
                raise
            result = {
                "patch_id": patch_id,
                "status": "APPLIED",
                "repository": repository,
                "base_revision": metadata["base_revision"],
                "changed_files": sorted(path for path in proposed),
                "diff": metadata["diff"],
                "workspace": f"{patch_id}/workspace",
                "validation": {"structural": "passed", "target_code_executed": False, "tests_run": []},
                "application": {"status": "APPLIED", "approval_state": "verified_by_execution_boundary"},
                "rollback": {"status": "AVAILABLE", "patch_id": patch_id},
                "remaining_uncertainties": ["Runtime behavior remains unverified because target code and tests were not executed."],
            }
        return result

    def rollback(self, inputs: dict[str, Any]) -> dict[str, Any]:
        errors = validate_schema_value(inputs, self.rollback_schema(), path="input")
        if errors:
            raise PatchSafetyError("invalid_input")
        patch_id = str(inputs["patch_id"])
        session_path = self._session_path(patch_id)
        self._load_metadata(patch_id)
        with self._session_lock(session_path):
            metadata = self._load_metadata(patch_id)
            if metadata.get("status") != "APPLIED":
                raise PatchSafetyError("rollback_not_available")
            workspace = session_path / "workspace"
            rollback_dir = session_path / "rollback"
            changed_files = [item for item in metadata.get("changed_files", []) if isinstance(item, dict)]
            expected_files = {
                str(item["path"]): str(item["proposed_sha256"])
                for item in changed_files
            }
            expected_dirs = _workspace_parent_directories(expected_files)
            actual_files, actual_dirs = _workspace_snapshot(workspace, self.limits)
            if actual_files != expected_files or actual_dirs != expected_dirs:
                raise PatchSafetyError("rollback_conflict")

            originals: dict[str, bytes] = {}
            for item in changed_files:
                path = _ensure_safe_path(str(item.get("path", "")), max_path_bytes=self.limits.max_path_bytes)
                if item.get("created"):
                    continue
                original_path = rollback_dir / path
                if not original_path.is_file() or original_path.is_symlink():
                    raise PatchSafetyError("rollback_backup_missing")
                original = _safe_read(rollback_dir, path, max_bytes=self.limits.max_file_bytes)
                if _sha256(original) != item.get("original_sha256"):
                    raise PatchSafetyError("rollback_backup_invalid")
                originals[path] = original

            working_files = dict(expected_files)
            working_dirs = set(expected_dirs)
            for item in changed_files:
                current_files, current_dirs = _workspace_snapshot(workspace, self.limits)
                if current_files != working_files or current_dirs != working_dirs:
                    raise PatchSafetyError("rollback_conflict")
                path = _ensure_safe_path(str(item["path"]), max_path_bytes=self.limits.max_path_bytes)
                if item.get("created"):
                    _safe_delete(workspace, path, expected_sha256=working_files[path])
                    parent = (workspace / path).parent
                    while parent != workspace:
                        try:
                            parent.rmdir()
                        except OSError:
                            break
                        parent = parent.parent
                else:
                    _safe_write(
                        workspace,
                        path,
                        originals[path],
                        overwrite=True,
                        expected_sha256=working_files[path],
                    )
                working_files, working_dirs = _workspace_snapshot(workspace, self.limits)

            final_files, final_dirs = _workspace_snapshot(workspace, self.limits)
            if final_files != working_files or final_dirs != working_dirs:
                raise PatchSafetyError("rollback_conflict")
            metadata.update({"status": "ROLLED_BACK", "rolled_back_at": _now(), "rollback": {"status": "ROLLED_BACK"}})
            _safe_json_write(session_path / "metadata.json", metadata)
            result = {
                "patch_id": patch_id,
                "status": "ROLLED_BACK",
                "changed_files": [str(item["path"]) for item in changed_files],
                "validation": {"structural": "passed", "target_code_executed": False},
                "application": {"status": "REVERTED_IN_ISOLATED_WORKSPACE"},
                "rollback": {"status": "ROLLED_BACK"},
                "remaining_uncertainties": [],
            }
        return result

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if not path.exists():
            return
        for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if child.is_symlink() or child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        path.rmdir()

    @staticmethod
    def prepare_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repository": {"type": "string"},
                "base_revision": {"type": "string"},
                "bug_report": {"type": "string"},
                "permitted_files": {"type": "array", "items": {"type": "string"}},
                "patch": {"type": "string"},
                "diagnosis": {"type": "object"},
            },
            "required": ["repository", "base_revision", "bug_report", "permitted_files", "patch"],
            "additionalProperties": False,
        }

    @staticmethod
    def apply_schema() -> dict[str, Any]:
        return {"type": "object", "properties": {"patch_id": {"type": "string"}}, "required": ["patch_id"], "additionalProperties": False}

    @staticmethod
    def rollback_schema() -> dict[str, Any]:
        return {"type": "object", "properties": {"patch_id": {"type": "string"}}, "required": ["patch_id"], "additionalProperties": False}


class _PatchTool:
    def __init__(self, name: str, manager: PatchWorkspaceManager) -> None:
        if name not in PATCH_TOOL_NAMES:
            raise ValueError("unknown_patch_tool")
        self.name = name
        self.manager = manager
        self.required_capabilities = ["READ_WORKSPACE"] if name == PATCH_PREPARE_TOOL else ["WRITE_WORKSPACE"]
        self.declared_capabilities = list(self.required_capabilities)
        self.network_required = False
        self.filesystem_required = True
        self.risk_level = "low" if name == PATCH_PREPARE_TOOL else "high"
        # isolated_write is intentionally distinct from private_write: the tool
        # never writes the configured source tree or an arbitrary caller path.
        self.permission_level = "public_read" if name == PATCH_PREPARE_TOOL else "isolated_write"
        self.planned_effect = {
            PATCH_PREPARE_TOOL: "Validate a bounded patch and persist review metadata; do not modify the target repository or execute target code",
            PATCH_APPLY_TOOL: "Apply an approved bounded patch only inside an agent-owned isolated workspace; never modify upstream",
            PATCH_ROLLBACK_TOOL: "Rollback an applied patch only inside its isolated workspace after conflict checks; never modify upstream",
        }[name]

    def description(self) -> str:
        return self.planned_effect

    def input_schema(self) -> dict[str, Any]:
        if self.name == PATCH_PREPARE_TOOL:
            return self.manager.prepare_schema()
        if self.name == PATCH_APPLY_TOOL:
            return self.manager.apply_schema()
        return self.manager.rollback_schema()

    def output_schema(self) -> dict[str, Any]:
        return {"type": "object"}

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        if self.name != PATCH_PREPARE_TOOL:
            return ToolResult.blocked(
                "security_execution_required",
                "Mutating patch operations require the trusted executor boundary",
            )
        return self._execute_operation(inputs)

    def _execute_trusted(
        self,
        inputs: dict[str, Any],
        context: Any,
        *,
        operation_binding: Any,
        authorization: Any,
        operation: Any,
    ) -> ToolResult:
        if self.name != PATCH_PREPARE_TOOL and not self._trusted_boundary_valid(
            inputs,
            context,
            operation_binding=operation_binding,
            authorization=authorization,
            operation=operation,
        ):
            return ToolResult.blocked(
                "security_execution_required",
                "Mutating patch operations require the trusted executor boundary",
            )
        return self._execute_operation(inputs)

    def _trusted_boundary_valid(
        self,
        inputs: dict[str, Any],
        context: Any,
        *,
        operation_binding: Any,
        authorization: Any,
        operation: Any,
    ) -> bool:
        controller = getattr(operation_binding, "controller", None)
        context_controller = getattr(context, "security_controller", None)
        approved_permissions = getattr(context, "approved_permissions", set())
        if (
            controller is None
            or (context_controller is not None and context_controller is not controller)
            or self.permission_level not in approved_permissions
        ):
            return False
        try:
            action_binding = operation_binding.action_binding
            if not isinstance(action_binding, tuple) or len(action_binding) < 2:
                return False
            metadata = getattr(context, "metadata", {})
            goal = str(metadata.get("goal", "")) if isinstance(metadata, dict) else ""
            action = controller.action_for_tool(
                str(getattr(context, "task_id", "")),
                str(action_binding[1]),
                self,
                inputs,
                goal=goal,
            )
            validate_binding = getattr(controller, "_validate_operation_binding", None)
            pending_boundary = getattr(controller, "_pending_boundary", None)
            if not callable(validate_binding) or not callable(pending_boundary):
                return False
            if not validate_binding(operation_binding, action, operation):
                return False
            evidence_lock = getattr(controller, "_authorization_evidence_lock", None)
            evidence = getattr(controller, "_authorization_evidence", None)
            inflight = getattr(controller, "_execution_inflight", None)
            if evidence_lock is None or not isinstance(evidence, dict) or not isinstance(inflight, set):
                return False
            with evidence_lock:
                issued = evidence.get(id(authorization))
                if issued is None or issued[0] is not authorization or id(authorization) not in inflight:
                    return False
            pending_boundary(action, authorization)
            return True
        except Exception:
            return False

    def _execute_operation(self, inputs: dict[str, Any]) -> ToolResult:
        try:
            if self.name == PATCH_PREPARE_TOOL:
                return ToolResult.success(self.manager.prepare(inputs), metadata={"target_code_executed": False, "isolated": True})
            if self.name == PATCH_APPLY_TOOL:
                return ToolResult.success(self.manager.apply(inputs), metadata={"target_code_executed": False, "isolated": True})
            return ToolResult.success(self.manager.rollback(inputs), metadata={"target_code_executed": False, "isolated": True})
        except PatchSafetyError as exc:
            if exc.code in {"patch_scope_exceeded", "invalid_path", "sensitive_or_unsafe_path", "excluded_path", "symlink_path", "symlink_in_repository", "base_revision_mismatch", "diagnosis_revision_mismatch", "diagnosis_evidence_stale", "diagnosis_invalid", "diagnosis_size_exceeded", "patch_not_linked_to_diagnosis_evidence", "patch_contains_sensitive_literal", "file_count_exceeded", "rollback_conflict", "modification_source_unsupported"}:
                return ToolResult.blocked(exc.code, "Patch operation was rejected by its safety boundary")
            return ToolResult.failed(exc.code, "Patch operation failed safely")
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            return ToolResult.failed("patch_operation_failed", "Patch operation failed safely")


def register_patch_tools(registry: Any, inspector: Any, *, workspace_root: Path | None = None) -> PatchWorkspaceManager:
    manager = PatchWorkspaceManager(inspector, workspace_root=workspace_root)
    for name in PATCH_TOOL_NAMES:
        registry.register(_PatchTool(name, manager))
    return manager


__all__ = [
    "PATCH_APPLY_TOOL",
    "PATCH_PREPARE_TOOL",
    "PATCH_ROLLBACK_TOOL",
    "PATCH_TOOL_NAMES",
    "PatchLimits",
    "PatchSafetyError",
    "PatchWorkspaceManager",
    "compute_local_revision",
    "register_patch_tools",
]
