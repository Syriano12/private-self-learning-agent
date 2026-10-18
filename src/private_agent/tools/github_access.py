from __future__ import annotations

import base64
import binascii
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import quote, urlsplit

import httpx


_API_BASE = "https://api.github.com"
_SHA_RE = re.compile(r"\A[0-9a-fA-F]{40}\Z")
_REPOSITORY_PART_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SAFE_MESSAGES = {
    "invalid_repository": "Repository identifier is invalid.",
    "invalid_path": "Repository path is invalid.",
    "timeout": "GitHub request timed out.",
    "rate_limited": "GitHub rate limit reached.",
    "repository_unavailable": "Repository is unavailable.",
    "private_repository_not_authorized": "Private repository is not authorized.",
    "missing_file": "File is unavailable.",
    "api_error": "GitHub API request failed.",
    "response_too_large": "GitHub API response is too large.",
    "file_too_large": "Repository file is too large.",
}


class RepositoryAccessError(Exception):
    """A repository access failure with a fixed, non-sensitive public message."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(_SAFE_MESSAGES.get(code, _SAFE_MESSAGES["api_error"]))


def _contains_control(value: str) -> bool:
    return any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)


def _repository_parts(value: str, *, from_url: bool) -> tuple[str, str]:
    if not isinstance(value, str) or not value or _contains_control(value) or "\\" in value or "%" in value:
        raise RepositoryAccessError("invalid_repository")
    if "?" in value or "#" in value:
        raise RepositoryAccessError("invalid_repository")

    if from_url:
        try:
            parsed = urlsplit(value)
        except ValueError:
            raise RepositoryAccessError("invalid_repository") from None
        try:
            port = parsed.port
        except ValueError:
            raise RepositoryAccessError("invalid_repository") from None
        if (
            parsed.scheme != "https"
            or parsed.hostname != "github.com"
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path
        ):
            raise RepositoryAccessError("invalid_repository")
        path = parsed.path
        if not path.startswith("/"):
            raise RepositoryAccessError("invalid_repository")
        path = path[1:]
        if path.endswith(".git"):
            path = path[:-4]
    else:
        if "://" in value or value.endswith(".git"):
            raise RepositoryAccessError("invalid_repository")
        path = value

    parts = path.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise RepositoryAccessError("invalid_repository")
    if any(not _REPOSITORY_PART_RE.fullmatch(part) for part in parts):
        raise RepositoryAccessError("invalid_repository")
    return parts[0], parts[1]


def parse_repository(value: str) -> str:
    """Return the canonical ``owner/name`` form for a GitHub repository."""

    if not isinstance(value, str):
        raise RepositoryAccessError("invalid_repository")
    is_url = value.startswith("https://")
    if value.startswith(("http://", "HTTP://", "HTTPS://")) or ("://" in value and not is_url):
        raise RepositoryAccessError("invalid_repository")
    owner, name = _repository_parts(value, from_url=is_url)
    return f"{owner}/{name}"


def validate_repository_path(path: str) -> str:
    """Validate and return a repository-relative slash-separated file path."""

    if (
        not isinstance(path, str)
        or not path
        or _contains_control(path)
        or "\\" in path
        or "%" in path
        or "?" in path
        or "#" in path
    ):
        raise RepositoryAccessError("invalid_path")
    if path.startswith("/") or path.endswith("/") or re.match(r"\A[A-Za-z]:/", path):
        raise RepositoryAccessError("invalid_path")
    parts = path.split("/")
    if any(
        not part
        or part in {".", ".."}
        or part.lower() == ".git"
        for part in parts
    ):
        raise RepositoryAccessError("invalid_path")
    return path


def _validate_ref(ref: str) -> str:
    """Validate a Git ref before percent-encoding it into an API route."""

    if not isinstance(ref, str) or not ref or _contains_control(ref) or "\\" in ref or "%" in ref:
        raise RepositoryAccessError("invalid_path")
    if ref == "@" or ref.startswith("/") or ref.endswith("/") or "//" in ref or "@{" in ref or ".." in ref:
        raise RepositoryAccessError("invalid_path")
    if any(
        part in {".", ".."}
        or part.startswith(".")
        or part.endswith(".")
        or part.endswith(".lock")
        for part in ref.split("/")
    ):
        raise RepositoryAccessError("invalid_path")
    if any(character in ref for character in (" ", "~", "^", ":", "?", "*", "[", "#", "<", ">", "&", "=", "`", '"', "'", "|", ";")):
        raise RepositoryAccessError("invalid_path")
    return ref


def _validate_sha(value: Any) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise RepositoryAccessError("api_error")
    return value.lower()


def _decode_base64_bounded(content: str, limit: int) -> bytes:
    compact = "".join(content.split())
    if len(compact) % 4 or not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", compact):
        raise RepositoryAccessError("api_error")
    decoded = bytearray()
    chunk_width = 8192
    for start in range(0, len(compact), chunk_width):
        chunk = compact[start:start + chunk_width]
        if "=" in chunk and start + chunk_width < len(compact):
            raise RepositoryAccessError("api_error")
        try:
            decoded_chunk = base64.b64decode(chunk, validate=True)
        except (binascii.Error, ValueError, TypeError):
            raise RepositoryAccessError("api_error") from None
        if len(decoded) + len(decoded_chunk) > limit:
            raise RepositoryAccessError("file_too_large")
        decoded.extend(decoded_chunk)
    return bytes(decoded)


def _route_part(value: str) -> str:
    return quote(value, safe="")


@dataclass(frozen=True)
class _FileEntry:
    path: str
    size: int
    sha: str


class GitHubRepository:
    """An immutable, commit-pinned view of a GitHub repository."""

    def __init__(
        self,
        access: "GitHubAccess",
        repository: str,
        ref: str,
        commit: str,
        private: bool,
        authenticated: bool,
    ) -> None:
        self._access = access
        self._repository = repository
        self._owner, self._name = repository.split("/", 1)
        self._ref = ref
        self._commit = commit
        self._private = private
        self._authenticated = authenticated
        self._tree_attempted = False
        self._tree_entries: tuple[_FileEntry, ...] = ()
        self._tree_truncated = False
        self._tree_error: RepositoryAccessError | None = None

    def __repr__(self) -> str:
        return (
            f"GitHubRepository(repository={self._repository!r}, ref={self._ref!r}, "
            f"commit={self._commit!r}, private={self._private!r})"
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "repository": self._repository,
            "ref": self._ref,
            "commit": self._commit,
            "private": self._private,
            "source": "github_api",
        }

    def _tree_route(self) -> str:
        return (
            f"/repos/{_route_part(self._owner)}/{_route_part(self._name)}"
            f"/git/trees/{_route_part(self._commit)}?recursive=1"
        )

    def _blob_route(self, sha: str) -> str:
        return f"/repos/{_route_part(self._owner)}/{_route_part(self._name)}/git/blobs/{_route_part(sha)}"

    def _ensure_tree(self) -> None:
        if self._tree_attempted:
            if self._tree_error is not None:
                raise self._tree_error
            return
        self._tree_attempted = True
        try:
            payload = self._access._request_json(self._tree_route(), authenticated=self._authenticated, kind="tree")
            if not isinstance(payload, dict) or not isinstance(payload.get("tree"), list) or not isinstance(payload.get("truncated"), bool):
                raise RepositoryAccessError("api_error")
            entries: list[_FileEntry] = []
            seen: set[str] = set()
            for raw_entry in payload["tree"]:
                if not isinstance(raw_entry, dict):
                    raise RepositoryAccessError("api_error")
                entry_type = raw_entry.get("type")
                mode = raw_entry.get("mode")
                raw_path = raw_entry.get("path")
                if not isinstance(raw_path, str):
                    raise RepositoryAccessError("api_error")
                if mode is not None and not isinstance(mode, str):
                    raise RepositoryAccessError("api_error")
                try:
                    path = validate_repository_path(raw_path)
                except RepositoryAccessError:
                    raise RepositoryAccessError("api_error") from None
                if entry_type in {"symlink", "commit"} or mode in {"120000", "160000"}:
                    raise RepositoryAccessError("api_error")
                if entry_type == "tree":
                    continue
                if entry_type != "blob":
                    raise RepositoryAccessError("api_error")
                size = raw_entry.get("size")
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    raise RepositoryAccessError("api_error")
                sha = _validate_sha(raw_entry.get("sha"))
                if path in seen:
                    raise RepositoryAccessError("api_error")
                seen.add(path)
                entries.append(_FileEntry(path=path, size=size, sha=sha))
            self._tree_entries = tuple(entries)
            self._tree_truncated = payload["truncated"]
        except RepositoryAccessError as exc:
            self._tree_error = exc
            raise

    def tree(self, limit: int = 200) -> dict[str, Any]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise RepositoryAccessError("api_error")
        self._ensure_tree()
        files = [
            {"path": entry.path, "type": "file", "size": entry.size, "sha": entry.sha}
            for entry in self._tree_entries[:limit]
        ]
        return {"files": files, "truncated": self._tree_truncated or len(self._tree_entries) > limit}

    def read(self, path: str, *, max_bytes: int | None = None) -> bytes:
        validate_repository_path(path)
        limit = self._access.max_file_bytes if max_bytes is None else min(self._access.max_file_bytes, max_bytes)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise RepositoryAccessError("file_too_large")
        self._ensure_tree()
        entry = next((candidate for candidate in self._tree_entries if candidate.path == path), None)
        if entry is None:
            raise RepositoryAccessError("missing_file")
        if entry.size > limit:
            raise RepositoryAccessError("file_too_large")
        payload = self._access._request_json(
            self._blob_route(entry.sha), authenticated=self._authenticated, kind="blob"
        )
        if not isinstance(payload, dict) or payload.get("encoding") != "base64" or not isinstance(payload.get("content"), str):
            raise RepositoryAccessError("api_error")
        decoded = _decode_base64_bounded(payload["content"], limit)
        if len(decoded) != entry.size:
            raise RepositoryAccessError("api_error")
        return decoded


class GitHubAccess:
    """Read-only GitHub repository access with an explicit private allowlist."""

    def __init__(
        self,
        token: str = "",
        authorized_private_repositories: Iterable[str] = (),
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        timeout: float = 10,
        max_api_bytes: int = 2 * 1024 * 1024,
        max_file_bytes: int = 64 * 1024,
    ) -> None:
        if not isinstance(token, str):
            raise ValueError("token must be a string")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if not isinstance(max_api_bytes, int) or isinstance(max_api_bytes, bool) or max_api_bytes <= 0:
            raise ValueError("max_api_bytes must be positive")
        if not isinstance(max_file_bytes, int) or isinstance(max_file_bytes, bool) or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        canonical: set[str] = set()
        for repository in authorized_private_repositories:
            canonical.add(parse_repository(repository).lower())
        self._token = token
        self.authorized_private_repositories = frozenset(canonical)
        self._transport = transport
        self.timeout = timeout
        self.max_api_bytes = max_api_bytes
        self.max_file_bytes = max_file_bytes

    @classmethod
    def from_environment(cls, **kwargs: Any) -> "GitHubAccess":
        token = os.environ.get("GITHUB_READ_TOKEN", "")
        repositories_value = os.environ.get("GITHUB_READ_REPOSITORIES", "")
        repositories = tuple(
            item.strip() for item in repositories_value.split(",") if item.strip()
        )
        return cls(token=token, authorized_private_repositories=repositories, **kwargs)

    def __repr__(self) -> str:
        return (
            "GitHubAccess("
            f"authorized_private_repositories={tuple(sorted(self.authorized_private_repositories))!r}, "
            f"timeout={self.timeout!r}, max_api_bytes={self.max_api_bytes!r}, "
            f"max_file_bytes={self.max_file_bytes!r})"
        )

    def redact(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not self._token:
            return text
        return text.replace(self._token, "[REDACTED]")

    def _allowlisted(self, repository: str) -> bool:
        return bool(self._token) and repository.lower() in self.authorized_private_repositories

    def _metadata_route(self, owner: str, name: str) -> str:
        return f"/repos/{_route_part(owner)}/{_route_part(name)}"

    def open(self, repository: str, ref: str = "") -> GitHubRepository:
        canonical = parse_repository(repository)
        owner, name = canonical.split("/", 1)
        allowlisted = self._allowlisted(canonical)
        metadata_route = self._metadata_route(owner, name)
        try:
            payload = self._request_json(metadata_route, authenticated=False, kind="metadata")
        except RepositoryAccessError as exc:
            if not allowlisted or exc.code != "repository_unavailable":
                raise
            payload = self._request_json(metadata_route, authenticated=True, kind="metadata")

        if not isinstance(payload, dict) or not isinstance(payload.get("private"), bool):
            raise RepositoryAccessError("api_error")
        private = payload["private"]
        if private and not allowlisted:
            if self._token:
                raise RepositoryAccessError("private_repository_not_authorized")
            raise RepositoryAccessError("repository_unavailable")

        default_branch = payload.get("default_branch")
        if not isinstance(default_branch, str):
            raise RepositoryAccessError("api_error")
        selected_ref = default_branch if ref == "" else ref
        _validate_ref(selected_ref)
        commit_payload = self._request_json(
            f"{metadata_route}/commits/{_route_part(selected_ref)}",
            authenticated=private and allowlisted,
            kind="commit",
        )
        if not isinstance(commit_payload, dict):
            raise RepositoryAccessError("api_error")
        commit = _validate_sha(commit_payload.get("sha"))
        return GitHubRepository(
            self,
            canonical,
            selected_ref,
            commit,
            private,
            private and allowlisted,
        )

    def _request_json(self, route: str, *, authenticated: bool, kind: str) -> Any:
        headers = {"Accept": "application/vnd.github+json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self._token}"
        url = _API_BASE + route
        try:
            timeout = httpx.Timeout(float(self.timeout))
            with httpx.Client(
                timeout=timeout,
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                with client.stream("GET", url, headers=headers) as response:
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes():
                        if not isinstance(chunk, bytes):
                            raise RepositoryAccessError("api_error")
                        total += len(chunk)
                        if total > self.max_api_bytes:
                            raise RepositoryAccessError("response_too_large")
                        chunks.append(chunk)
                    body = b"".join(chunks)
                    status = response.status_code
        except RepositoryAccessError:
            raise
        except (httpx.TimeoutException, TimeoutError):
            raise RepositoryAccessError("timeout") from None
        except httpx.RequestError:
            raise RepositoryAccessError("api_error") from None
        except OSError:
            raise RepositoryAccessError("api_error") from None
        except Exception:
            raise RepositoryAccessError("api_error") from None

        if status in {403, 429}:
            raise RepositoryAccessError("rate_limited")
        if status == 404:
            if kind == "blob":
                raise RepositoryAccessError("missing_file")
            raise RepositoryAccessError("repository_unavailable")
        if status == 413:
            raise RepositoryAccessError("response_too_large")
        if status < 200 or status >= 300:
            raise RepositoryAccessError("api_error")
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            raise RepositoryAccessError("api_error") from None


__all__ = [
    "GitHubAccess",
    "GitHubRepository",
    "RepositoryAccessError",
    "parse_repository",
    "validate_repository_path",
]
