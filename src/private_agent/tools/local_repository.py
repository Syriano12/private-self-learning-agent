"""Operator-selected local fixture source; never accepts model-supplied root paths."""
from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from private_agent.tools.github_access import RepositoryAccessError, validate_repository_path

EXCLUDED_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}


def sensitive_path(path: str) -> bool:
    """Do not surface credential files even if committed to the repository."""
    parts = path.lower().split("/")
    return any(
        part.startswith(".env") or part in {
            ".git", ".ssh", ".aws", ".azure", ".kube", ".netrc", ".npmrc", ".pypirc", ".git-credentials",
            ".secret", ".secrets", "id_rsa", "id_ed25519", "credentials", "secret", "secrets",
        }
        or part.endswith((".pem", ".key", ".p12", ".pfx"))
        or part.startswith(("credentials.", "secret.", "secrets.", ".secret.", ".secrets."))
        for part in parts
    )


class LocalRepository:
    """Bounded non-following directory reads. No subprocess or repository execution."""

    def __init__(self, repository: str, root: Path, *, max_file_bytes: int = 65536) -> None:
        self.repository = repository
        self.max_file_bytes = max_file_bytes
        try:
            self._root = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError:
            raise RepositoryAccessError("repository_unavailable") from None

    def close(self) -> None:
        if self._root >= 0:
            os.close(self._root)
            self._root = -1

    def metadata(self) -> dict[str, Any]:
        return {"repository": self.repository, "ref": "working-tree", "commit": "",
                "private": False, "source": "local_fixture"}

    def tree(self, limit: int = 200) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        visited = 0
        truncated = False

        def walk(fd: int, prefix: str, depth: int) -> None:
            nonlocal visited, truncated
            if depth > 8:
                truncated = True
                return
            # scandir is streamed: no unrestricted list of directory entries.
            with os.scandir(fd) as entries:
                for entry in entries:
                    visited += 1
                    if visited > 4000 or len(files) >= limit:
                        truncated = True
                        break
                    path = f"{prefix}{entry.name}"
                    if sensitive_path(path) or entry.name in EXCLUDED_DIRECTORIES:
                        continue
                    try:
                        validate_repository_path(path)
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISREG(info.st_mode):
                            files.append({"path": path, "type": "file", "size": info.st_size, "sha": ""})
                        elif stat.S_ISDIR(info.st_mode):
                            child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                            try:
                                walk(child, path + "/", depth + 1)
                            finally:
                                os.close(child)
                        else:
                            truncated = True  # Symlink/special entries are not followed or silently called complete.
                    except (OSError, RepositoryAccessError):
                        truncated = True
                    if visited > 4000 or len(files) >= limit:
                        truncated = True
                        break

        walk(self._root, "", 0)
        return {"files": sorted(files, key=lambda item: item["path"]), "truncated": truncated}

    def read(self, path: str, *, max_bytes: int | None = None) -> bytes:
        validate_repository_path(path)
        if sensitive_path(path):
            raise RepositoryAccessError("sensitive_path")
        limit = self.max_file_bytes if max_bytes is None else min(self.max_file_bytes, max_bytes)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise RepositoryAccessError("read_budget_exceeded")
        fd = os.dup(self._root)
        try:
            segments = path.split("/")
            for segment in segments[:-1]:
                child = os.open(segment, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            file_fd = os.open(segments[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            try:
                info = os.fstat(file_fd)
                if not stat.S_ISREG(info.st_mode):
                    raise RepositoryAccessError("unsupported_file")
                if info.st_size > limit:
                    raise RepositoryAccessError("file_too_large")
                data = bytearray()
                while len(data) <= limit:
                    chunk = os.read(file_fd, min(8192, limit + 1 - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
                if len(data) > limit:
                    raise RepositoryAccessError("file_too_large")
                return bytes(data)
            finally:
                os.close(file_fd)
        except FileNotFoundError:
            raise RepositoryAccessError("missing_file") from None
        except OSError:
            raise RepositoryAccessError("unsafe_or_unavailable_file") from None
        finally:
            os.close(fd)
