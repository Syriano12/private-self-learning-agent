"""Operator-selected local fixture source; never accepts model-supplied root paths."""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any

from private_agent.tools.github_access import RepositoryAccessError, validate_repository_path
from private_agent.tools.repository_paths import sensitive_path

EXCLUDED_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}


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
        return {"repository": self.repository, "ref": "working-tree", "commit": self._git_commit(),
                "private": False, "source": "local_fixture"}

    def _git_commit(self) -> str:
        """Read a standard .git directory without following links or running Git."""
        try:
            git_fd = os.open(".git", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self._root)
        except OSError:
            return ""
        try:
            head = self._read_git_file(git_fd, "HEAD", 4096).decode("ascii").strip()
            if re.fullmatch(r"[0-9a-fA-F]{40}", head):
                return head.lower()
            match = re.fullmatch(r"ref: (refs/(?:heads|tags)/[A-Za-z0-9._/-]{1,512})", head)
            if not match or any(part in {"", ".", ".."} for part in match.group(1).split("/")):
                return ""
            reference = match.group(1)
            try:
                value = self._read_git_file(git_fd, reference, 4096).decode("ascii").strip()
            except OSError:
                value = ""
            if re.fullmatch(r"[0-9a-fA-F]{40}", value):
                return value.lower()
            try:
                packed = self._read_git_file(git_fd, "packed-refs", 1024 * 1024).decode("ascii")
            except (OSError, UnicodeDecodeError):
                return ""
            for line in packed.splitlines():
                parts = line.split(" ", 1)
                if len(parts) == 2 and parts[1] == reference and re.fullmatch(r"[0-9a-fA-F]{40}", parts[0]):
                    return parts[0].lower()
            return ""
        except (OSError, UnicodeDecodeError):
            return ""
        finally:
            os.close(git_fd)

    @staticmethod
    def _read_git_file(git_fd: int, path: str, max_bytes: int) -> bytes:
        current = os.dup(git_fd)
        try:
            parts = path.split("/")
            for segment in parts[:-1]:
                child = os.open(segment, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
                os.close(current)
                current = child
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
                    raise OSError("unsafe_git_metadata")
                return os.read(descriptor, max_bytes + 1)
            finally:
                os.close(descriptor)
        finally:
            os.close(current)

    def tree(self, limit: int = 200) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []
        visited = 0
        truncated = False
        exclusions_truncated = False

        def exclude(path: str, reason: str) -> None:
            nonlocal truncated, exclusions_truncated
            if path == "[TREE_REMAINDER]" and any(item["path"] == path for item in excluded):
                return
            if len(excluded) < 200:
                excluded.append({"path": path, "reason": reason})
            elif path == "[TREE_REMAINDER]":
                excluded[-1] = {"path": path, "reason": reason}
                truncated = True
                exclusions_truncated = True
            else:
                truncated = True
                exclusions_truncated = True

        def walk(fd: int, prefix: str, depth: int) -> None:
            nonlocal visited, truncated
            if depth > 8:
                truncated = True
                exclude("[TREE_REMAINDER]", "depth_limit")
                return
            # scandir is streamed: no unrestricted list of directory entries.
            with os.scandir(fd) as entries:
                for entry in entries:
                    visited += 1
                    if visited > 4000:
                        truncated = True
                        exclude("[TREE_REMAINDER]", "tree_visit_limit")
                        break
                    if len(files) >= limit:
                        truncated = True
                        exclude("[TREE_REMAINDER]", "tree_file_limit")
                        break
                    path = f"{prefix}{entry.name}"
                    if entry.name == ".git":
                        exclude(path, "git_metadata")
                        continue
                    if sensitive_path(path):
                        exclude(path, "sensitive_path")
                        continue
                    if entry.name in EXCLUDED_DIRECTORIES:
                        exclude(path, "excluded_directory")
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
                        elif stat.S_ISLNK(info.st_mode):
                            exclude(path, "symlink")
                            truncated = True
                        else:
                            exclude(path, "special_file")
                            truncated = True
                    except (OSError, RepositoryAccessError):
                        truncated = True
                        exclude(path, "unsafe_or_unavailable_entry")

        walk(self._root, "", 0)
        return {"files": sorted(files, key=lambda item: item["path"]), "truncated": truncated,
                "excluded": sorted(excluded, key=lambda item: (item["path"], item["reason"])),
                "exclusions_truncated": exclusions_truncated}

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
