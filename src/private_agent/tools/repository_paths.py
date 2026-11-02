"""Shared fail-closed path classification for repository source adapters."""
from __future__ import annotations


_DATABASE_EXTENSIONS = (".sqlite", ".sqlite3", ".db", ".db3")
_DATABASE_SIDECARS = ("-journal", "-wal", "-shm")


def sensitive_path(path: str) -> bool:
    """Return true for common credential, database, and log paths.

    This is defense in depth rather than a general secret detector. Invalid
    path values are treated as sensitive; callers still perform canonical path
    validation before using a path for I/O.
    """
    if not isinstance(path, str):
        return True
    parts = path.lower().split("/")
    for part in parts:
        if part.startswith(".env") or part in {
            ".git", ".ssh", ".aws", ".azure", ".kube", ".netrc", ".npmrc", ".pypirc", ".git-credentials",
            ".secret", ".secrets", "id_rsa", "id_ed25519", "credentials", "secret", "secrets",
        }:
            return True
        if part.endswith((".pem", ".key", ".p12", ".pfx")):
            return True
        if any(part.endswith(extension) for extension in _DATABASE_EXTENSIONS):
            return True
        if any(part.endswith(extension + sidecar) for extension in _DATABASE_EXTENSIONS for sidecar in _DATABASE_SIDECARS):
            return True
        if part.endswith((".wal", ".shm", ".log")) or ".log." in part:
            return True
        if part.startswith(("credentials.", "secret.", "secrets.", ".secret.", ".secrets.")):
            return True
    return False
