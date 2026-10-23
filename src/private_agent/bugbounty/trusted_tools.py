"""Fail-closed provenance verification for ProjectDiscovery binaries.

This module never downloads or executes a scanner. It verifies a locally provisioned
release archive only after an OS-level gpgv check validates the official checksum file.
The caller must provision the keyring and signed files through an out-of-band trusted
process; arbitrary hashes and environment flags are not accepted as provenance.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit


_TOOL_REPOSITORIES = {"httpx": "httpx", "nuclei": "nuclei"}
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_FINGERPRINT_RE = re.compile(r"^[0-9A-Fa-f]{40}$")
_CHECKSUM_LINE_RE = re.compile(r"^\s*([0-9a-fA-F]{64})\s+\*?([^\s]+)\s*$")


class ToolVerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class TrustedToolInstallSpec:
    tool: str
    version: str
    binary_path: Path
    archive_path: Path
    archive_member: str
    release_asset_name: str
    release_url: str
    checksum_file: Path
    signature_file: Path
    keyring_path: Path
    signer_fingerprint: str
    gpgv_path: Path = Path("/usr/bin/gpgv")

    def __post_init__(self) -> None:
        if self.tool not in _TOOL_REPOSITORIES:
            raise ValueError("trusted_tool_unsupported")
        if not isinstance(self.version, str) or not _VERSION_RE.fullmatch(self.version):
            raise ValueError("trusted_tool_version_invalid")
        if not isinstance(self.signer_fingerprint, str) or not _FINGERPRINT_RE.fullmatch(self.signer_fingerprint):
            raise ValueError("trusted_tool_signer_fingerprint_invalid")
        for value, error in (
            (self.binary_path, "trusted_tool_binary_path_invalid"),
            (self.archive_path, "trusted_tool_archive_path_invalid"),
            (self.checksum_file, "trusted_tool_checksum_path_invalid"),
            (self.signature_file, "trusted_tool_signature_path_invalid"),
            (self.keyring_path, "trusted_tool_keyring_path_invalid"),
            (self.gpgv_path, "trusted_tool_gpgv_path_invalid"),
        ):
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(error)
        if self.gpgv_path not in {Path("/usr/bin/gpgv"), Path("/bin/gpgv")}:
            raise ValueError("trusted_tool_gpgv_path_untrusted")
        if not isinstance(self.archive_member, str) or not self.archive_member or Path(self.archive_member).is_absolute() or ".." in Path(self.archive_member).parts:
            raise ValueError("trusted_tool_archive_member_invalid")
        if not isinstance(self.release_asset_name, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", self.release_asset_name):
            raise ValueError("trusted_tool_asset_name_invalid")
        expected_url = f"https://github.com/projectdiscovery/{self.tool}/releases/download/v{self.version}/{self.release_asset_name}"
        parsed = urlsplit(self.release_url)
        if self.release_url != expected_url or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("trusted_tool_release_url_invalid")


@dataclass(frozen=True, slots=True)
class ToolVerificationResult:
    status: ToolVerificationStatus
    tool: str
    version: str
    reason: str
    artifact_sha256: str = ""
    verification_method: str = ""

    @property
    def verified(self) -> bool:
        return self.status is ToolVerificationStatus.VERIFIED


def _secure_regular_file(path: Path, *, executable: bool = False) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return False
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        return False
    if executable and not os.access(path, os.X_OK):
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed_checksum(spec: TrustedToolInstallSpec) -> tuple[bool, str]:
    if not _secure_regular_file(spec.gpgv_path, executable=True):
        return False, "gpgv_unavailable"
    if not _secure_regular_file(spec.checksum_file) or not _secure_regular_file(spec.signature_file) or not _secure_regular_file(spec.keyring_path):
        return False, "signed_provenance_files_unavailable"
    try:
        completed = subprocess.run(
            [
                str(spec.gpgv_path),
                "--status-fd",
                "1",
                "--keyring",
                str(spec.keyring_path),
                str(spec.signature_file),
                str(spec.checksum_file),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            close_fds=True,
            timeout=10.0,
            check=False,
            env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "signature_verification_failed"
    if completed.returncode != 0:
        return False, "signature_verification_failed"
    status_output = completed.stdout.decode("utf-8", errors="replace") if isinstance(completed.stdout, bytes) else str(completed.stdout or "")
    valid_signers = {
        parts[2].upper()
        for line in status_output.splitlines()
        if (parts := line.split()) and len(parts) >= 3 and parts[0] == "[GNUPG:]" and parts[1] == "VALIDSIG"
    }
    if spec.signer_fingerprint.upper() not in valid_signers:
        return False, "signature_signer_untrusted"
    return True, ""


def _checksum_for_asset(spec: TrustedToolInstallSpec) -> tuple[str | None, str]:
    try:
        lines = spec.checksum_file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None, "checksum_file_invalid"
    matches: list[str] = []
    for line in lines:
        match = _CHECKSUM_LINE_RE.fullmatch(line)
        if match and match.group(2) == spec.release_asset_name:
            matches.append(match.group(1).lower())
    if len(matches) != 1 or not _SHA256_RE.fullmatch(matches[0]):
        return None, "release_asset_checksum_missing_or_ambiguous"
    return matches[0], ""


def _binary_matches_signed_archive(spec: TrustedToolInstallSpec) -> tuple[bool, str]:
    if not _secure_regular_file(spec.archive_path) or not _secure_regular_file(spec.binary_path, executable=True):
        return False, "trusted_artifact_file_unavailable"
    try:
        with zipfile.ZipFile(spec.archive_path) as archive:
            infos = [info for info in archive.infolist() if info.filename == spec.archive_member]
            if len(infos) != 1 or infos[0].is_dir():
                return False, "archive_member_missing_or_ambiguous"
            info = infos[0]
            if info.filename.startswith("/") or ".." in Path(info.filename).parts:
                return False, "archive_member_path_invalid"
            if archive.read(info) != spec.binary_path.read_bytes():
                return False, "binary_archive_content_mismatch"
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return False, "trusted_archive_invalid"
    return True, ""


def _preflight_artifacts(spec: TrustedToolInstallSpec) -> str | None:
    """Reject unsafe or absent local artifacts before invoking any verifier."""
    try:
        binary_info = spec.binary_path.lstat()
    except OSError:
        return "artifact_missing"
    if stat.S_ISLNK(binary_info.st_mode):
        return "binary_symlink_rejected"
    if not _secure_regular_file(spec.binary_path, executable=True):
        return "artifact_missing"
    for path in (spec.archive_path, spec.checksum_file, spec.signature_file, spec.keyring_path):
        try:
            info = path.lstat()
        except OSError:
            return "artifact_missing"
        if stat.S_ISLNK(info.st_mode) or not _secure_regular_file(path):
            return "artifact_missing"
    return None


def verify_tool_installation(spec: TrustedToolInstallSpec) -> ToolVerificationResult:
    """Verify publisher-signed release metadata and the exact extracted binary."""
    try:
        preflight_reason = _preflight_artifacts(spec)
        if preflight_reason:
            return ToolVerificationResult(ToolVerificationStatus.UNAVAILABLE, spec.tool, spec.version, preflight_reason)
        signed, reason = _signed_checksum(spec)
        if not signed:
            return ToolVerificationResult(ToolVerificationStatus.UNAVAILABLE, spec.tool, spec.version, reason)
        expected, reason = _checksum_for_asset(spec)
        if expected is None:
            return ToolVerificationResult(ToolVerificationStatus.INVALID, spec.tool, spec.version, reason)
        try:
            actual = _sha256(spec.archive_path)
        except OSError:
            return ToolVerificationResult(ToolVerificationStatus.UNAVAILABLE, spec.tool, spec.version, "trusted_artifact_file_unavailable")
        if actual.lower() != expected.lower():
            return ToolVerificationResult(ToolVerificationStatus.INVALID, spec.tool, spec.version, "release_asset_checksum_mismatch", actual)
        matches, reason = _binary_matches_signed_archive(spec)
        if not matches:
            return ToolVerificationResult(ToolVerificationStatus.INVALID, spec.tool, spec.version, reason, actual)
        return ToolVerificationResult(ToolVerificationStatus.VERIFIED, spec.tool, spec.version, "", actual, "gpgv_signed_release_checksum")
    except (TypeError, ValueError):
        return ToolVerificationResult(ToolVerificationStatus.INVALID, spec.tool, spec.version, "trusted_tool_spec_invalid")


def load_tool_spec_from_environment(tool: str, binary_path: str) -> TrustedToolInstallSpec | None:
    """Load only a complete pinned spec; incomplete environment config fails closed."""
    prefix = f"AGENT_{tool.upper()}_"
    values = {
        "version": os.getenv(prefix + "VERSION"),
        "archive_path": os.getenv(prefix + "ARCHIVE_PATH"),
        "archive_member": os.getenv(prefix + "ARCHIVE_MEMBER"),
        "asset_name": os.getenv(prefix + "ASSET_NAME"),
        "checksum_file": os.getenv(prefix + "CHECKSUM_FILE"),
        "signature_file": os.getenv(prefix + "SIGNATURE_FILE"),
        "keyring_path": os.getenv(prefix + "KEYRING_PATH"),
        "signer_fingerprint": os.getenv(prefix + "SIGNER_FINGERPRINT"),
        "release_url": os.getenv(prefix + "RELEASE_URL"),
    }
    if not all(isinstance(value, str) and value for value in values.values()):
        return None
    try:
        return TrustedToolInstallSpec(
            tool=tool,
            version=values["version"],
            binary_path=Path(binary_path),
            archive_path=Path(values["archive_path"]),
            archive_member=values["archive_member"],
            release_asset_name=values["asset_name"],
            release_url=values["release_url"],
            checksum_file=Path(values["checksum_file"]),
            signature_file=Path(values["signature_file"]),
            keyring_path=Path(values["keyring_path"]),
            signer_fingerprint=values["signer_fingerprint"],
        )
    except (TypeError, ValueError):
        return None


__all__ = [
    "ToolVerificationResult",
    "ToolVerificationStatus",
    "TrustedToolInstallSpec",
    "load_tool_spec_from_environment",
    "verify_tool_installation",
]
