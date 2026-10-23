from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from private_agent.storage import Store, now


DEFAULT_OWNER_EMAIL = "ashbm1995@gmail.com"
SESSION_COOKIE_NAME = "vulnora_session"
CSRF_COOKIE_NAME = "vulnora_csrf"
DEFAULT_SESSION_TTL_SECONDS = 8 * 60 * 60
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 1024
MAX_SESSION_TOKEN_LENGTH = 256
_EMAIL_PATTERN = re.compile(r"^[^@\s]{1,128}@[^@\s]{1,255}$")

# scrypt is provided by Python's standard library/OpenSSL and is a trusted,
# memory-hard password KDF equivalent in purpose to Argon2id.  The encoded
# format includes parameters so future upgrades can be introduced explicitly.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SCRYPT_MAXMEM = 64 * 1024 * 1024
_DUMMY_SALT = b"vulnora-auth-dummy-salt-v1"

ROLE_OWNER = "OWNER"
ROLE_OPERATOR = "OPERATOR"
ROLE_VIEWER = "VIEWER"
ROLES = frozenset({ROLE_OWNER, ROLE_OPERATOR, ROLE_VIEWER})
ACTIVE = "ACTIVE"
DISABLED = "DISABLED"


class AuthError(RuntimeError):
    def __init__(self, code: str, message: str = "Authentication failed") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class AuthUser:
    user_id: str
    email: str
    role: str
    status: str
    created_at: str
    updated_at: str
    last_login_at: str


@dataclass(frozen=True)
class AuthPrincipal:
    subject: str
    user_id: str
    email: str
    role: str
    session_id: str = ""
    auth_type: str = "session"

    @property
    def is_legacy(self) -> bool:
        return self.auth_type == "legacy_bearer"


@dataclass(frozen=True)
class AuthLogin:
    user: AuthUser
    session_token: str
    csrf_token: str
    expires_at: str


def normalize_email(value: str) -> str:
    if not isinstance(value, str):
        raise AuthError("invalid_email", "A valid email address is required")
    normalized = value.strip().lower()
    if not _EMAIL_PATTERN.fullmatch(normalized):
        raise AuthError("invalid_email", "A valid email address is required")
    return normalized


def validate_password(value: str) -> str:
    if not isinstance(value, str) or len(value) < MIN_PASSWORD_LENGTH or len(value) > MAX_PASSWORD_LENGTH:
        raise AuthError("invalid_password", f"Password must be {MIN_PASSWORD_LENGTH}-{MAX_PASSWORD_LENGTH} characters")
    if any(ord(char) < 32 for char in value):
        raise AuthError("invalid_password", "Password contains unsupported control characters")
    return value


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _scrypt_digest(password: str, salt: bytes, *, n: int = _SCRYPT_N, r: int = _SCRYPT_R, p: int = _SCRYPT_P) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        maxmem=_SCRYPT_MAXMEM,
        dklen=_SCRYPT_DKLEN,
    )


def hash_password(password: str) -> str:
    password = validate_password(password)
    salt = secrets.token_bytes(16)
    digest = _scrypt_digest(password, salt)
    return f"scrypt$n={_SCRYPT_N},r={_SCRYPT_R},p={_SCRYPT_P}${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    if not isinstance(password, str) or not isinstance(encoded, str):
        return False
    try:
        algorithm, parameters, salt_text, digest_text = encoded.split("$", 3)
        if algorithm != "scrypt":
            return False
        parsed: dict[str, int] = {}
        for item in parameters.split(","):
            key, value = item.split("=", 1)
            parsed[key] = int(value)
        n = parsed["n"]
        r = parsed["r"]
        p = parsed["p"]
        if n < 2**12 or n > 2**20 or r < 1 or r > 32 or p < 1 or p > 16:
            return False
        salt = _b64decode(salt_text)
        expected = _b64decode(digest_text)
        actual = _scrypt_digest(password, salt, n=n, r=r, p=p)
        return hmac.compare_digest(actual, expected)
    except (binascii.Error, KeyError, ValueError, TypeError):
        return False


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _remote_key(remote: str | None) -> str:
    if not remote:
        return ""
    return hashlib.sha256(str(remote).encode("utf-8")).hexdigest()[:32]


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_json_metadata(value: dict[str, Any] | None) -> str:
    if not isinstance(value, dict):
        return "{}"
    sanitized = {str(key): str(item)[:200] for key, item in list(value.items())[:20] if key not in {"password", "token", "session", "csrf"}}
    return json.dumps(sanitized, sort_keys=True, ensure_ascii=False)


class AuthManager:
    def __init__(
        self,
        store: Store,
        *,
        owner_email: str | None = None,
        session_ttl_seconds: int | None = None,
        max_failed_attempts: int | None = None,
        lockout_seconds: int | None = None,
        max_login_attempts_per_window: int | None = None,
        login_rate_window_seconds: int | None = None,
        max_concurrent_kdfs: int | None = None,
    ) -> None:
        self.store = store
        self.owner_email = normalize_email(owner_email or os.getenv("AGENT_OWNER_EMAIL", DEFAULT_OWNER_EMAIL))
        self.session_ttl_seconds = int(session_ttl_seconds or os.getenv("AGENT_SESSION_TTL_SECONDS", str(DEFAULT_SESSION_TTL_SECONDS)))
        self.max_failed_attempts = int(max_failed_attempts or os.getenv("AGENT_AUTH_MAX_FAILED_ATTEMPTS", "5"))
        self.lockout_seconds = int(lockout_seconds or os.getenv("AGENT_AUTH_LOCKOUT_SECONDS", "300"))
        self.max_login_attempts_per_window = int(max_login_attempts_per_window or os.getenv("AGENT_AUTH_MAX_LOGIN_ATTEMPTS", "20"))
        self.login_rate_window_seconds = int(login_rate_window_seconds or os.getenv("AGENT_AUTH_RATE_WINDOW_SECONDS", "60"))
        kdf_slots = int(max_concurrent_kdfs or os.getenv("AGENT_AUTH_MAX_CONCURRENT_KDFS", "2"))
        if not 300 <= self.session_ttl_seconds <= 7 * 24 * 60 * 60:
            raise ValueError("session_ttl_invalid")
        if not 3 <= self.max_failed_attempts <= 20:
            raise ValueError("auth_max_failed_attempts_invalid")
        if not 30 <= self.lockout_seconds <= 24 * 60 * 60:
            raise ValueError("auth_lockout_invalid")
        if not 3 <= self.max_login_attempts_per_window <= 100:
            raise ValueError("auth_login_rate_limit_invalid")
        if not 10 <= self.login_rate_window_seconds <= 60 * 60:
            raise ValueError("auth_login_rate_window_invalid")
        if not 1 <= kdf_slots <= 8:
            raise ValueError("auth_kdf_concurrency_invalid")
        self._kdf_slots = threading.BoundedSemaphore(kdf_slots)

    def _user_from_row(self, row: Any) -> AuthUser:
        return AuthUser(
            user_id=str(row["user_id"]),
            email=str(row["email"]),
            role=str(row["role"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_login_at=str(row["last_login_at"]),
        )

    def _record_event(self, event_type: str, *, user_id: str = "", email: str = "", remote: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        self.store.db.execute(
            """INSERT INTO auth_events
               (event_id, event_type, user_id, email, created_at, remote_key, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (f"auth:{uuid.uuid4()}", event_type, user_id[:256], email[:256], now(), _remote_key(remote), _safe_json_metadata(metadata)),
        )

    def owner_initialized(self) -> bool:
        return self.owner_status() == "configured"

    def owner_status(self) -> str:
        with self.store._db_lock:
            row = self.store.db.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN role = ? THEN 1 ELSE 0 END) AS owners,
                          SUM(CASE WHEN role = ? AND status = ? THEN 1 ELSE 0 END) AS active_owners
                   FROM auth_users""",
                (ROLE_OWNER, ROLE_OWNER, ACTIVE),
            ).fetchone()
        total = int(row["total"] if row else 0)
        owners = int(row["owners"] or 0) if row else 0
        active_owners = int(row["active_owners"] or 0) if row else 0
        if total == 0:
            return "missing"
        return "configured" if owners == 1 and active_owners == 1 else "invalid"

    def initialize_owner(self, email: str, password: str, *, remote: str | None = None) -> AuthUser:
        normalized = normalize_email(email)
        if normalized != self.owner_email:
            raise AuthError("owner_email_not_allowed", "The configured owner email does not match")
        password_hash = hash_password(password)
        stamp = now()
        user_id = f"user:{uuid.uuid4()}"
        with self.store._db_lock, self.store.db:
            existing = self.store.db.execute("SELECT COUNT(*) AS count FROM auth_users").fetchone()
            if existing and int(existing["count"]) != 0:
                raise AuthError("owner_already_initialized", "The initial owner can only be initialized once")
            self.store.db.execute(
                """INSERT INTO auth_users
                   (user_id, email, role, password_hash, status, failed_login_count,
                    locked_until, last_login_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 0, '', '', ?, ?)""",
                (user_id, normalized, ROLE_OWNER, password_hash, ACTIVE, stamp, stamp),
            )
            self._record_event("OWNER_INITIALIZED", user_id=user_id, email=normalized, remote=remote)
            row = self.store.db.execute("SELECT * FROM auth_users WHERE user_id = ?", (user_id,)).fetchone()
        return self._user_from_row(row)

    def _find_user(self, email: str) -> Any:
        return self.store.db.execute("SELECT * FROM auth_users WHERE email = ?", (email,)).fetchone()

    def _record_login_attempt_or_raise(self, email: str, remote: str | None) -> None:
        """Atomically cap expensive password work by a privacy-preserving remote key."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=self.login_rate_window_seconds)).isoformat()
        remote_key = _remote_key(remote)
        with self.store._db_lock, self.store.db:
            row = self.store.db.execute(
                """SELECT COUNT(*) AS count FROM auth_events
                   WHERE event_type = 'LOGIN_ATTEMPT' AND created_at >= ? AND remote_key = ?""",
                (cutoff, remote_key),
            ).fetchone()
            if row is not None and int(row["count"]) >= self.max_login_attempts_per_window:
                self._record_event("LOGIN_RATE_LIMITED", email=email, remote=remote)
                raise AuthError("login_rate_limited", "Too many login attempts. Try again later")
            self._record_event("LOGIN_ATTEMPT", email=email, remote=remote)

    def login(self, email: str, password: str, *, remote: str | None = None) -> AuthLogin:
        normalized = normalize_email(email)
        password = validate_password(password)
        self._record_login_attempt_or_raise(normalized, remote)
        with self.store._db_lock:
            row = self._find_user(normalized)
            locked = False
            if row is not None and row["locked_until"]:
                try:
                    locked = _parse_time(str(row["locked_until"])) > datetime.now(timezone.utc)
                except (TypeError, ValueError) as exc:
                    raise AuthError("auth_state_invalid", "Authentication state is unavailable") from exc
            encoded = str(row["password_hash"]) if row else ""
        if not self._kdf_slots.acquire(blocking=False):
            raise AuthError("login_rate_limited", "Too many login attempts. Try again later")
        try:
            if locked:
                # Keep the failure path close to the unknown-user cost without
                # extending the account lock on every subsequent attempt.
                _scrypt_digest(password, _DUMMY_SALT)
                password_ok = False
            elif encoded:
                password_ok = verify_password(password, encoded)
            else:
                _scrypt_digest(password, _DUMMY_SALT)
                password_ok = False
        finally:
            self._kdf_slots.release()
        if locked or not password_ok or row is None or str(row["status"]) != ACTIVE:
            with self.store._db_lock, self.store.db:
                current = self._find_user(normalized)
                if current is not None and str(current["status"]) == ACTIVE:
                    if locked:
                        self._record_event("LOGIN_LOCKED", user_id=str(current["user_id"]), email=normalized, remote=remote)
                    else:
                        failed = int(current["failed_login_count"]) + 1
                        lock_until = ""
                        if failed >= self.max_failed_attempts:
                            lock_until = (datetime.now(timezone.utc) + timedelta(seconds=self.lockout_seconds)).isoformat()
                        self.store.db.execute(
                            "UPDATE auth_users SET failed_login_count = ?, locked_until = ?, updated_at = ? WHERE user_id = ?",
                            (failed, lock_until, now(), str(current["user_id"])),
                        )
                        self._record_event("LOGIN_LOCKED" if lock_until else "LOGIN_FAILED", user_id=str(current["user_id"]), email=normalized, remote=remote)
                else:
                    self._record_event("LOGIN_FAILED", email=normalized, remote=remote)
            raise AuthError("invalid_credentials", "Invalid email or password")

        session_token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        stamp = now()
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=self.session_ttl_seconds)).isoformat()
        session_hash = _hash_token(session_token)
        csrf_hash = _hash_token(csrf_token)
        with self.store._db_lock, self.store.db:
            current = self._find_user(normalized)
            if current is None or str(current["status"]) != ACTIVE:
                raise AuthError("invalid_credentials", "Invalid email or password")
            self.store.db.execute(
                """UPDATE auth_users SET failed_login_count = 0, locked_until = '', last_login_at = ?, updated_at = ?
                   WHERE user_id = ?""",
                (stamp, stamp, str(current["user_id"])),
            )
            self.store.db.execute(
                """INSERT INTO auth_sessions
                   (session_hash, user_id, csrf_hash, created_at, expires_at, last_seen_at, revoked_at)
                   VALUES (?, ?, ?, ?, ?, ?, '')""",
                (session_hash, str(current["user_id"]), csrf_hash, stamp, expires_at, stamp),
            )
            self._record_event("LOGIN_SUCCEEDED", user_id=str(current["user_id"]), email=normalized, remote=remote)
            updated = self.store.db.execute("SELECT * FROM auth_users WHERE user_id = ?", (str(current["user_id"]),)).fetchone()
        return AuthLogin(self._user_from_row(updated), session_token, csrf_token, expires_at)

    def authenticate_session(self, session_token: str | None) -> AuthPrincipal:
        if not isinstance(session_token, str) or not session_token or len(session_token) > MAX_SESSION_TOKEN_LENGTH:
            raise AuthError("unauthenticated", "Authentication is required")
        with self.store._db_lock, self.store.db:
            row = self.store.db.execute(
                """SELECT s.session_hash, s.user_id, s.expires_at, s.revoked_at,
                          u.email, u.role, u.status
                   FROM auth_sessions s JOIN auth_users u ON u.user_id = s.user_id
                   WHERE s.session_hash = ?""",
                (_hash_token(session_token),),
            ).fetchone()
            if row is None or row["revoked_at"] or str(row["status"]) != ACTIVE:
                raise AuthError("unauthenticated", "Authentication is required")
            try:
                expired = _parse_time(str(row["expires_at"])) <= datetime.now(timezone.utc)
            except (TypeError, ValueError):
                expired = True
            if expired:
                raise AuthError("session_expired", "Session expired")
            self.store.db.execute("UPDATE auth_sessions SET last_seen_at = ? WHERE session_hash = ?", (now(), str(row["session_hash"])))
        return AuthPrincipal(
            subject=str(row["user_id"]),
            user_id=str(row["user_id"]),
            email=str(row["email"]),
            role=str(row["role"]),
            session_id=str(row["session_hash"]),
        )

    def verify_csrf(self, principal: AuthPrincipal, token: str | None) -> None:
        if principal.is_legacy:
            return
        if not token or len(token) > MAX_SESSION_TOKEN_LENGTH:
            raise AuthError("csrf_failed", "CSRF validation failed")
        with self.store._db_lock:
            row = self.store.db.execute(
                "SELECT csrf_hash FROM auth_sessions WHERE session_hash = ? AND user_id = ? AND revoked_at = ''",
                (principal.session_id, principal.user_id),
            ).fetchone()
        if row is None or not hmac.compare_digest(str(row["csrf_hash"]), _hash_token(token)):
            raise AuthError("csrf_failed", "CSRF validation failed")

    def session_expiry(self, principal: AuthPrincipal) -> str:
        if principal.is_legacy or not principal.session_id:
            return ""
        with self.store._db_lock:
            row = self.store.db.execute(
                "SELECT expires_at FROM auth_sessions WHERE session_hash = ? AND user_id = ? AND revoked_at = ''",
                (principal.session_id, principal.user_id),
            ).fetchone()
        if row is None:
            raise AuthError("unauthenticated", "Authentication is required")
        return str(row["expires_at"])

    def revoke_session(self, principal: AuthPrincipal) -> None:
        if principal.is_legacy or not principal.session_id:
            return
        with self.store._db_lock, self.store.db:
            self.store.db.execute(
                "UPDATE auth_sessions SET revoked_at = ? WHERE session_hash = ? AND user_id = ? AND revoked_at = ''",
                (now(), principal.session_id, principal.user_id),
            )
            self._record_event("LOGOUT", user_id=principal.user_id, email=principal.email)

    def list_users(self) -> list[AuthUser]:
        with self.store._db_lock:
            rows = self.store.db.execute("SELECT * FROM auth_users ORDER BY created_at, user_id").fetchall()
        return [self._user_from_row(row) for row in rows]

    def create_user(self, email: str, password: str, role: str, *, actor_user_id: str = "") -> AuthUser:
        normalized = normalize_email(email)
        if role not in {ROLE_OPERATOR, ROLE_VIEWER}:
            raise AuthError("role_not_allowed", "Only OPERATOR or VIEWER accounts can be created")
        password_hash = hash_password(password)
        stamp = now()
        user_id = f"user:{uuid.uuid4()}"
        with self.store._db_lock, self.store.db:
            if self._find_user(normalized) is not None:
                raise AuthError("user_exists", "A user with this email already exists")
            self.store.db.execute(
                """INSERT INTO auth_users
                   (user_id, email, role, password_hash, status, failed_login_count,
                    locked_until, last_login_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 0, '', '', ?, ?)""",
                (user_id, normalized, role, password_hash, ACTIVE, stamp, stamp),
            )
            self._record_event("USER_CREATED", user_id=user_id, email=normalized, metadata={"role": role, "actor_user_id": actor_user_id})
            row = self.store.db.execute("SELECT * FROM auth_users WHERE user_id = ?", (user_id,)).fetchone()
        return self._user_from_row(row)

    def set_role(self, user_id: str, role: str, *, actor_user_id: str = "") -> AuthUser:
        if role not in {ROLE_OPERATOR, ROLE_VIEWER}:
            raise AuthError("role_not_allowed", "An existing account cannot be assigned OWNER")
        with self.store._db_lock, self.store.db:
            row = self.store.db.execute("SELECT * FROM auth_users WHERE user_id = ?", (user_id,)).fetchone()
            if row is None:
                raise AuthError("user_not_found", "User was not found")
            if str(row["role"]) == ROLE_OWNER:
                raise AuthError("owner_immutable", "The initial owner role cannot be changed")
            stamp = now()
            self.store.db.execute("UPDATE auth_users SET role = ?, updated_at = ? WHERE user_id = ?", (role, stamp, user_id))
            self._record_event("ROLE_CHANGED", user_id=user_id, email=str(row["email"]), metadata={"role": role, "actor_user_id": actor_user_id})
            updated = self.store.db.execute("SELECT * FROM auth_users WHERE user_id = ?", (user_id,)).fetchone()
        return self._user_from_row(updated)

    def disable_user(self, user_id: str, *, actor_user_id: str = "") -> AuthUser:
        with self.store._db_lock, self.store.db:
            row = self.store.db.execute("SELECT * FROM auth_users WHERE user_id = ?", (user_id,)).fetchone()
            if row is None:
                raise AuthError("user_not_found", "User was not found")
            if str(row["role"]) == ROLE_OWNER:
                raise AuthError("owner_immutable", "The initial owner cannot be disabled")
            stamp = now()
            self.store.db.execute("UPDATE auth_users SET status = ?, updated_at = ? WHERE user_id = ?", (DISABLED, stamp, user_id))
            self.store.db.execute("UPDATE auth_sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at = ''", (stamp, user_id))
            self._record_event("USER_DISABLED", user_id=user_id, email=str(row["email"]), metadata={"actor_user_id": actor_user_id})
            updated = self.store.db.execute("SELECT * FROM auth_users WHERE user_id = ?", (user_id,)).fetchone()
        return self._user_from_row(updated)


__all__ = [
    "ACTIVE",
    "AuthError",
    "AuthLogin",
    "AuthManager",
    "AuthPrincipal",
    "AuthUser",
    "CSRF_COOKIE_NAME",
    "DEFAULT_OWNER_EMAIL",
    "DISABLED",
    "ROLE_OPERATOR",
    "ROLE_OWNER",
    "ROLE_VIEWER",
    "ROLES",
    "SESSION_COOKIE_NAME",
    "hash_password",
    "normalize_email",
    "validate_password",
    "verify_password",
]
