from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from private_agent.bugbounty.models import ScopeDecision, ScopeDefinition, safe_json, stable_digest
from private_agent.storage import Store, now


MAX_WORKER_ID = 128
MAX_NONCE = 128
MAX_SIGNATURE = 128
MAX_RESULT_BYTES = 64 * 1024
MAX_OBSERVATIONS = 100
MAX_FINDINGS = 50
MAX_ERRORS = 50
MAX_TEXT = 2048
_CLOCK_SKEW_SECONDS = 300
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class WorkerProtocolError(RuntimeError):
    """A fail-closed protocol or lease error safe to expose as a code."""

    def __init__(self, code: str) -> None:
        self.code = str(code)
        super().__init__(self.code)


class AuthenticationReplay(WorkerProtocolError):
    def __init__(self) -> None:
        super().__init__("worker_request_replay")


@dataclass(frozen=True)
class WorkerAuthentication:
    worker_id: str
    credential_version: int
    nonce: str
    request_digest: str


class RemoteWorkerProtocol:
    """Durable control-plane protocol for an isolated remote worker.

    This class deliberately does not launch processes, scanners, containers, or
    network clients. It authenticates a future worker, issues bounded envelopes,
    and persists lease/result state. Active jobs require a separately verified
    runtime acceptance record; fixture jobs exist only for deterministic tests.
    """

    def __init__(
        self,
        store: Store,
        *,
        clock: Callable[[], datetime] | None = None,
        lease_seconds: int = 60,
        credential_grace_seconds: int = 300,
        clock_skew_seconds: int = _CLOCK_SKEW_SECONDS,
    ) -> None:
        if lease_seconds < 5 or lease_seconds > 3600:
            raise ValueError("worker_lease_seconds_invalid")
        if credential_grace_seconds < 0 or credential_grace_seconds > 3600:
            raise ValueError("worker_credential_grace_invalid")
        if clock_skew_seconds < 1 or clock_skew_seconds > 3600:
            raise ValueError("worker_clock_skew_invalid")
        self.store = store
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.lease_seconds = int(lease_seconds)
        self.credential_grace_seconds = int(credential_grace_seconds)
        self.clock_skew_seconds = int(clock_skew_seconds)

    @staticmethod
    def sign_request(
        *, credential: str, method: str, path: str, timestamp: int, nonce: str, body: Mapping[str, Any]
    ) -> str:
        if not isinstance(credential, str) or not credential:
            raise ValueError("worker_credential_required")
        canonical_body = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        body_digest = hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()
        message = f"{method.upper()}\n{path}\n{int(timestamp)}\n{nonce}\n{body_digest}".encode("utf-8")
        return hmac.new(credential.encode("utf-8"), message, hashlib.sha256).hexdigest()

    def _stamp(self) -> str:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _credential_digest(credential: str) -> str:
        return hashlib.sha256(credential.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_digest(value: str, field: str) -> str:
        if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
            raise ValueError(f"{field}_must_be_pinned_sha256")
        return value

    @staticmethod
    def _validate_worker_id(worker_id: str) -> str:
        if not isinstance(worker_id, str) or len(worker_id) > MAX_WORKER_ID or not _ID_RE.fullmatch(worker_id):
            raise ValueError("worker_id_invalid")
        return worker_id

    @staticmethod
    def _validate_identity(value: str, field: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 256 or any(ord(c) < 32 for c in value):
            raise ValueError(f"{field}_invalid")
        return value.strip()

    def register_worker(self, worker_id: str, *, runtime_identity: str, image_digest: str, manifest_digest: str) -> str:
        worker_id = self._validate_worker_id(worker_id)
        runtime_identity = self._validate_identity(runtime_identity, "runtime_identity")
        image_digest = self._validate_digest(image_digest, "image_digest")
        manifest_digest = self._validate_digest(manifest_digest, "manifest_digest")
        credential = secrets.token_urlsafe(32)
        stamp = self._stamp()
        with self.store._db_lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                self.store.db.execute(
                    """INSERT INTO remote_workers
                    (worker_id, credential_digest, credential_version, previous_credential_digest,
                     previous_credential_version, previous_credential_expires_at, status, runtime_identity,
                     image_digest, manifest_digest, acceptance_status, created_at, updated_at, last_seen_at)
                    VALUES (?, ?, 1, '', 0, '', 'ACTIVE', ?, ?, ?, 'UNVERIFIED', ?, ?, '')""",
                    (worker_id, self._credential_digest(credential), runtime_identity, image_digest, manifest_digest, stamp, stamp),
                )
                self.store.db.commit()
            except sqlite3.IntegrityError as exc:
                self.store.db.rollback()
                raise WorkerProtocolError("worker_already_registered") from exc
            except Exception:
                self.store.db.rollback()
                raise
        return credential

    def rotate_worker(self, worker_id: str) -> str:
        worker_id = self._validate_worker_id(worker_id)
        credential = secrets.token_urlsafe(32)
        stamp = self._stamp()
        previous_expires = (self._now() + timedelta(seconds=self.credential_grace_seconds)).isoformat()
        with self.store._db_lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.store.db.execute("SELECT * FROM remote_workers WHERE worker_id = ?", (worker_id,)).fetchone()
                if row is None or row["status"] == "REVOKED":
                    raise WorkerProtocolError("worker_not_active")
                self.store.db.execute(
                    """UPDATE remote_workers SET credential_digest = ?, credential_version = ?,
                       previous_credential_digest = ?, previous_credential_version = ?,
                       previous_credential_expires_at = ?, status = 'ACTIVE', updated_at = ?
                       WHERE worker_id = ?""",
                    (
                        self._credential_digest(credential), int(row["credential_version"]) + 1,
                        str(row["credential_digest"]), int(row["credential_version"]), previous_expires,
                        stamp, worker_id,
                    ),
                )
                self.store.db.commit()
            except Exception:
                self.store.db.rollback()
                raise
        return credential

    def revoke_worker(self, worker_id: str) -> None:
        worker_id = self._validate_worker_id(worker_id)
        with self.store._db_lock, self.store.db:
            updated = self.store.db.execute(
                "UPDATE remote_workers SET status = 'REVOKED', updated_at = ? WHERE worker_id = ? AND status <> 'REVOKED'",
                (self._stamp(), worker_id),
            )
            if updated.rowcount != 1:
                raise WorkerProtocolError("worker_not_active")

    def activate_worker(self, worker_id: str, runtime_proof: Any) -> None:
        """Persist acceptance only when issued by the trusted runtime adapter."""
        worker_id = self._validate_worker_id(worker_id)
        try:
            from private_agent.skills.sandbox_runtime import RuntimeActivationProof
        except ImportError as exc:
            raise WorkerProtocolError("runtime_acceptance_required") from exc
        if type(runtime_proof) is not RuntimeActivationProof or not runtime_proof.ready:
            raise WorkerProtocolError("runtime_acceptance_required")
        stamp = self._stamp()
        with self.store._db_lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                row = self._worker_row(worker_id)
                if (
                    runtime_proof.runtime_identity != str(row["runtime_identity"])
                    or runtime_proof.image_digest != str(row["image_digest"])
                    or runtime_proof.command_manifest_digest != str(row["manifest_digest"])
                ):
                    raise WorkerProtocolError("runtime_identity_mismatch")
                self.store.db.execute(
                    "UPDATE remote_workers SET acceptance_status = 'VERIFIED', updated_at = ? WHERE worker_id = ? AND status = 'ACTIVE'",
                    (stamp, worker_id),
                )
                self.store.db.commit()
            except Exception:
                self.store.db.rollback()
                raise

    def worker_record(self, worker_id: str) -> dict[str, Any]:
        with self.store._db_lock:
            row = self.store.db.execute(
                """SELECT worker_id, credential_version, previous_credential_version,
                   previous_credential_expires_at, status, runtime_identity, image_digest,
                   manifest_digest, acceptance_status, created_at, updated_at, last_seen_at
                   FROM remote_workers WHERE worker_id = ?""",
                (worker_id,),
            ).fetchone()
        if row is None:
            raise WorkerProtocolError("worker_not_found")
        return dict(row)

    def authenticate(
        self,
        *,
        worker_id: str,
        credential: str,
        method: str,
        path: str,
        timestamp: int,
        nonce: str,
        body: Mapping[str, Any],
        signature: str,
    ) -> WorkerAuthentication:
        worker_id = self._validate_worker_id(worker_id)
        if not isinstance(credential, str) or not credential or len(credential) > 256:
            raise WorkerProtocolError("worker_authentication_failed")
        if not isinstance(nonce, str) or not 1 <= len(nonce) <= MAX_NONCE:
            raise WorkerProtocolError("worker_nonce_invalid")
        if not isinstance(signature, str) or not 1 <= len(signature) <= MAX_SIGNATURE:
            raise WorkerProtocolError("worker_signature_invalid")
        if not isinstance(path, str) or not path.startswith("/") or len(path) > 512:
            raise WorkerProtocolError("worker_path_invalid")
        try:
            request_time = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        except (TypeError, ValueError, OverflowError) as exc:
            raise WorkerProtocolError("worker_timestamp_invalid") from exc
        if abs((self._now() - request_time).total_seconds()) > self.clock_skew_seconds:
            raise WorkerProtocolError("worker_timestamp_out_of_window")
        request_digest = hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        expected_signature = self.sign_request(
            credential=credential, method=method, path=path, timestamp=int(timestamp), nonce=nonce, body=body
        )
        with self.store._db_lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.store.db.execute("SELECT * FROM remote_workers WHERE worker_id = ?", (worker_id,)).fetchone()
                if row is None or row["status"] == "REVOKED":
                    raise WorkerProtocolError("worker_authentication_failed")
                digest = self._credential_digest(credential)
                credential_version = 0
                if hmac.compare_digest(digest, str(row["credential_digest"])):
                    credential_version = int(row["credential_version"])
                elif hmac.compare_digest(digest, str(row["previous_credential_digest"])):
                    expiry = str(row["previous_credential_expires_at"] or "")
                    if not expiry or datetime.fromisoformat(expiry) <= self._now():
                        raise WorkerProtocolError("worker_credential_expired")
                    credential_version = int(row["previous_credential_version"])
                else:
                    raise WorkerProtocolError("worker_authentication_failed")
                if not hmac.compare_digest(expected_signature, signature):
                    raise WorkerProtocolError("worker_authentication_failed")
                try:
                    self.store.db.execute(
                        "INSERT INTO remote_worker_nonces(worker_id, nonce, seen_at, request_digest) VALUES (?, ?, ?, ?)",
                        (worker_id, nonce, self._stamp(), request_digest),
                    )
                except sqlite3.IntegrityError as exc:
                    raise AuthenticationReplay() from exc
                self.store.db.execute(
                    "UPDATE remote_workers SET last_seen_at = ?, updated_at = ? WHERE worker_id = ?",
                    (self._stamp(), self._stamp(), worker_id),
                )
                self.store.db.commit()
                return WorkerAuthentication(worker_id, credential_version, nonce, request_digest)
            except Exception:
                self.store.db.rollback()
                raise

    def _require_auth(self, auth: WorkerAuthentication) -> None:
        if not isinstance(auth, WorkerAuthentication) or not auth.worker_id:
            raise WorkerProtocolError("worker_authentication_required")

    def _worker_row(self, worker_id: str) -> Any:
        row = self.store.db.execute("SELECT * FROM remote_workers WHERE worker_id = ?", (worker_id,)).fetchone()
        if row is None or row["status"] == "REVOKED":
            raise WorkerProtocolError("worker_not_active")
        return row

    def enqueue_fixture_job(self, *, worker_id: str, idempotency_key: str, fixture_name: str) -> dict[str, Any]:
        worker_id = self._validate_worker_id(worker_id)
        if not isinstance(idempotency_key, str) or not _ID_RE.fullmatch(idempotency_key):
            raise ValueError("worker_idempotency_key_invalid")
        if fixture_name not in {"safe", "idor_horizontal", "idor_vertical", "idor_nested", "false_positive", "inconclusive"}:
            raise ValueError("fixture_unsupported")
        stamp = self._stamp()
        job_id = f"remote-job:{secrets.token_hex(16)}"
        lease_id = f"remote-lease:{secrets.token_hex(16)}"
        envelope = {
            "protocol_version": 1, "job_id": job_id, "task_id": "", "worker_id": worker_id,
            "idempotency_key": idempotency_key, "lease_id": lease_id, "execution_mode": "fixture",
            "scanner": "deterministic_fixture", "fixture_name": fixture_name, "target": "",
            "authorization_reference": "", "approval_id": "", "scope_digest": "",
            "runtime_policy": "fixture_only_no_active_target", "network_policy": "disabled",
            "no_credentials": True, "no_control_plane_database": True,
            "issued_at": stamp, "expires_at": (self._now() + timedelta(seconds=self.lease_seconds)).isoformat(),
        }
        return self._insert_job(worker_id, idempotency_key, job_id, lease_id, envelope, stamp)

    def enqueue_active_job(
        self,
        *,
        worker_id: str,
        task_id: str = "",
        idempotency_key: str,
        scanner: str,
        target: str,
        scope: Mapping[str, Any] | None = None,
        authorization_reference: str,
        approval_id: str,
        approval_digest: str,
        identity: str,
        resource_id: str,
        scope_digest: str,
        runtime_proof: Any | None = None,
    ) -> dict[str, Any]:
        worker_id = self._validate_worker_id(worker_id)
        if scanner not in {"httpx", "nuclei"}:
            raise WorkerProtocolError("active_scanner_not_supported")
        with self.store._db_lock:
            row = self._worker_row(worker_id)
            if str(row["acceptance_status"]) != "VERIFIED" or runtime_proof is None:
                raise WorkerProtocolError("runtime_acceptance_required")
            try:
                from private_agent.skills.sandbox_runtime import RuntimeActivationProof
            except ImportError as exc:
                raise WorkerProtocolError("runtime_acceptance_required") from exc
            if type(runtime_proof) is not RuntimeActivationProof or not runtime_proof.ready:
                raise WorkerProtocolError("runtime_acceptance_required")
            if str(row["runtime_identity"]) == "" or not _DIGEST_RE.fullmatch(str(row["image_digest"])) or not _DIGEST_RE.fullmatch(str(row["manifest_digest"])):
                raise WorkerProtocolError("runtime_identity_invalid")
            if (
                runtime_proof.runtime_identity != str(row["runtime_identity"])
                or runtime_proof.image_digest != str(row["image_digest"])
                or runtime_proof.command_manifest_digest != str(row["manifest_digest"])
            ):
                raise WorkerProtocolError("runtime_identity_mismatch")
        try:
            parsed_scope = ScopeDefinition.from_dict(scope or {}, now=self._now())
        except (TypeError, ValueError) as exc:
            raise WorkerProtocolError("scope_invalid") from exc
        if parsed_scope.authorization_reference != authorization_reference:
            raise WorkerProtocolError("authorization_reference_mismatch")
        try:
            scope_digest_value = stable_digest(parsed_scope.to_dict())
        except (TypeError, ValueError) as exc:
            raise WorkerProtocolError("scope_digest_invalid") from exc
        if scope_digest != scope_digest_value:
            raise WorkerProtocolError("scope_digest_mismatch")
        approval = self.store.get_approval_request(approval_id)
        if not approval or approval.get("status") != "APPROVED" or str(approval.get("task_id", "")) != str(task_id):
            raise WorkerProtocolError("approval_required")
        if stable_digest(approval) != approval_digest:
            raise WorkerProtocolError("approval_digest_mismatch")
        decision, reason = parsed_scope.authorize(
            target, method="GET", mode="active", identity=identity, resource_id=resource_id, now=self._now()
        )
        if decision is not ScopeDecision.ALLOWED:
            raise WorkerProtocolError(f"target_not_authorized:{reason}")
        stamp = self._stamp()
        job_id = f"remote-job:{secrets.token_hex(16)}"
        lease_id = f"remote-lease:{secrets.token_hex(16)}"
        envelope = {
            "protocol_version": 1, "job_id": job_id, "task_id": task_id, "worker_id": worker_id,
            "idempotency_key": idempotency_key, "lease_id": lease_id, "execution_mode": "active",
            "scanner": scanner, "target": target, "fixture_name": "",
            "authorization_reference": authorization_reference, "approval_id": approval_id,
            "approval_digest": approval_digest, "scope_digest": scope_digest,
            "identity": identity, "resource_id": resource_id,
            "runtime_identity": str(row["runtime_identity"]), "image_digest": str(row["image_digest"]),
            "manifest_digest": str(row["manifest_digest"]), "network_policy": "scope_restricted",
            "no_credentials": True, "no_control_plane_database": True,
            "issued_at": stamp, "expires_at": (self._now() + timedelta(seconds=self.lease_seconds)).isoformat(),
        }
        return self._insert_job(worker_id, idempotency_key, job_id, lease_id, envelope, stamp)

    def _insert_job(self, worker_id: str, idempotency_key: str, job_id: str, lease_id: str, envelope: dict[str, Any], stamp: str) -> dict[str, Any]:
        payload = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.store._db_lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                existing = self.store.db.execute(
                    "SELECT * FROM remote_worker_jobs WHERE worker_id = ? AND idempotency_key = ?",
                    (worker_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    if json.loads(existing["envelope_json"]).get("fixture_name") != envelope.get("fixture_name"):
                        raise WorkerProtocolError("worker_idempotency_conflict")
                    self.store.db.commit()
                    return self._job_dict(existing)
                self.store.db.execute(
                    """INSERT INTO remote_worker_jobs
                    (job_id, task_id, worker_id, idempotency_key, envelope_json, lease_id, status,
                     lease_expires_at, heartbeat_at, attempt_count, result_json, result_digest,
                     last_error_code, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'QUEUED', '', '', 0, '{}', '', '', ?, ?)""",
                    (job_id, str(envelope.get("task_id", "")), worker_id, idempotency_key, payload, lease_id, stamp, stamp),
                )
                self.store.db.commit()
                return self._job_dict(self.store.db.execute("SELECT * FROM remote_worker_jobs WHERE job_id = ?", (job_id,)).fetchone())
            except sqlite3.IntegrityError as exc:
                self.store.db.rollback()
                raise WorkerProtocolError("worker_idempotency_conflict") from exc
            except Exception:
                self.store.db.rollback()
                raise

    @staticmethod
    def _job_dict(row: Any) -> dict[str, Any]:
        if row is None:
            raise WorkerProtocolError("worker_job_not_found")
        result = dict(row)
        result["envelope"] = json.loads(result.pop("envelope_json"))
        try:
            result["result"] = json.loads(result.pop("result_json"))
        except (TypeError, ValueError):
            result["result"] = {}
        return result

    def claim(self, auth: WorkerAuthentication) -> dict[str, Any] | None:
        self._require_auth(auth)
        stamp = self._stamp()
        expiry = (self._now() + timedelta(seconds=self.lease_seconds)).isoformat()
        with self.store._db_lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                row = self._worker_row(auth.worker_id)
                job = self.store.db.execute(
                    "SELECT * FROM remote_worker_jobs WHERE worker_id = ? AND status = 'QUEUED' ORDER BY created_at, job_id LIMIT 1",
                    (auth.worker_id,),
                ).fetchone()
                if job is None:
                    self.store.db.commit()
                    return None
                updated = self.store.db.execute(
                    """UPDATE remote_worker_jobs SET status = 'LEASED', lease_expires_at = ?,
                       heartbeat_at = ?, attempt_count = attempt_count + 1, updated_at = ?
                       WHERE job_id = ? AND worker_id = ? AND status = 'QUEUED'""",
                    (expiry, stamp, stamp, job["job_id"], auth.worker_id),
                )
                if updated.rowcount != 1:
                    raise WorkerProtocolError("worker_claim_fenced")
                self.store.db.commit()
                updated_row = self.store.db.execute("SELECT * FROM remote_worker_jobs WHERE job_id = ?", (job["job_id"],)).fetchone()
                return self._job_dict(updated_row)
            except Exception:
                self.store.db.rollback()
                raise

    def heartbeat(self, auth: WorkerAuthentication, *, job_id: str, lease_id: str) -> dict[str, Any]:
        self._require_auth(auth)
        stamp = self._stamp()
        expiry = (self._now() + timedelta(seconds=self.lease_seconds)).isoformat()
        with self.store._db_lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.store.db.execute("SELECT * FROM remote_worker_jobs WHERE job_id = ?", (job_id,)).fetchone()
                if row is None or row["worker_id"] != auth.worker_id:
                    raise WorkerProtocolError("worker_job_not_found")
                if row["status"] != "LEASED" or row["lease_id"] != lease_id:
                    raise WorkerProtocolError("worker_lease_fenced")
                if datetime.fromisoformat(str(row["lease_expires_at"])) <= self._now():
                    self.store.db.execute("UPDATE remote_worker_jobs SET status = 'EXECUTION_UNKNOWN', last_error_code = 'lease_expired', updated_at = ? WHERE job_id = ?", (stamp, job_id))
                    # Preserve the conservative outcome before raising; the handler below rolls back errors.
                    self.store.db.commit()
                    raise WorkerProtocolError("execution_unknown")
                self.store.db.execute(
                    "UPDATE remote_worker_jobs SET lease_expires_at = ?, heartbeat_at = ?, updated_at = ? WHERE job_id = ? AND status = 'LEASED' AND lease_id = ? AND worker_id = ?",
                    (expiry, stamp, stamp, job_id, lease_id, auth.worker_id),
                )
                self.store.db.commit()
                return {"job_id": job_id, "status": "LEASED", "lease_expires_at": expiry, "worker_id": auth.worker_id}
            except Exception:
                self.store.db.rollback()
                raise

    def submit_result(self, auth: WorkerAuthentication, *, job_id: str, lease_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
        self._require_auth(auth)
        sanitized = self._sanitize_result(result, auth.worker_id)
        result_json = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        result_for_digest = dict(sanitized)
        provenance = dict(result_for_digest.get("provenance", {}))
        provenance.pop("received_at", None)
        result_for_digest["provenance"] = provenance
        result_digest = hashlib.sha256(
            json.dumps(result_for_digest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        stamp = self._stamp()
        with self.store._db_lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.store.db.execute("SELECT * FROM remote_worker_jobs WHERE job_id = ?", (job_id,)).fetchone()
                if row is None or row["worker_id"] != auth.worker_id:
                    raise WorkerProtocolError("worker_job_not_found")
                if row["status"] in {"COMPLETED", "FAILED"}:
                    if row["result_digest"] != result_digest:
                        raise WorkerProtocolError("worker_result_idempotency_conflict")
                    self.store.db.commit()
                    return self._job_dict(row)
                if row["status"] == "EXECUTION_UNKNOWN":
                    raise WorkerProtocolError("execution_unknown")
                if row["status"] != "LEASED" or row["lease_id"] != lease_id:
                    raise WorkerProtocolError("worker_lease_fenced")
                if datetime.fromisoformat(str(row["lease_expires_at"])) <= self._now():
                    self.store.db.execute("UPDATE remote_worker_jobs SET status = 'EXECUTION_UNKNOWN', last_error_code = 'lease_expired', updated_at = ? WHERE job_id = ?", (stamp, job_id))
                    # Preserve the conservative outcome before raising; the handler below rolls back errors.
                    self.store.db.commit()
                    raise WorkerProtocolError("execution_unknown")
                status = "FAILED" if sanitized["status"] in {"FAILED", "UNAVAILABLE", "INCOMPLETE"} else "COMPLETED"
                self.store.db.execute(
                    "UPDATE remote_worker_jobs SET status = ?, result_json = ?, result_digest = ?, lease_expires_at = '', heartbeat_at = ?, updated_at = ? WHERE job_id = ? AND worker_id = ? AND lease_id = ? AND status = 'LEASED'",
                    (status, result_json, result_digest, stamp, stamp, job_id, auth.worker_id, lease_id),
                )
                self.store.db.commit()
                return self._job_dict(self.store.db.execute("SELECT * FROM remote_worker_jobs WHERE job_id = ?", (job_id,)).fetchone())
            except Exception:
                self.store.db.rollback()
                raise

    def recover_expired_jobs(self) -> list[str]:
        stamp = self._stamp()
        recovered: list[str] = []
        with self.store._db_lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                rows = self.store.db.execute(
                    "SELECT job_id FROM remote_worker_jobs WHERE status = 'LEASED' AND lease_expires_at <> '' AND lease_expires_at <= ?",
                    (stamp,),
                ).fetchall()
                for row in rows:
                    updated = self.store.db.execute(
                        "UPDATE remote_worker_jobs SET status = 'EXECUTION_UNKNOWN', last_error_code = 'worker_lost_ambiguous_execution', updated_at = ? WHERE job_id = ? AND status = 'LEASED' AND lease_expires_at <= ?",
                        (stamp, row["job_id"], stamp),
                    )
                    if updated.rowcount == 1:
                        recovered.append(str(row["job_id"]))
                self.store.db.commit()
            except Exception:
                self.store.db.rollback()
                raise
        return recovered

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.store._db_lock:
            row = self.store.db.execute("SELECT * FROM remote_worker_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._job_dict(row)

    @staticmethod
    def _sanitize_text(value: str) -> str:
        text = str(value)[:MAX_TEXT]
        lowered = text.lower()
        if any(marker in lowered for marker in ("authorization", "cookie", "api_key", "apikey", "password", "private key")):
            return "[REDACTED]"
        return text

    @classmethod
    def _sanitize_value(cls, value: Any, *, key: str = "") -> Any:
        if key.lower().replace("-", "_") in {"authorization", "cookie", "set_cookie", "token", "password", "secret", "api_key", "credentials"}:
            return "[REDACTED]"
        if isinstance(value, Mapping):
            return {str(k)[:128]: cls._sanitize_value(v, key=str(k)) for k, v in list(value.items())[:100]}
        if isinstance(value, list):
            return [cls._sanitize_value(item, key=key) for item in value[:100]]
        if isinstance(value, str):
            return cls._sanitize_text(value)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return "[REDACTED]"

    @classmethod
    def _sanitize_result(cls, result: Mapping[str, Any], worker_id: str) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise WorkerProtocolError("worker_result_invalid")
        status = str(result.get("status", ""))
        if status not in {"COMPLETED", "FAILED", "UNAVAILABLE", "INCOMPLETE"}:
            raise WorkerProtocolError("worker_result_status_invalid")
        observations = result.get("observations", [])
        findings = result.get("candidate_findings", [])
        errors = result.get("errors", [])
        if not isinstance(observations, list) or not isinstance(findings, list) or not isinstance(errors, list):
            raise WorkerProtocolError("worker_result_shape_invalid")
        sanitized = {
            "status": status,
            "observations": [cls._sanitize_value(item) for item in observations[:MAX_OBSERVATIONS]],
            "candidate_findings": [
                {**cls._sanitize_value(item), "status": "HYPOTHESIS", "verification_status": "UNVERIFIED"}
                if isinstance(cls._sanitize_value(item), dict) else {"value": cls._sanitize_value(item), "status": "HYPOTHESIS", "verification_status": "UNVERIFIED"}
                for item in findings[:MAX_FINDINGS]
            ],
            "errors": [cls._sanitize_text(str(item)) for item in errors[:MAX_ERRORS]],
            "limitations": [cls._sanitize_text(str(item)) for item in (result.get("limitations", []) if isinstance(result.get("limitations", []), list) else [])[:MAX_ERRORS]],
            "verification_status": "UNVERIFIED",
            "finding_confirmation_allowed": False,
            "provenance": {"source": "remote_worker", "worker_id": worker_id, "received_at": now()},
        }
        encoded = json.dumps(safe_json(sanitized), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
            raise WorkerProtocolError("worker_result_too_large")
        return json.loads(encoded)
