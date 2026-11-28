from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from private_agent.bugbounty.remote_worker import (
    AuthenticationReplay,
    RemoteWorkerProtocol,
    WorkerProtocolError,
)
from private_agent.storage import Store


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _signed(protocol: RemoteWorkerProtocol, worker_id: str, credential: str, method: str, path: str, body: dict, nonce: str):
    timestamp = int(protocol.clock().timestamp())
    signature = protocol.sign_request(
        credential=credential,
        method=method,
        path=path,
        timestamp=timestamp,
        nonce=nonce,
        body=body,
    )
    return protocol.authenticate(
        worker_id=worker_id,
        credential=credential,
        method=method,
        path=path,
        timestamp=timestamp,
        nonce=nonce,
        body=body,
        signature=signature,
    )


def test_fixture_job_protocol_is_authenticated_idempotent_and_fenced(tmp_path: Path) -> None:
    clock = Clock()
    store = Store(tmp_path / "remote-worker.sqlite3")
    protocol = RemoteWorkerProtocol(store, clock=clock, lease_seconds=30)
    credential = protocol.register_worker(
        "worker-fixture-1",
        runtime_identity="podman@5.0.0",
        image_digest=_digest("fixture-image"),
        manifest_digest=_digest("fixture-manifest"),
    )

    job = protocol.enqueue_fixture_job(
        worker_id="worker-fixture-1",
        idempotency_key="fixture-job-1",
        fixture_name="safe",
    )
    body = {}
    auth = _signed(protocol, "worker-fixture-1", credential, "POST", "/worker/jobs/claim", body, "nonce-claim")
    claimed = protocol.claim(auth)
    assert claimed["job_id"] == job["job_id"]
    assert claimed["lease_id"]
    assert claimed["envelope"]["execution_mode"] == "fixture"
    assert credential not in json.dumps(claimed)
    assert "password" not in json.dumps(claimed).lower()

    heartbeat_body = {"job_id": job["job_id"], "lease_id": claimed["lease_id"]}
    heartbeat_auth = _signed(
        protocol, "worker-fixture-1", credential, "POST", f"/worker/jobs/{job['job_id']}/heartbeat", heartbeat_body, "nonce-heartbeat"
    )
    assert protocol.heartbeat(heartbeat_auth, **heartbeat_body)["status"] == "LEASED"

    result_body = {
        "job_id": job["job_id"],
        "lease_id": claimed["lease_id"],
        "result": {
            "status": "COMPLETED",
            "observations": [{"name": "fixture_safe", "verification_status": "UNVERIFIED"}],
            "candidate_findings": [{"title": "not a confirmed finding", "status": "HYPOTHESIS"}],
            "notes": "Authorization: Bearer secret-must-not-persist",
        },
    }
    result_auth = _signed(
        protocol, "worker-fixture-1", credential, "POST", f"/worker/jobs/{job['job_id']}/result", result_body, "nonce-result"
    )
    completed = protocol.submit_result(result_auth, **result_body)
    assert completed["status"] == "COMPLETED"
    assert completed["result"]["verification_status"] == "UNVERIFIED"
    assert "secret-must-not-persist" not in json.dumps(completed)
    assert "Authorization" not in json.dumps(completed)

    replay_body = {}
    with pytest.raises(AuthenticationReplay):
        _signed(protocol, "worker-fixture-1", credential, "POST", "/worker/jobs/claim", replay_body, "nonce-claim")

    duplicate_body = dict(result_body)
    duplicate_auth = _signed(
        protocol, "worker-fixture-1", credential, "POST", f"/worker/jobs/{job['job_id']}/result", duplicate_body, "nonce-result-duplicate"
    )
    assert protocol.submit_result(duplicate_auth, **duplicate_body)["status"] == "COMPLETED"

    store.close()


def test_expired_remote_lease_becomes_execution_unknown_without_retry(tmp_path: Path) -> None:
    clock = Clock()
    store = Store(tmp_path / "remote-worker-expiry.sqlite3")
    protocol = RemoteWorkerProtocol(store, clock=clock, lease_seconds=5)
    credential = protocol.register_worker(
        "worker-expiry-1",
        runtime_identity="podman@5.0.0",
        image_digest=_digest("fixture-image"),
        manifest_digest=_digest("fixture-manifest"),
    )
    job = protocol.enqueue_fixture_job(
        worker_id="worker-expiry-1",
        idempotency_key="fixture-expiry-1",
        fixture_name="safe",
    )
    claimed = protocol.claim(_signed(protocol, "worker-expiry-1", credential, "POST", "/worker/jobs/claim", {}, "claim-expiry"))
    clock.advance(6)
    assert protocol.recover_expired_jobs() == [job["job_id"]]
    assert protocol.get_job(job["job_id"])["status"] == "EXECUTION_UNKNOWN"
    result_body = {
        "job_id": job["job_id"],
        "lease_id": claimed["lease_id"],
        "result": {"status": "COMPLETED", "observations": []},
    }
    auth = _signed(
        protocol, "worker-expiry-1", credential, "POST", f"/worker/jobs/{job['job_id']}/result", result_body, "late-result"
    )
    with pytest.raises(WorkerProtocolError, match="execution_unknown"):
        protocol.submit_result(auth, **result_body)
    store.close()


@pytest.mark.parametrize("expired_operation", ["heartbeat", "submit_result"])
def test_expired_remote_operation_durably_marks_execution_unknown(
    tmp_path: Path, expired_operation: str
) -> None:
    clock = Clock()
    store = Store(tmp_path / f"remote-worker-expired-{expired_operation}.sqlite3")
    protocol = RemoteWorkerProtocol(store, clock=clock, lease_seconds=5)
    credential = protocol.register_worker(
        f"worker-expired-{expired_operation}",
        runtime_identity="podman@5.0.0",
        image_digest=_digest("fixture-image"),
        manifest_digest=_digest("fixture-manifest"),
    )
    job = protocol.enqueue_fixture_job(
        worker_id=f"worker-expired-{expired_operation}",
        idempotency_key=f"fixture-expired-{expired_operation}",
        fixture_name="safe",
    )
    worker_id = f"worker-expired-{expired_operation}"
    claimed = protocol.claim(
        _signed(protocol, worker_id, credential, "POST", "/worker/jobs/claim", {}, f"claim-{expired_operation}")
    )
    assert claimed is not None
    clock.advance(6)

    heartbeat_body = {"job_id": job["job_id"], "lease_id": claimed["lease_id"]}
    result_body = {
        **heartbeat_body,
        "result": {"status": "COMPLETED", "observations": []},
    }
    if expired_operation == "heartbeat":
        auth = _signed(
            protocol,
            worker_id,
            credential,
            "POST",
            f"/worker/jobs/{job['job_id']}/heartbeat",
            heartbeat_body,
            "expired-heartbeat",
        )
        with pytest.raises(WorkerProtocolError, match="execution_unknown") as error:
            protocol.heartbeat(auth, **heartbeat_body)
        later_auth = _signed(
            protocol,
            worker_id,
            credential,
            "POST",
            f"/worker/jobs/{job['job_id']}/result",
            result_body,
            "late-result-after-heartbeat",
        )
        with pytest.raises(WorkerProtocolError, match="execution_unknown"):
            protocol.submit_result(later_auth, **result_body)
    else:
        auth = _signed(
            protocol,
            worker_id,
            credential,
            "POST",
            f"/worker/jobs/{job['job_id']}/result",
            result_body,
            "expired-result",
        )
        with pytest.raises(WorkerProtocolError, match="execution_unknown") as error:
            protocol.submit_result(auth, **result_body)
        later_auth = _signed(
            protocol,
            worker_id,
            credential,
            "POST",
            f"/worker/jobs/{job['job_id']}/heartbeat",
            heartbeat_body,
            "late-heartbeat-after-result",
        )
        with pytest.raises(WorkerProtocolError, match="worker_lease_fenced"):
            protocol.heartbeat(later_auth, **heartbeat_body)

    assert error.value.code == "execution_unknown"
    assert protocol.get_job(job["job_id"])["status"] == "EXECUTION_UNKNOWN"
    store.close()


def test_active_job_is_blocked_until_runtime_acceptance_and_approval_are_verified(tmp_path: Path) -> None:
    store = Store(tmp_path / "remote-worker-active.sqlite3")
    protocol = RemoteWorkerProtocol(store)
    credential = protocol.register_worker(
        "worker-active-1",
        runtime_identity="podman@5.0.0",
        image_digest=_digest("fixture-image"),
        manifest_digest=_digest("fixture-manifest"),
    )
    class FakeProof:
        ready = True
        runtime_identity = "podman@5.0.0"
        image_digest = _digest("fixture-image")
        command_manifest_digest = _digest("fixture-manifest")

    with pytest.raises(WorkerProtocolError, match="runtime_acceptance_required"):
        protocol.activate_worker("worker-active-1", FakeProof())
    with pytest.raises(WorkerProtocolError, match="runtime_acceptance_required"):
        protocol.enqueue_active_job(
            worker_id="worker-active-1",
            idempotency_key="active-job-1",
            scanner="httpx",
            target="https://example.test/",
            authorization_reference="written-scope",
            approval_id="approval-1",
            approval_digest=_digest("approval"),
            identity="authorized-identity",
            resource_id="authorized-resource",
            scope_digest=_digest("scope"),
        )
    store.close()


def test_credential_rotation_accepts_new_secret_and_never_persists_raw_secret(tmp_path: Path) -> None:
    store = Store(tmp_path / "remote-worker-rotation.sqlite3")
    protocol = RemoteWorkerProtocol(store)
    old_credential = protocol.register_worker(
        "worker-rotation-1",
        runtime_identity="podman@5.0.0",
        image_digest=_digest("fixture-image"),
        manifest_digest=_digest("fixture-manifest"),
    )
    new_credential = protocol.rotate_worker("worker-rotation-1")
    assert new_credential != old_credential
    _signed(protocol, "worker-rotation-1", new_credential, "POST", "/worker/jobs/claim", {}, "rotation-new")
    _signed(protocol, "worker-rotation-1", old_credential, "POST", "/worker/jobs/claim", {}, "rotation-old-grace")
    rows = store.db.execute("SELECT * FROM remote_workers").fetchall()
    assert len(rows) == 1
    row_text = json.dumps(dict(rows[0]))
    assert old_credential not in row_text
    assert new_credential not in row_text
    store.close()


def test_schema_11_database_upgrades_through_schema_21_with_all_new_tables(tmp_path: Path) -> None:
    path = tmp_path / "schema-11.sqlite3"
    store = Store(path)
    store.db.execute("DELETE FROM task_state_schema WHERE schema_version >= 12")
    for table in (
        "knowledge_attestations",
        "task_messages",
        "task_conversation_sequences",
        "remote_worker_nonces",
        "remote_worker_jobs",
        "remote_workers",
    ):
        store.db.execute(f"DROP TABLE {table}")
    store.db.commit()
    store.close()

    upgraded = Store(path)
    versions = [row["schema_version"] for row in upgraded.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")]
    assert versions[-9:] == list(range(13, 22))
    for table in (
        "remote_workers",
        "remote_worker_jobs",
        "remote_worker_nonces",
        "knowledge_attestations",
        "task_messages",
        "task_result_events",
        "task_assistant_replies",
    ):
        assert upgraded.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (table,)).fetchone()
    upgraded.close()
