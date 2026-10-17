from __future__ import annotations

import json
from pathlib import Path

import pytest

from private_agent import storage as storage_module
from private_agent.storage import Store, UnsupportedSchemaVersion

from tests.test_phase12_api import APITestTool, auth_headers, client_for, make_service


def test_schema_v3_is_upgraded_to_v4_with_queue_objects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "schema-v3.sqlite3"
    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 3)
    legacy = Store(path)
    assert [row["schema_version"] for row in legacy.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")] == [1, 2, 3]
    legacy.close()

    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 4)
    migrated = Store(path)
    assert [row["schema_version"] for row in migrated.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")] == [1, 2, 3, 4]
    assert migrated.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_queue'").fetchone()
    assert migrated.db.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_task_queue_claim'").fetchone()
    migrated.close()


def test_schema_v4_marker_without_queue_objects_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "inconsistent-v4.sqlite3"
    store = Store(path)
    store.db.execute("DROP TABLE task_queue")
    store.db.commit()
    store.close()

    with pytest.raises(UnsupportedSchemaVersion, match="inconsistent_schema_version:4"):
        Store(path)


def test_create_queued_task_persists_state_checkpoint_queue_and_idempotency_atomically(tmp_path: Path) -> None:
    path = tmp_path / "queued.sqlite3"
    store = Store(path)
    created = store.create_queued_task(
        "persist this task",
        idempotency_key="queued-key",
        request_fingerprint="fingerprint-1",
    )

    task_id = created["task_id"]
    state = store.get_task_state(task_id)
    queue = store.task_queue_record(task_id)
    checkpoint = store.latest_task_checkpoint(task_id)
    legacy = store.db.execute("SELECT status, plan_json, result_json FROM tasks WHERE id = ?", (task_id,)).fetchone()
    idempotency = store.get_api_idempotency("queued-key")

    assert created["replayed"] is False
    assert state is not None and state["status"] == "QUEUED"
    assert state["current_phase"] == "QUEUED"
    assert queue is not None and queue["queue_status"] == "QUEUED"
    assert queue["attempt_count"] == 0
    assert checkpoint is not None and checkpoint["checkpoint_type"] == "TASK_QUEUED"
    assert legacy is not None and legacy["status"] == "queued"
    assert json.loads(legacy["plan_json"]) == {}
    assert json.loads(legacy["result_json"]) == {}
    assert idempotency is not None and idempotency["task_id"] == task_id
    store.close()


def test_queued_task_and_history_survive_database_reopen(tmp_path: Path) -> None:
    path = tmp_path / "queued-restart.sqlite3"
    first = Store(path)
    created = first.create_queued_task("survive restart")
    task_id = created["task_id"]
    first.close()

    reopened = Store(path)
    state = reopened.get_task_state(task_id)
    queue = reopened.task_queue_record(task_id)
    assert state is not None and state["status"] == "QUEUED"
    assert queue is not None and queue["queue_status"] == "QUEUED"
    assert reopened.latest_task_checkpoint(task_id)["checkpoint_type"] == "TASK_QUEUED"
    reopened.close()


def test_store_idempotency_replays_and_conflicts_without_creating_second_queue_item(tmp_path: Path) -> None:
    store = Store(tmp_path / "queued-idempotency.sqlite3")
    first = store.create_queued_task(
        "same goal",
        idempotency_key="same-key",
        request_fingerprint="same-fingerprint",
    )
    replay = store.create_queued_task(
        "same goal",
        idempotency_key="same-key",
        request_fingerprint="same-fingerprint",
    )
    assert replay["replayed"] is True
    assert replay["task_id"] == first["task_id"]
    with pytest.raises(ValueError, match="idempotency_conflict"):
        store.create_queued_task(
            "different goal",
            idempotency_key="same-key",
            request_fingerprint="different-fingerprint",
        )
    assert store.db.execute("SELECT COUNT(*) AS count FROM task_queue").fetchone()["count"] == 1
    store.close()


def test_api_task_creation_is_durable_queued_and_does_not_execute_in_request(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    response = client.post(
        "/api/v1/tasks",
        json={"goal": "queued API task"},
        headers=auth_headers(key="queued-api"),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "QUEUED"
    assert body["current_phase"] == "QUEUED"
    assert body["idempotent_replay"] is False
    assert store.task_queue_record(body["task_id"])["queue_status"] == "QUEUED"
    store.close()


def test_task_listing_requires_authentication_and_returns_allowlisted_summaries(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    created = client.post(
        "/api/v1/tasks",
        json={"goal": "list-safe-goal"},
        headers=auth_headers(key="list-safe"),
    )
    task_id = created.json()["task_id"]

    unauthenticated = client.get("/api/v1/tasks")
    assert unauthenticated.status_code == 401
    assert client.get("/api/v1/tasks", headers=auth_headers()).status_code == 200
    payload = client.get("/api/v1/tasks", headers=auth_headers()).json()
    assert payload["items"]
    item = next(entry for entry in payload["items"] if entry["task_id"] == task_id)
    assert item["status"] == "QUEUED"
    assert item["queue_status"] == "QUEUED"
    assert item["goal"] == "list-safe-goal"
    for forbidden in ("state_json", "plan", "input", "inputs", "output", "outputs", "lease_id", "authorization"):
        assert forbidden not in item
    assert "task_states" not in json.dumps(payload)
    store.close()


def test_task_listing_uses_bounded_keyset_pagination(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    for index in range(3):
        response = client.post(
            "/api/v1/tasks",
            json={"goal": f"page-goal-{index}"},
            headers=auth_headers(key=f"page-{index}"),
        )
        assert response.status_code == 201

    first = client.get("/api/v1/tasks?limit=2", headers=auth_headers())
    assert first.status_code == 200
    first_body = first.json()
    assert len(first_body["items"]) == 2
    assert first_body["next_cursor"]

    second = client.get(
        f"/api/v1/tasks?limit=2&cursor={first_body['next_cursor']}",
        headers=auth_headers(),
    )
    assert second.status_code == 200
    second_body = second.json()
    assert len(second_body["items"]) == 1
    assert set(item["task_id"] for item in first_body["items"]).isdisjoint(
        item["task_id"] for item in second_body["items"]
    )
    store.close()


def test_task_listing_projects_direct_orchestrator_tasks_without_queue_leases(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    result = service.orchestrator.run("direct core task")
    task_id = str(result["task_id"])
    client = client_for(service)

    response = client.get("/api/v1/tasks", headers=auth_headers())
    assert response.status_code == 200
    item = next(entry for entry in response.json()["items"] if entry["task_id"] == task_id)
    assert item["status"] == "COMPLETED"
    assert item["queue_status"] == "COMPLETED"
    assert item["attempt_count"] == 0
    assert "lease_id" not in item
    store.close()


def test_task_listing_rejects_invalid_limits_cursors_and_status_filters(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    for query in ("limit=0", "limit=101"):
        response = client.get(f"/api/v1/tasks?{query}", headers=auth_headers())
        assert response.status_code == 422
        assert response.json()["error_code"] == "validation_error"
    malformed_cursor = client.get("/api/v1/tasks?cursor=not-a-valid-cursor", headers=auth_headers())
    assert malformed_cursor.status_code == 400
    assert malformed_cursor.json()["error_code"] == "invalid_cursor"
    invalid_status = client.get("/api/v1/tasks?status=NOT_A_STATUS", headers=auth_headers())
    assert invalid_status.status_code == 422
    assert invalid_status.json()["error_code"] == "validation_error"
    store.close()


def test_api_idempotency_replays_queued_task_and_conflicts_without_execution(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    client = client_for(service)
    first = client.post(
        "/api/v1/tasks",
        json={"goal": "idempotent queued goal"},
        headers=auth_headers(key="api-idempotent"),
    )
    replay = client.post(
        "/api/v1/tasks",
        json={"goal": "idempotent queued goal"},
        headers=auth_headers(key="api-idempotent"),
    )
    conflict = client.post(
        "/api/v1/tasks",
        json={"goal": "different queued goal"},
        headers=auth_headers(key="api-idempotent"),
    )
    assert first.status_code == 201
    assert replay.status_code == 201
    assert replay.json()["task_id"] == first.json()["task_id"]
    assert replay.json()["idempotent_replay"] is True
    assert conflict.status_code == 409
    assert conflict.json()["error_code"] == "idempotency_conflict"
    assert store.db.execute("SELECT COUNT(*) AS count FROM task_queue").fetchone()["count"] == 1
    store.close()
