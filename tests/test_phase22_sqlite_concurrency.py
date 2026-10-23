from __future__ import annotations

import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from private_agent.storage import Store
from private_agent.skills.registry import SkillRegistry


def _event(index: int) -> dict:
    return {
        "event_id": f"phase22-event-{index}-{uuid.uuid4()}",
        "event_type": "API_REQUEST_COMPLETED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": "",
        "action_id": "",
        "step_id": "",
        "correlation_id": f"phase22-correlation-{index}",
        "component": "phase22-test",
        "severity": "INFO",
        "schema_version": 1,
        "metadata": {"index": index},
    }


def test_same_database_store_connections_serialize_concurrent_observability_writes(tmp_path: Path) -> None:
    path = tmp_path / "shared-store.sqlite3"
    first = Store(path)
    second = Store(path)
    first.db.execute("PRAGMA busy_timeout = 0")
    second.db.execute("PRAGMA busy_timeout = 0")
    first.db.commit()
    second.db.commit()

    def write(index: int) -> int:
        store = first if index % 2 == 0 else second
        return store.save_observability_event(_event(index))

    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            sequences = list(pool.map(write, range(120)))
        assert len(sequences) == 120
        assert len(set(sequences)) == 120
        count = first.db.execute("SELECT COUNT(*) AS count FROM observability_events").fetchone()["count"]
        assert count == 120
    finally:
        first.close()
        second.close()


def test_observability_transaction_rolls_back_on_trigger_failure(tmp_path: Path) -> None:
    store = Store(tmp_path / "rollback.sqlite3")
    store.db.execute(
        """
        CREATE TRIGGER fail_phase22_observation
        BEFORE INSERT ON observability_events
        BEGIN
            SELECT RAISE(ABORT, 'phase22_injected_failure');
        END
        """
    )
    store.db.commit()
    try:
        with pytest.raises(sqlite3.IntegrityError, match="phase22_injected_failure"):
            store.save_observability_event(_event(1))
        assert store.db.execute("SELECT COUNT(*) AS count FROM observability_events").fetchone()["count"] == 0
    finally:
        store.close()


def test_concurrent_queue_claims_have_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "claim-race.sqlite3"
    first = Store(path)
    second = Store(path)
    first.db.execute("PRAGMA busy_timeout = 0")
    second.db.execute("PRAGMA busy_timeout = 0")
    first.db.commit()
    second.db.commit()
    task_id = str(first.create_queued_task("phase22 claim race")["task_id"])

    def claim(index: int):
        store = first if index == 0 else second
        return store.claim_next_task(f"phase22-worker-{index}", lease_ttl_seconds=60)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(claim, range(2)))
        winners = [item for item in results if item is not None]
        assert len(winners) == 1
        assert winners[0]["task_id"] == task_id
        assert first.task_queue_record(task_id)["queue_status"] == "LEASED"
    finally:
        first.close()
        second.close()


def test_skill_registry_direct_writes_use_shared_store_lock(tmp_path: Path) -> None:
    path = tmp_path / "skill-registry-race.sqlite3"
    stores = [Store(path) for _ in range(4)]
    for store in stores:
        store.db.execute("PRAGMA busy_timeout = 0")
        store.db.commit()

    def create(index: int) -> None:
        SkillRegistry(stores[index % len(stores)]).create(
            f"phase22-skill-{index}",
            "controlled skill",
            "non-executable metadata",
        )

    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(create, range(80)))
        count = stores[0].db.execute("SELECT COUNT(*) AS count FROM skills WHERE name LIKE 'phase22-skill-%'").fetchone()["count"]
        assert count == 80
    finally:
        for store in stores:
            store.close()


def test_finish_failure_rolls_back_state_and_queue_together(tmp_path: Path) -> None:
    store = Store(tmp_path / "finish-rollback.sqlite3")
    created = store.create_queued_task("phase22 finish rollback")
    task_id = str(created["task_id"])
    claim = store.claim_next_task("phase22-worker", lease_ttl_seconds=60)
    assert claim is not None
    store.db.execute(
        """
        CREATE TRIGGER fail_phase22_finish
        BEFORE UPDATE OF queue_status ON task_queue
        BEGIN
            SELECT RAISE(ABORT, 'phase22_finish_failure');
        END
        """
    )
    store.db.commit()
    try:
        with pytest.raises(sqlite3.IntegrityError, match="phase22_finish_failure"):
            store.finish_task_claim(claim, "FAILED", error_code="injected", state_status="FAILED")
        assert store.task_queue_record(task_id)["queue_status"] == "LEASED"
        assert store.task_queue_record(task_id)["lease_id"] == claim["lease_id"]
        assert store.get_task_state(task_id)["status"] == "QUEUED"
    finally:
        store.close()
