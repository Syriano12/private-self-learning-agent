from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from private_agent.api import create_app
from private_agent.storage import Store, StoreClosedError, StoreClosingError
from private_agent.worker import BackgroundWorker
from tests.test_phase12_api import APITestTool, make_service


def test_app_shutdown_drains_terminal_task_before_store_close(tmp_path: Path, monkeypatch) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    created = store.create_queued_task("shutdown while terminal claim is being finalized")
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="phase25-shutdown-drain",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.01,
    )
    app = create_app(service=service, worker=worker, worker_shutdown_timeout=0.02)
    finish_entered = threading.Event()
    release_finish = threading.Event()
    begin_shutdown = threading.Event()
    shutdown_started = threading.Event()
    shutdown_timeout_observed = threading.Event()
    app_finished = threading.Event()
    errors: list[BaseException] = []
    original_finish = store.finish_task_claim
    original_stop = worker.stop

    def delayed_finish(*args, **kwargs):
        finish_entered.set()
        if not release_finish.wait(timeout=5):
            raise TimeoutError("test_finish_release_timeout")
        return original_finish(*args, **kwargs)

    def observed_stop(*, timeout=5.0):
        stopped = original_stop(timeout=timeout)
        if not stopped:
            shutdown_timeout_observed.set()
        return stopped

    monkeypatch.setattr(store, "finish_task_claim", delayed_finish)
    monkeypatch.setattr(worker, "stop", observed_stop)

    def serve_until_shutdown() -> None:
        try:
            with TestClient(app, raise_server_exceptions=False):
                if not finish_entered.wait(timeout=5):
                    raise AssertionError("worker_did_not_reach_finish_task_claim")
                shutdown_started.set()
                if not begin_shutdown.wait(timeout=5):
                    raise AssertionError("test_shutdown_start_timeout")
        except BaseException as exc:
            errors.append(exc)
        finally:
            app_finished.set()

    app_thread = threading.Thread(target=serve_until_shutdown, name="phase25-test-app-lifecycle")
    app_thread.start()
    try:
        assert shutdown_started.wait(timeout=5)
        assert store.get_task_state(created["task_id"])["status"] == "COMPLETED"
        assert store.task_queue_record(created["task_id"])["queue_status"] == "LEASED"

        # Permit TestClient to begin app shutdown while finish_task_claim remains blocked.
        begin_shutdown.set()
        assert shutdown_timeout_observed.wait(timeout=3)
        assert not app_finished.is_set(), "lifespan returned while the worker still owned a task claim"
        assert not store.closed, "Store must remain open while the worker drains its current claim"
        assert app_thread.is_alive()

        release_finish.set()
        app_thread.join(timeout=5)
        assert not app_thread.is_alive()
        assert not errors, [repr(error) for error in errors]
        assert app.state.worker_shutdown_clean is True
        assert store.get_task_state(created["task_id"])["status"] == "COMPLETED"
        assert store.task_queue_record(created["task_id"])["queue_status"] == "COMPLETED"
    finally:
        release_finish.set()
        app_thread.join(timeout=5)
        store.close()


def test_store_close_waits_for_worker_and_rejects_new_operations(tmp_path: Path, monkeypatch) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    observer = Store(store.path)
    created = store.create_queued_task("direct Store close drains active claim")
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="phase25-store-close-drain",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.01,
    )
    finish_entered = threading.Event()
    release_finish = threading.Event()
    close_finished = threading.Event()
    join_entered = threading.Event()
    original_finish = store.finish_task_claim
    original_join = worker.join

    def delayed_finish(*args, **kwargs):
        finish_entered.set()
        if not release_finish.wait(timeout=5):
            raise TimeoutError("test_finish_release_timeout")
        return original_finish(*args, **kwargs)

    def observed_join(*, timeout=5.0):
        join_entered.set()
        return original_join(timeout=timeout)

    store.finish_task_claim = delayed_finish  # type: ignore[method-assign]
    monkeypatch.setattr(worker, "join", observed_join)
    worker_thread = worker.start()
    close_errors: list[BaseException] = []

    def close_store() -> None:
        try:
            store.close()
        except BaseException as exc:
            close_errors.append(exc)
        finally:
            close_finished.set()

    closer = threading.Thread(target=close_store, name="phase25-store-close")
    try:
        assert finish_entered.wait(timeout=5)
        assert observer.get_task_state(created["task_id"])["status"] == "COMPLETED"
        assert observer.task_queue_record(created["task_id"])["queue_status"] == "LEASED"
        closer.start()
        assert join_entered.wait(timeout=3)
        assert store.closing
        assert not store.closed
        with pytest.raises(StoreClosingError):
            store.get_task_state(created["task_id"])
        assert join_entered.wait(timeout=3)
        assert not close_finished.is_set()

        release_finish.set()
        closer.join(timeout=5)
        worker_thread.join(timeout=5)
        assert not closer.is_alive()
        assert not worker_thread.is_alive()
        assert not close_errors, [repr(error) for error in close_errors]
        assert store.closed
        assert observer.get_task_state(created["task_id"])["status"] == "COMPLETED"
        assert observer.task_queue_record(created["task_id"])["queue_status"] == "COMPLETED"
    finally:
        release_finish.set()
        if closer.ident is not None:
            closer.join(timeout=5)
        worker.stop(timeout=2)
        worker_thread.join(timeout=2)
        observer.close()
        store.close()


def test_store_close_is_idempotent_and_rejects_operations_after_close(tmp_path: Path) -> None:
    store = Store(tmp_path / "idempotent-close.sqlite3")
    store.close()
    store.close()
    assert store.closed
    with pytest.raises(StoreClosedError):
        store.get_task_state("missing-task")


def test_worker_stop_timeout_does_not_close_store_and_close_completes_after_release(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    observer = Store(store.path)
    created = store.create_queued_task("bounded stop timeout preserves active claim")
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="phase25-stop-timeout",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.01,
    )
    finish_entered = threading.Event()
    release_finish = threading.Event()
    original_finish = store.finish_task_claim

    def delayed_finish(*args, **kwargs):
        finish_entered.set()
        if not release_finish.wait(timeout=5):
            raise TimeoutError("test_finish_release_timeout")
        return original_finish(*args, **kwargs)

    store.finish_task_claim = delayed_finish  # type: ignore[method-assign]
    worker_thread = worker.start()
    try:
        assert finish_entered.wait(timeout=5)
        assert worker.stop(timeout=0.01) is False
        assert worker_thread.is_alive()
        assert not store.closed
        assert observer.task_queue_record(created["task_id"])["queue_status"] == "LEASED"

        release_finish.set()
        assert worker.stop(timeout=None) is True
        worker_thread.join(timeout=5)
        assert not worker_thread.is_alive()
        assert observer.get_task_state(created["task_id"])["status"] == "COMPLETED"
        assert observer.task_queue_record(created["task_id"])["queue_status"] == "COMPLETED"
    finally:
        release_finish.set()
        worker.stop(timeout=None)
        worker_thread.join(timeout=5)
        store.close()
        observer.close()


def test_app_shutdown_records_stop_exception_but_still_joins_worker(tmp_path: Path, monkeypatch) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="phase25-stop-exception",
        poll_interval_seconds=0.01,
        idle_poll_interval_seconds=0.01,
    )
    app = create_app(service=service, worker=worker, worker_shutdown_timeout=0.02)

    def fail_stop(*, timeout=5.0):
        raise RuntimeError("controlled_stop_failure")

    def fail_request_stop():
        raise RuntimeError("controlled_request_stop_failure")

    monkeypatch.setattr(worker, "stop", fail_stop)
    monkeypatch.setattr(worker, "request_stop", fail_request_stop)
    try:
        with TestClient(app, raise_server_exceptions=False):
            assert worker.running
        thread = app.state.agent_worker_thread
        assert not thread.is_alive()
        assert app.state.worker_shutdown_clean is False
        assert app.state.worker_shutdown_error == "RuntimeError"
        assert worker.last_report is not None
        assert worker.last_report["status"] == "STOPPED"
    finally:
        worker.join(timeout=None)
        store.close()


@pytest.mark.parametrize("terminal_sql", ["COMMIT", "ROLLBACK"])
def test_guarded_cursor_releases_transaction_lock_on_sql_terminal_statement(
    tmp_path: Path,
    terminal_sql: str,
) -> None:
    store = Store(tmp_path / f"cursor-{terminal_sql.lower()}.sqlite3")
    store.db.execute("CREATE TABLE tx_probe (value INTEGER)")
    cursor = store.db.cursor()
    cursor.execute("BEGIN IMMEDIATE")
    cursor.executemany("INSERT INTO tx_probe(value) VALUES (?)", [(1,), (2,)])
    cursor.execute(terminal_sql)
    assert store.db._store_transaction_owner is None

    query_finished = threading.Event()
    query_result: list[object] = []

    def query_from_another_thread() -> None:
        try:
            query_result.append(store.db.execute("SELECT COUNT(*) FROM tx_probe").fetchone()[0])
        except BaseException as exc:
            query_result.append(exc)
        finally:
            query_finished.set()

    query_thread = threading.Thread(target=query_from_another_thread, name="phase25-transaction-probe")
    query_thread.start()
    try:
        assert query_finished.wait(timeout=3), "transaction lock remained owned after SQL COMMIT/ROLLBACK"
    finally:
        if not query_finished.is_set():
            store.db._store_release_transaction_lock()
        query_thread.join(timeout=3)
        store.close()
    assert not query_thread.is_alive()
    assert len(query_result) == 1
    assert not isinstance(query_result[0], BaseException)
    assert query_result[0] == (2 if terminal_sql == "COMMIT" else 0)


def test_store_close_waits_for_unregistered_direct_run_once(tmp_path: Path, monkeypatch) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    observer = Store(store.path)
    created = store.create_queued_task("close while direct run_once is finalizing")
    worker = BackgroundWorker(service.orchestrator, worker_id="phase25-direct-run-once")
    finish_entered = threading.Event()
    release_finish = threading.Event()
    join_entered = threading.Event()
    close_finished = threading.Event()
    errors: list[BaseException] = []
    result: list[dict] = []
    original_finish = store.finish_task_claim
    original_join = worker.join

    def delayed_finish(*args, **kwargs):
        finish_entered.set()
        if not release_finish.wait(timeout=5):
            raise TimeoutError("test_finish_release_timeout")
        return original_finish(*args, **kwargs)

    def observed_join(*, timeout=5.0):
        join_entered.set()
        return original_join(timeout=timeout)

    monkeypatch.setattr(store, "finish_task_claim", delayed_finish)
    monkeypatch.setattr(worker, "join", observed_join)

    def run_directly() -> None:
        try:
            result.append(worker.run_once())
        except BaseException as exc:
            errors.append(exc)

    def close_store() -> None:
        try:
            store.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            close_finished.set()

    run_thread = threading.Thread(target=run_directly, name="phase25-direct-run-once")
    closer = threading.Thread(target=close_store, name="phase25-direct-run-once-store-close")
    run_thread.start()
    try:
        assert finish_entered.wait(timeout=5)
        assert observer.get_task_state(created["task_id"])["status"] == "COMPLETED"
        assert observer.task_queue_record(created["task_id"])["queue_status"] == "LEASED"
        closer.start()
        assert join_entered.wait(timeout=3)
        assert store.closing
        assert not close_finished.is_set()
        with pytest.raises(StoreClosingError):
            store.get_task_state(created["task_id"])

        release_finish.set()
        closer.join(timeout=5)
        run_thread.join(timeout=5)
        assert not closer.is_alive()
        assert not run_thread.is_alive()
        assert store.closed
        assert not errors, [repr(error) for error in errors]
        assert result and result[0]["status"] == "COMPLETED"
        assert observer.get_task_state(created["task_id"])["status"] == "COMPLETED"
        assert observer.task_queue_record(created["task_id"])["queue_status"] == "COMPLETED"
    finally:
        release_finish.set()
        if closer.ident is not None:
            closer.join(timeout=5)
        run_thread.join(timeout=5)
        observer.close()
        store.close()


def test_negative_worker_stop_timeout_does_not_request_shutdown(tmp_path: Path, monkeypatch) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    worker = BackgroundWorker(service.orchestrator, worker_id="phase25-negative-timeout")
    calls: list[bool] = []
    monkeypatch.setattr(worker, "request_stop", lambda: calls.append(True))
    try:
        with pytest.raises(ValueError, match="shutdown_timeout_invalid"):
            worker.stop(timeout=-1)
        assert calls == []
    finally:
        store.close()



def test_store_close_waits_for_worker_startup_between_registration_and_thread_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    worker = BackgroundWorker(service.orchestrator, worker_id="phase25-start-close-race")
    start_entered = threading.Event()
    release_start = threading.Event()
    join_entered = threading.Event()
    close_finished = threading.Event()
    worker_thread_holder: list[threading.Thread] = []
    errors: list[BaseException] = []
    original_thread_start = threading.Thread.start
    original_worker_join = worker.join

    def delayed_thread_start(thread: threading.Thread) -> None:
        if thread.name == "agent-worker-phase25-start-close-race":
            start_entered.set()
            if not release_start.wait(timeout=5):
                raise TimeoutError("test_worker_start_release_timeout")
        original_thread_start(thread)

    def observed_join(*, timeout=5.0):
        join_entered.set()
        return original_worker_join(timeout=timeout)

    monkeypatch.setattr(threading.Thread, "start", delayed_thread_start)
    monkeypatch.setattr(worker, "join", observed_join)

    def start_worker() -> None:
        try:
            worker_thread_holder.append(worker.start())
        except BaseException as exc:
            errors.append(exc)

    def close_store() -> None:
        try:
            store.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            close_finished.set()

    starter = threading.Thread(target=start_worker, name="phase25-worker-starter")
    closer = threading.Thread(target=close_store, name="phase25-start-race-store-close")
    starter.start()
    try:
        assert start_entered.wait(timeout=3)
        closer.start()
        assert join_entered.wait(timeout=3)
        assert store.closing
        assert not store.closed
        assert not close_finished.is_set()

        release_start.set()
        starter.join(timeout=5)
        closer.join(timeout=5)
        assert not starter.is_alive()
        assert not closer.is_alive()
        assert not errors, [repr(error) for error in errors]
        assert store.closed
        assert len(worker_thread_holder) == 1
        assert not worker_thread_holder[0].is_alive()
    finally:
        release_start.set()
        starter.join(timeout=5)
        if closer.ident is not None:
            closer.join(timeout=5)
        worker.request_stop()
        worker.join(timeout=None)
        store.close()



def test_store_close_continues_drain_when_stop_signal_fallback_raises(tmp_path: Path) -> None:
    store = Store(tmp_path / "store-close-fallback.sqlite3")
    join_entered = threading.Event()
    close_finished = threading.Event()
    close_errors: list[BaseException] = []

    class BrokenStopEvent:
        def set(self) -> None:
            raise RuntimeError("controlled_stop_event_failure")

    class BrokenStopWorker:
        _stop_event = BrokenStopEvent()

        def request_stop(self) -> None:
            raise RuntimeError("controlled_request_stop_failure")

        def join(self, *, timeout=None) -> bool:
            join_entered.set()
            store.unregister_background_worker(self)
            return True

    store.register_background_worker(BrokenStopWorker())

    def close_store() -> None:
        try:
            store.close()
        except BaseException as exc:
            close_errors.append(exc)
        finally:
            close_finished.set()

    closer = threading.Thread(target=close_store, name="phase25-store-close-stop-fallback")
    closer.start()
    assert join_entered.wait(timeout=3)
    closer.join(timeout=3)
    assert not closer.is_alive()
    assert close_finished.is_set()
    assert not close_errors, [repr(error) for error in close_errors]
    assert store.closed



def test_real_worker_exits_on_store_closing_when_both_stop_signals_fail(tmp_path: Path, monkeypatch) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    idle_observed = threading.Event()

    class BrokenStopEvent:
        def __init__(self) -> None:
            self.event = threading.Event()

        def clear(self) -> None:
            self.event.clear()

        def set(self) -> None:
            raise RuntimeError("controlled_stop_event_failure")

        def is_set(self) -> bool:
            return self.event.is_set()

        def wait(self, timeout: float | None = None) -> bool:
            return self.event.wait(timeout)

    def observe(event: dict) -> None:
        if event.get("event_type") == "worker_idle":
            idle_observed.set()

    def fail_request_stop() -> None:
        raise RuntimeError("controlled_request_stop_failure")

    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="phase25-real-worker-stop-failure",
        idle_poll_interval_seconds=0.01,
        event_logger=observe,
    )
    worker._stop_event = BrokenStopEvent()  # type: ignore[assignment]
    monkeypatch.setattr(worker, "request_stop", fail_request_stop)
    worker_thread = worker.start()
    close_errors: list[BaseException] = []

    def close_store() -> None:
        try:
            store.close()
        except BaseException as exc:
            close_errors.append(exc)

    closer = threading.Thread(target=close_store, name="phase25-real-worker-failed-stop")
    try:
        assert idle_observed.wait(timeout=3)
        closer.start()
        closer.join(timeout=3)
        assert not closer.is_alive(), "Store.close did not drain a worker that observes Store.closing"
        worker_thread.join(timeout=3)
        assert not worker_thread.is_alive()
        assert store.closed
        assert not close_errors, [repr(error) for error in close_errors]
        assert worker.last_report is not None
        assert worker.last_report["status"] == "STOPPED"
    finally:
        if closer.ident is not None:
            closer.join(timeout=3)
        worker_thread.join(timeout=3)
        store.close()


def test_executemany_partial_failure_rolls_back_and_releases_transaction_guard(tmp_path: Path) -> None:
    store = Store(tmp_path / "executemany-partial-failure.sqlite3")
    store.db.execute("CREATE TABLE tx_probe (value INTEGER UNIQUE)")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store.db.executemany("INSERT INTO tx_probe(value) VALUES (?)", [(1,), (1,)])

        assert not store.db.in_transaction
        assert store.db._store_transaction_owner is None
        assert store.db.execute("SELECT COUNT(*) FROM tx_probe").fetchone()[0] == 0

        finished = threading.Event()
        errors: list[BaseException] = []

        def write_from_other_thread() -> None:
            try:
                store.db.execute("INSERT INTO tx_probe(value) VALUES (?)", (2,))
                store.db.commit()
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=write_from_other_thread, name="phase25-executemany-recovery")
        thread.start()
        assert finished.wait(timeout=3), "SQLite guard remained locked after failed executemany"
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not errors, [repr(error) for error in errors]
        assert store.db.execute("SELECT value FROM tx_probe ORDER BY value").fetchall()[0][0] == 2
        assert not store.db.in_transaction
        assert store.db._store_transaction_owner is None
    finally:
        store.close()


def test_failed_executescript_rolls_back_its_partial_transaction(tmp_path: Path) -> None:
    store = Store(tmp_path / "executescript-partial-failure.sqlite3")
    store.db.execute("CREATE TABLE tx_probe (value INTEGER)")
    try:
        with pytest.raises(sqlite3.OperationalError):
            store.db.executescript(
                "BEGIN IMMEDIATE; INSERT INTO tx_probe(value) VALUES (1); THIS IS NOT VALID SQL;"
            )

        assert not store.db.in_transaction
        assert store.db._store_transaction_owner is None
        assert store.db.execute("SELECT COUNT(*) FROM tx_probe").fetchone()[0] == 0
    finally:
        store.close()


@pytest.mark.parametrize("use_cursor", [False, True])
def test_executescript_inside_transaction_is_rejected_without_implicit_commit(
    tmp_path: Path,
    use_cursor: bool,
) -> None:
    store = Store(tmp_path / f"executescript-existing-transaction-{use_cursor}.sqlite3")
    store.db.execute("CREATE TABLE tx_probe (value INTEGER)")
    try:
        store.db.execute("BEGIN IMMEDIATE")
        store.db.execute("INSERT INTO tx_probe(value) VALUES (10)")
        current_thread_id = threading.get_ident()
        assert store.db.in_transaction
        assert store.db._store_transaction_owner == current_thread_id

        executescript = store.db.cursor().executescript if use_cursor else store.db.executescript
        with pytest.raises(sqlite3.ProgrammingError, match="executescript_not_allowed_in_transaction"):
            executescript("INSERT INTO tx_probe(value) VALUES (20);")

        # Refusal must preserve the caller-owned transaction and its lock.
        assert store.db.in_transaction
        assert store.db._store_transaction_owner == current_thread_id
        assert [row[0] for row in store.db.execute("SELECT value FROM tx_probe ORDER BY value").fetchall()] == [10]
        store.db.rollback()
        assert not store.db.in_transaction
        assert store.db._store_transaction_owner is None
        assert store.db.execute("SELECT COUNT(*) FROM tx_probe").fetchone()[0] == 0
    finally:
        if store.db.in_transaction and store.db._store_transaction_owner == threading.get_ident():
            store.db.rollback()
        store.close()


def test_transaction_owned_by_one_thread_cannot_be_committed_by_another(tmp_path: Path) -> None:
    store = Store(tmp_path / "transaction-owner-thread.sqlite3")
    store.db.execute("CREATE TABLE tx_probe (value INTEGER)")
    store.db.execute("BEGIN IMMEDIATE")
    store.db.execute("INSERT INTO tx_probe(value) VALUES (1)")
    owner_thread_id = threading.get_ident()
    attempted = threading.Event()
    finished = threading.Event()
    errors: list[BaseException] = []

    def commit_from_another_thread() -> None:
        attempted.set()
        try:
            store.db.commit()
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=commit_from_another_thread, name="phase25-foreign-transaction-commit")
    thread.start()
    try:
        assert attempted.wait(timeout=3)
        assert not finished.wait(timeout=0.05), "another thread committed a transaction it did not own"
        assert store.db.in_transaction
        assert store.db._store_transaction_owner == owner_thread_id
        store.db.rollback()
        assert finished.wait(timeout=3)
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not errors, [repr(error) for error in errors]
        assert not store.db.in_transaction
        assert store.db._store_transaction_owner is None
        assert store.db.execute("SELECT COUNT(*) FROM tx_probe").fetchone()[0] == 0
    finally:
        if store.db.in_transaction and store.db._store_transaction_owner == owner_thread_id:
            store.db.rollback()
        thread.join(timeout=3)
        store.close()


def test_store_close_rejects_call_from_thread_owning_transaction(tmp_path: Path) -> None:
    store = Store(tmp_path / "close-owned-transaction.sqlite3")
    store.db.execute("CREATE TABLE tx_probe (value INTEGER)")
    store.db.execute("BEGIN IMMEDIATE")
    store.db.execute("INSERT INTO tx_probe(value) VALUES (1)")
    owner_thread_id = threading.get_ident()

    with pytest.raises(RuntimeError, match="store_transaction_active"):
        store.close()

    assert not store.closed
    assert not store.closing
    assert store.db.in_transaction
    assert store.db._store_transaction_owner == owner_thread_id
    store.db.rollback()
    assert not store.db.in_transaction
    assert store.db._store_transaction_owner is None
    store.close()
    assert store.closed


@pytest.mark.parametrize("request_stop_error_type", [RuntimeError, KeyboardInterrupt])
def test_api_shutdown_drains_worker_when_both_stop_signals_fail(
    tmp_path: Path,
    monkeypatch,
    request_stop_error_type: type[BaseException],
) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"))
    created = store.create_queued_task("terminal state before delayed queue finalization")
    finish_entered = threading.Event()
    release_finish = threading.Event()
    client_ready = threading.Event()
    leave_client = threading.Event()
    lifecycle_join_entered = threading.Event()
    app_finished = threading.Event()
    errors: list[BaseException] = []

    class BrokenStopEvent:
        def __init__(self) -> None:
            self.event = threading.Event()

        def clear(self) -> None:
            self.event.clear()

        def set(self) -> None:
            raise RuntimeError("controlled_stop_event_failure")

        def is_set(self) -> bool:
            return self.event.is_set()

        def wait(self, timeout: float | None = None) -> bool:
            return self.event.wait(timeout)

    worker = BackgroundWorker(
        service.orchestrator,
        worker_id="phase25-api-broken-stop-signals",
        idle_poll_interval_seconds=0.01,
    )
    broken_event = BrokenStopEvent()
    worker._stop_event = broken_event  # type: ignore[assignment]
    original_join = worker.join
    original_finish = store.finish_task_claim
    app = create_app(service=service, worker=worker, worker_shutdown_timeout=0.02)

    def delayed_finish(*args, **kwargs):
        finish_entered.set()
        if not release_finish.wait(timeout=5):
            raise TimeoutError("test_finish_release_timeout")
        return original_finish(*args, **kwargs)

    store.finish_task_claim = delayed_finish  # type: ignore[method-assign]

    def fail_request_stop() -> None:
        raise request_stop_error_type("controlled_request_stop_failure")

    def fail_stop(*, timeout=5.0):
        raise RuntimeError("controlled_stop_failure")

    def observe_lifecycle_join(*, timeout=5.0) -> bool:
        if timeout is None:
            lifecycle_join_entered.set()
        return original_join(timeout=timeout)

    monkeypatch.setattr(worker, "request_stop", fail_request_stop)
    monkeypatch.setattr(worker, "stop", fail_stop)
    monkeypatch.setattr(worker, "join", observe_lifecycle_join)

    def serve_until_shutdown() -> None:
        try:
            with TestClient(app, raise_server_exceptions=False):
                if not finish_entered.wait(timeout=5):
                    raise AssertionError("worker_did_not_reach_delayed_finish")
                client_ready.set()
                if not leave_client.wait(timeout=5):
                    raise AssertionError("test_shutdown_release_timeout")
        except BaseException as exc:
            errors.append(exc)
        finally:
            app_finished.set()

    app_thread = threading.Thread(target=serve_until_shutdown, name="phase25-api-broken-stop-lifecycle")
    app_thread.start()
    try:
        assert client_ready.wait(timeout=5)
        assert store.get_task_state(created["task_id"])["status"] == "COMPLETED"
        assert store.task_queue_record(created["task_id"])["queue_status"] == "LEASED"
        leave_client.set()
        assert lifecycle_join_entered.wait(timeout=3)
        assert store.closing
        assert not store.closed
        assert not app_finished.is_set(), "API shutdown returned before worker drain"
        assert store.task_queue_record(created["task_id"])["queue_status"] == "LEASED"

        release_finish.set()
        assert app_finished.wait(timeout=3), "API shutdown remained blocked after Store shutdown request"
        app_thread.join(timeout=3)
        worker_thread = app.state.agent_worker_thread
        assert not app_thread.is_alive()
        assert not worker_thread.is_alive()
        if issubclass(request_stop_error_type, Exception):
            assert not errors, [repr(error) for error in errors]
        else:
            assert len(errors) == 1
            assert type(errors[0]).__name__ in {"KeyboardInterrupt", "BaseExceptionGroup", "CancelledError"}
        assert app.state.worker_shutdown_clean is False
        assert app.state.worker_shutdown_error == request_stop_error_type.__name__
        assert worker.last_report is not None
        assert worker.last_report["status"] == "STOPPED"
        assert store.get_task_state(created["task_id"])["status"] == "COMPLETED"
        assert store.task_queue_record(created["task_id"])["queue_status"] == "COMPLETED"

        store.close()
        assert store.closed
    finally:
        leave_client.set()
        release_finish.set()
        # Test-only escape hatch keeps the pre-fix implementation from leaking
        # a non-daemon TestClient thread after the expected RED assertion.
        broken_event.event.set()
        app_thread.join(timeout=5)
        worker_thread = getattr(app.state, "agent_worker_thread", None)
        if worker_thread is not None:
            worker_thread.join(timeout=5)
        if not store.closed:
            store.close()


def test_executescript_without_explicit_begin_is_atomic_on_failure(tmp_path: Path) -> None:
    store = Store(tmp_path / "executescript-autowrap.sqlite3")
    store.db.execute("CREATE TABLE tx_probe (value INTEGER)")
    try:
        with pytest.raises(sqlite3.OperationalError):
            store.db.executescript(
                "INSERT INTO tx_probe(value) VALUES (1); THIS IS NOT VALID SQL;"
            )

        assert not store.db.in_transaction
        assert store.db._store_transaction_owner is None
        assert store.db.execute("SELECT COUNT(*) FROM tx_probe").fetchone()[0] == 0
    finally:
        store.close()


def test_custom_cursor_factory_cannot_bypass_store_guards(tmp_path: Path) -> None:
    store = Store(tmp_path / "custom-cursor-factory.sqlite3")
    try:
        with pytest.raises(sqlite3.ProgrammingError, match="custom_cursor_factory_not_supported"):
            store.db.cursor(sqlite3.Cursor)
        assert not store.db.in_transaction
        assert store.db._store_transaction_owner is None
    finally:
        if store.db.in_transaction and store.db._store_transaction_owner == threading.get_ident():
            store.db.rollback()
        store.close()


@pytest.mark.parametrize("savepoint_is_root", [False, True])
def test_executescript_rollback_to_savepoint_is_not_a_transaction_boundary(
    tmp_path: Path,
    savepoint_is_root: bool,
) -> None:
    store = Store(tmp_path / f"rollback-to-savepoint-{savepoint_is_root}.sqlite3")
    store.db.execute("CREATE TABLE tx_probe (value INTEGER)")
    if savepoint_is_root:
        script = (
            "SAVEPOINT s; INSERT INTO tx_probe(value) VALUES (1); "
            "INSERT INTO tx_probe(value) VALUES (2); ROLLBACK TO s; RELEASE s;"
        )
        expected = []
    else:
        script = (
            "BEGIN IMMEDIATE; INSERT INTO tx_probe(value) VALUES (1); SAVEPOINT s; "
            "INSERT INTO tx_probe(value) VALUES (2); ROLLBACK TO s; RELEASE s; COMMIT;"
        )
        expected = [1]
    try:
        store.db.executescript(script)
        assert not store.db.in_transaction
        assert store.db._store_transaction_owner is None
        assert [row[0] for row in store.db.execute("SELECT value FROM tx_probe ORDER BY value").fetchall()] == expected
    finally:
        if store.db.in_transaction and store.db._store_transaction_owner == threading.get_ident():
            store.db.rollback()
        store.close()


def test_executescript_preserves_autocommit_only_sqlite_commands(tmp_path: Path) -> None:
    store = Store(tmp_path / "executescript-autocommit.sqlite3")
    try:
        store.db.executescript("PRAGMA journal_mode=WAL;")
        assert store.db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

        store.db.execute("CREATE TABLE tx_probe (value INTEGER)")
        store.db.execute("INSERT INTO tx_probe(value) VALUES (1)")
        store.db.commit()
        store.db.executescript("PRAGMA wal_checkpoint(FULL);")
        store.db.executescript("VACUUM;")

        assert not store.db.in_transaction
        assert store.db._store_transaction_owner is None
        assert store.db.execute("SELECT value FROM tx_probe").fetchall()[0][0] == 1
    finally:
        store.close()


def test_executescript_rejects_mixed_autocommit_and_transactional_commands(tmp_path: Path) -> None:
    store = Store(tmp_path / "executescript-mixed-autocommit.sqlite3")
    try:
        with pytest.raises(
            sqlite3.ProgrammingError,
            match="executescript_mixed_autocommit_and_transactional_unsupported",
        ):
            store.db.executescript(
                "PRAGMA journal_mode=WAL; CREATE TABLE should_not_exist (value INTEGER);"
            )

        assert store.db.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal"
        assert store.db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'should_not_exist'"
        ).fetchone()[0] == 0
        assert not store.db.in_transaction
        assert store.db._store_transaction_owner is None
    finally:
        store.close()


def test_executescript_with_utf8_bom_before_begin_is_recognized(tmp_path: Path) -> None:
    store = Store(tmp_path / "executescript-bom.sqlite3")
    store.db.execute("CREATE TABLE tx_probe (value INTEGER)")
    try:
        store.db.executescript(
            "\ufeffBEGIN IMMEDIATE; INSERT INTO tx_probe(value) VALUES (1); COMMIT;"
        )

        assert not store.db.in_transaction
        assert store.db._store_transaction_owner is None
        assert store.db.execute("SELECT value FROM tx_probe").fetchall()[0][0] == 1
    finally:
        store.close()


@pytest.mark.parametrize(
    "script",
    [
        "BEGIN IMMEDIATE; CREATE TABLE should_not_exist (value INTEGER); VACUUM; COMMIT;",
        "SAVEPOINT s; PRAGMA journal_mode=WAL; CREATE TABLE should_not_exist (value INTEGER); RELEASE s;",
    ],
)
def test_executescript_rejects_explicitly_mixed_autocommit_commands_before_execution(
    tmp_path: Path,
    script: str,
) -> None:
    store = Store(tmp_path / "executescript-explicit-mixed-autocommit.sqlite3")
    try:
        with pytest.raises(
            sqlite3.ProgrammingError,
            match="executescript_mixed_autocommit_and_transactional_unsupported",
        ):
            store.db.executescript(script)

        assert store.db.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal"
        assert store.db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'should_not_exist'"
        ).fetchone()[0] == 0
        assert not store.db.in_transaction
        assert store.db._store_transaction_owner is None
    finally:
        if store.db.in_transaction and store.db._store_transaction_owner == threading.get_ident():
            store.db.rollback()
        store.close()
