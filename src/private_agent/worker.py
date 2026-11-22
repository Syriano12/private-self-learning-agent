from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from private_agent.core.orchestrator import Orchestrator
from private_agent.core.task_state import TaskState
from private_agent.bugbounty.programs import ProgramAnalysisRepository
from private_agent.bugbounty.research import ResearchStageRepository, ResearchStageStatus
from private_agent.bugbounty.stage_execution import ResearchStageExecutor, StageExecutionOutcome
from private_agent.storage import QueueLeaseLost, Store, StoreClosingError


LOGGER = logging.getLogger(__name__)
WorkerEventLogger = Callable[[dict[str, Any]], None]


class _LeaseHeartbeat:
    """Renew one claim while work runs, with a bounded execution-age watchdog."""

    def __init__(
        self,
        store: Store,
        claim: dict[str, Any],
        *,
        interval_seconds: float,
        max_execution_age_seconds: float,
    ) -> None:
        self.store = store
        self.claim = claim
        self.interval_seconds = float(interval_seconds)
        self.max_execution_age_seconds = float(max_execution_age_seconds)
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._renew_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._cleanup_stuck = False

    @property
    def cleanup_stuck(self) -> bool:
        return self._cleanup_stuck

    @property
    def is_alive(self) -> bool:
        return any(
            thread is not None and thread.is_alive()
            for thread in (self._renew_thread, self._watchdog_thread)
        )

    def start(self) -> None:
        self._renew_thread = threading.Thread(target=self._run, name="agent-lease-heartbeat", daemon=True)
        self._watchdog_thread = threading.Thread(target=self._watchdog, name="agent-lease-watchdog", daemon=True)
        self._renew_thread.start()
        self._watchdog_thread.start()

    def stop(self, *, wait_forever: bool = False) -> bool:
        """Request shutdown and report whether all heartbeat threads actually stopped."""
        self._stop.set()
        deadline = None if wait_forever else time.monotonic() + max(2.0, self.interval_seconds * 2.0 + 1.0)
        for thread in (self._renew_thread, self._watchdog_thread):
            if thread is None:
                continue
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
        self._cleanup_stuck = any(
            thread is not None and thread.is_alive()
            for thread in (self._renew_thread, self._watchdog_thread)
        )
        return not self._cleanup_stuck

    def raise_if_lost(self) -> None:
        if not self.lost.is_set():
            return
        if isinstance(self._error, QueueLeaseLost):
            raise self._error
        detail = type(self._error).__name__ if self._error is not None else "unknown"
        raise QueueLeaseLost(f"queue_lease_heartbeat_failed:{detail}") from self._error

    def _run(self) -> None:
        heartbeat_store: Store | None = None
        try:
            # Never share the orchestrator's sqlite3 connection across threads.
            heartbeat_store = Store(self.store.path)
            while not self._stop.wait(self.interval_seconds):
                try:
                    heartbeat_store.renew_task_lease(
                        str(self.claim["task_id"]),
                        str(self.claim["lease_id"]),
                        str(self.claim["worker_id"]),
                        lease_ttl_seconds=int(self.claim["lease_ttl_seconds"]),
                    )
                except BaseException as exc:
                    self._error = exc
                    self.lost.set()
                    return
        except BaseException as exc:
            self._error = exc
            self.lost.set()
        finally:
            if heartbeat_store is not None:
                heartbeat_store.close()

    def _watchdog(self) -> None:
        watchdog_store: Store | None = None
        try:
            if self._stop.wait(self.max_execution_age_seconds):
                return
            watchdog_store = Store(self.store.path)
            try:
                watchdog_store.mark_task_execution_unknown_if_owned(
                    self.claim,
                    reason="max_execution_age_exceeded",
                )
            except BaseException as exc:
                self._error = exc
            else:
                self._error = QueueLeaseLost("queue_lease_max_execution_age")
            self.lost.set()
            # Stop future renewals. A renewal call already blocked in SQLite
            # cannot be forcibly cancelled; stop() reports that explicitly.
            self._stop.set()
        except BaseException as exc:
            self._error = exc
            self.lost.set()
            self._stop.set()
        finally:
            if watchdog_store is not None:
                watchdog_store.close()


class _ResearchStageLeaseLost(ValueError):
    pass


class _ResearchStageLeaseHeartbeat:
    """Renew a claimed research stage without sharing SQLite connections."""

    def __init__(self, store: Store, claim: dict[str, Any], *, interval_seconds: float, lease_ttl_seconds: int) -> None:
        self.store = store
        self.claim = claim
        self.interval_seconds = float(interval_seconds)
        self.lease_ttl_seconds = int(lease_ttl_seconds)
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="agent-stage-lease-heartbeat", daemon=True)
        self._thread.start()

    def stop(self, *, wait_forever: bool = False) -> bool:
        self._stop.set()
        if self._thread is not None:
            timeout = None if wait_forever else max(2.0, self.interval_seconds * 2.0 + 1.0)
            self._thread.join(timeout=timeout)
        return not self.is_alive

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def raise_if_lost(self) -> None:
        if not self.lost.is_set():
            return
        raise _ResearchStageLeaseLost("stage_lease_heartbeat_failed") from self._error

    def _run(self) -> None:
        heartbeat_store: Store | None = None
        try:
            heartbeat_store = Store(self.store.path)
            repository = ResearchStageRepository(heartbeat_store)
            while not self._stop.wait(self.interval_seconds):
                if not repository.renew_claim(self.claim, lease_ttl_seconds=self.lease_ttl_seconds):
                    self._error = _ResearchStageLeaseLost("stage_lease_renewal_failed")
                    self.lost.set()
                    return
        except BaseException as exc:
            self._error = exc
            self.lost.set()
        finally:
            if heartbeat_store is not None:
                heartbeat_store.close()


class BackgroundWorker:
    """Controlled durable worker with one loop per instance.

    ``run_once`` remains the single-claim primitive. ``run_forever`` adds the
    operational lifecycle around it: bounded polling, recoverable-error
    backoff, structured lifecycle events, and cooperative shutdown. It never
    cancels an in-flight external operation; the existing lease fencing and
    ``EXECUTION_UNKNOWN`` rules remain authoritative.
    """

    _RECOVERABLE_ERRORS = (QueueLeaseLost, sqlite3.OperationalError, TimeoutError)

    def __init__(
        self,
        orchestrator: Orchestrator,
        *,
        worker_id: str | None = None,
        lease_ttl_seconds: int = 60,
        heartbeat_interval_seconds: float | None = None,
        max_execution_age_seconds: float = 300.0,
        poll_interval_seconds: float = 0.1,
        idle_poll_interval_seconds: float = 1.0,
        error_backoff_seconds: float = 1.0,
        event_logger: WorkerEventLogger | None = None,
        stage_executor: ResearchStageExecutor | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.store: Store = orchestrator.store
        self.research_stages = ResearchStageRepository(self.store)
        self.program_analyses = ProgramAnalysisRepository(self.store)
        self.stage_executor = stage_executor or ResearchStageExecutor()
        self.worker_id = str(worker_id or f"worker-{uuid.uuid4()}")
        self.lease_ttl_seconds = int(lease_ttl_seconds)
        if self.lease_ttl_seconds < 1:
            raise ValueError("lease_ttl_invalid")
        default_interval = max(0.05, min(self.lease_ttl_seconds / 3.0, 5.0))
        self.heartbeat_interval_seconds = float(
            default_interval if heartbeat_interval_seconds is None else heartbeat_interval_seconds
        )
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_invalid")
        self.max_execution_age_seconds = float(max_execution_age_seconds)
        if self.max_execution_age_seconds <= 0:
            raise ValueError("max_execution_age_invalid")
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.idle_poll_interval_seconds = float(idle_poll_interval_seconds)
        self.error_backoff_seconds = float(error_backoff_seconds)
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_invalid")
        if self.idle_poll_interval_seconds <= 0:
            raise ValueError("idle_poll_interval_invalid")
        if self.error_backoff_seconds <= 0:
            raise ValueError("error_backoff_invalid")
        self.event_logger = event_logger
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._lifecycle = threading.Condition(self._lifecycle_lock)
        self._running = False
        self._loop_thread: threading.Thread | None = None
        self._active_run_once = 0
        self._heartbeat_lock = threading.Lock()
        self._active_heartbeats: set[Any] = set()
        self._last_report: dict[str, Any] | None = None
        self._fatal_error: BaseException | None = None

    @property
    def running(self) -> bool:
        with self._lifecycle_lock:
            return self._running

    @property
    def last_report(self) -> dict[str, Any] | None:
        with self._lifecycle_lock:
            return dict(self._last_report) if self._last_report is not None else None

    @property
    def fatal_error(self) -> BaseException | None:
        with self._lifecycle_lock:
            return self._fatal_error

    def start(
        self,
        *,
        stop_event: threading.Event | None = None,
        daemon: bool = True,
    ) -> threading.Thread:
        """Start the loop on a thread and return that thread.

        Starting a second loop on the same worker instance is rejected. The
        returned thread is intentionally observable so callers can join it.
        """
        with self._lifecycle_lock:
            if self._running:
                raise RuntimeError("worker_already_running")
            if self._active_run_once:
                raise RuntimeError("worker_run_once_active")
            if self._loop_thread is not None and self._loop_thread.is_alive():
                raise RuntimeError("worker_already_running")
            self._loop_thread = None
            self._stop_event.clear()
            self._fatal_error = None
            self._running = True
            thread = threading.Thread(
                target=self._run_registered_loop,
                args=(stop_event,),
                name=f"agent-worker-{self.worker_id}",
                daemon=daemon,
            )
            self._loop_thread = thread
            try:
                self.store.register_background_worker(self)
            except BaseException:
                self._running = False
                self._loop_thread = None
                raise
        try:
            thread.start()
        except BaseException:
            with self._lifecycle:
                self._running = False
                self._loop_thread = None
                self._lifecycle.notify_all()
            self.store.unregister_background_worker(self)
            raise
        with self._lifecycle:
            self._lifecycle.notify_all()
        return thread

    def request_stop(self) -> None:
        """Set the cooperative stop signal without waiting for active work."""
        self._stop_event.set()

    def join(self, *, timeout: float | None = 5.0) -> bool:
        """Wait for the actual worker thread; ``None`` means drain without a deadline."""
        if timeout is not None and timeout < 0:
            raise ValueError("shutdown_timeout_invalid")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._lifecycle:
            thread = self._loop_thread
            if thread is threading.current_thread():
                return False
            while thread is not None and thread.ident is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._lifecycle.wait(timeout=remaining)
                thread = self._loop_thread
        if thread is not None:
            thread.join(timeout=None if deadline is None else max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                return False
        with self._lifecycle:
            while self._active_run_once:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._lifecycle.wait(timeout=remaining)
            if thread is not None and self._loop_thread is thread:
                self._loop_thread = None
            return not self._running

    def stop(self, *, timeout: float | None = 5.0) -> bool:
        """Request cooperative shutdown and wait for at most ``timeout`` seconds."""
        if timeout is not None and timeout < 0:
            raise ValueError("shutdown_timeout_invalid")
        self.request_stop()
        return self.join(timeout=timeout)

    def _track_heartbeat(self, heartbeat: Any) -> None:
        with self._heartbeat_lock:
            self._active_heartbeats.add(heartbeat)

    def _untrack_heartbeat(self, heartbeat: Any) -> None:
        with self._heartbeat_lock:
            self._active_heartbeats.discard(heartbeat)

    def _drain_heartbeats(self) -> None:
        while True:
            with self._heartbeat_lock:
                active = tuple(self._active_heartbeats)
            if not active:
                return
            for heartbeat in active:
                heartbeat.stop(wait_forever=True)
                if not heartbeat.is_alive:
                    self._untrack_heartbeat(heartbeat)

    def _run_registered_loop(self, external_stop_event: threading.Event | None) -> dict[str, Any]:
        self.store.enter_background_worker(self)
        try:
            return self._run_loop(external_stop_event)
        finally:
            try:
                self._drain_heartbeats()
            finally:
                self.store.leave_background_worker(self)

    def run_forever(self, *, stop_event: threading.Event | None = None) -> dict[str, Any]:
        """Run until stopped, returning a serializable lifecycle report.

        Fatal exceptions are recorded and re-raised. Recoverable queue/lease
        errors are reported and retried after bounded backoff.
        """
        with self._lifecycle_lock:
            if self._running:
                raise RuntimeError("worker_already_running")
            if self._active_run_once:
                raise RuntimeError("worker_run_once_active")
            if self._loop_thread is not None and self._loop_thread.is_alive():
                raise RuntimeError("worker_already_running")
            self._loop_thread = None
            self._stop_event.clear()
            self._fatal_error = None
            self._running = True
            self._loop_thread = threading.current_thread()
            try:
                self.store.register_background_worker(self)
            except BaseException:
                self._running = False
                self._loop_thread = None
                raise
        try:
            self.store.enter_background_worker(self)
        except BaseException:
            self.store.unregister_background_worker(self)
            with self._lifecycle_lock:
                self._running = False
                self._loop_thread = None
            raise
        try:
            return self._run_loop(stop_event)
        finally:
            try:
                self._drain_heartbeats()
            finally:
                self.store.leave_background_worker(self)
                with self._lifecycle_lock:
                    if self._loop_thread is threading.current_thread():
                        self._loop_thread = None

    def _run_loop(self, external_stop_event: threading.Event | None) -> dict[str, Any]:
        report: dict[str, Any] = {
            "status": "RUNNING",
            "worker_id": self.worker_id,
            "iterations": 0,
            "processed": 0,
            "idle_polls": 0,
            "recoverable_errors": 0,
            "lease_losses": 0,
            "execution_unknown": 0,
            "last_error_type": "",
            "last_error_code": "",
            "fatal_error_type": "",
        }
        self._emit_event("worker_started")
        try:
            while not self._stop_requested(external_stop_event):
                report["iterations"] += 1
                try:
                    result = self.run_once()
                except self._RECOVERABLE_ERRORS as exc:
                    report["recoverable_errors"] += 1
                    report["last_error_type"] = type(exc).__name__
                    report["last_error_code"] = self._error_code(exc)
                    self._emit_event(
                        "worker_recoverable_error",
                        error_type=type(exc).__name__,
                        error_code=self._error_code(exc),
                    )
                    if self._wait_for(self.error_backoff_seconds, external_stop_event):
                        break
                    continue
                except StoreClosingError:
                    if self._stop_requested(external_stop_event) or self.store.closing:
                        break
                    raise
                except BaseException as exc:
                    report["status"] = "FATAL"
                    report["fatal_error_type"] = type(exc).__name__
                    with self._lifecycle_lock:
                        self._fatal_error = exc
                    self._emit_event("worker_fatal_error", error_type=type(exc).__name__)
                    raise

                status = str(result.get("status", ""))
                task_id = str(result.get("task_id", ""))
                if status == "NO_TASK":
                    report["idle_polls"] += 1
                    self._emit_event("worker_idle", idle_polls=report["idle_polls"])
                    delay = self.idle_poll_interval_seconds
                else:
                    if bool(result.get("processed")):
                        report["processed"] += 1
                    if status == "LEASE_LOST":
                        report["lease_losses"] += 1
                        self._emit_event(
                            "worker_lease_lost",
                            task_id=task_id,
                            error_code=str(result.get("error_code", "queue_lease_lost")),
                        )
                    elif status == "EXECUTION_UNKNOWN":
                        report["execution_unknown"] += 1
                        self._emit_event("worker_execution_unknown", task_id=task_id)
                    self._emit_event(
                        "worker_result",
                        task_id=task_id,
                        stage_id=str(result.get("stage_id", "")),
                        status=status,
                        processed=bool(result.get("processed")),
                        error_code=str(result.get("error_code", "")),
                    )
                    delay = self.poll_interval_seconds
                if self.store.closing:
                    self._emit_event("worker_store_closing")
                    break
                if self._wait_for(delay, external_stop_event):
                    break
            report["status"] = "STOPPED"
            return report
        finally:
            if report["status"] == "RUNNING":
                report["status"] = "STOPPED"
            with self._lifecycle_lock:
                self._last_report = dict(report)
                self._running = False
            self._emit_event("worker_stopped", status=str(report["status"]))

    def _stop_requested(self, external_stop_event: threading.Event | None) -> bool:
        return (
            self._stop_event.is_set()
            or bool(external_stop_event is not None and external_stop_event.is_set())
            or self.store.closing
        )

    def _wait_for(self, delay: float, external_stop_event: threading.Event | None) -> bool:
        delay = max(0.0, float(delay))
        if self._stop_requested(external_stop_event):
            return True
        deadline = time.monotonic() + delay
        while not self._stop_requested(external_stop_event):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._stop_requested(external_stop_event)
            self._stop_event.wait(min(remaining, 0.05))
            if external_stop_event is not None and external_stop_event.wait(0):
                return True
        return True

    def _emit_event(self, event_type: str, **fields: Any) -> None:
        event = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "worker_id": self.worker_id,
            **fields,
        }
        try:
            if self.event_logger is not None:
                self.event_logger(dict(event))
            else:
                LOGGER.info("background_worker_event", extra={"worker_event": event})
        except Exception:
            # Diagnostics must never change authorization or task outcome.
            LOGGER.exception("background_worker_event_logger_failed")

    @staticmethod
    def _error_code(exc: BaseException) -> str:
        return str(exc).split(":", 1)[0][:120] or type(exc).__name__

    def run_once(self) -> dict[str, Any]:
        """Claim and process at most one task, returning a serializable result."""
        with self._lifecycle:
            if self._active_run_once:
                raise RuntimeError("worker_run_once_already_active")
            if self._running and self._loop_thread is not threading.current_thread():
                raise RuntimeError("worker_already_running")
            self._active_run_once += 1
        try:
            with self.store.background_worker_operation(self):
                return self._run_once()
        finally:
            with self._lifecycle:
                self._active_run_once -= 1
                self._lifecycle.notify_all()

    def _run_once(self) -> dict[str, Any]:
        """Execute one task while the caller owns a Store lifecycle registration."""
        if self.store.closing:
            return {"status": "STOPPED", "processed": False, "task_id": ""}
        self._reconcile_approved_waiting_tasks()
        claim = self.store.claim_next_task(
            self.worker_id,
            lease_ttl_seconds=self.lease_ttl_seconds,
        )
        if claim is None:
            stage_result = self._run_one_research_stage()
            if stage_result is not None:
                return stage_result
            return {"status": "NO_TASK", "processed": False, "task_id": ""}
        if claim.get("kind") == "EXECUTION_UNKNOWN":
            task_id = str(claim["task_id"])
            self._emit_event(
                "worker_execution_unknown_recovered",
                task_id=task_id,
                error_code=str(claim.get("error_code", "lease_expired_execution_unknown")),
            )
            return {
                "status": "EXECUTION_UNKNOWN",
                "processed": False,
                "task_id": task_id,
                "error_code": str(claim.get("error_code", "lease_expired_execution_unknown")),
            }
        if claim.get("kind") != "CLAIMED":
            task_id = str(claim.get("task_id", ""))
            status = str(claim.get("queue_status") or claim.get("kind") or "BLOCKED")
            self._emit_event(
                "worker_non_executable_claim",
                task_id=task_id,
                status=status,
                error_code=str(claim.get("error_code", "queue_state_not_executable")),
            )
            return {
                "status": status,
                "processed": False,
                "task_id": task_id,
                "error_code": str(claim.get("error_code", "queue_state_not_executable")),
            }

        task_id = str(claim["task_id"])
        self._emit_event(
            "worker_task_claimed",
            task_id=task_id,
            attempt_count=int(claim["attempt_count"]),
        )
        claim = {**claim, "lease_ttl_seconds": self.lease_ttl_seconds}
        heartbeat = _LeaseHeartbeat(
            self.store,
            claim,
            interval_seconds=self.heartbeat_interval_seconds,
            max_execution_age_seconds=self.max_execution_age_seconds,
        )
        self._track_heartbeat(heartbeat)
        heartbeat.start()
        result: dict[str, Any] | None = None
        try:
            result = self._process_claim(claim, task_id, heartbeat)
        finally:
            if not heartbeat.stop() and result is not None:
                result["heartbeat_cleanup"] = "stuck"
                result["heartbeat_stuck"] = True
            if not heartbeat.is_alive:
                self._untrack_heartbeat(heartbeat)
        if result is None:
            raise RuntimeError("worker_result_missing")
        return result

    def _run_one_research_stage(self) -> dict[str, Any] | None:
        claim = self.research_stages.claim_next(self.worker_id, lease_ttl_seconds=self.lease_ttl_seconds)
        if claim is None:
            return None
        stage_id = str(claim["stage_id"])
        owner_user_id = str(claim["owner_user_id"])
        stage = self.research_stages.get(stage_id, owner_user_id=owner_user_id)
        if stage is None:
            self.research_stages.mark_unknown_if_owned(claim, reason="stage_record_missing")
            return {
                "status": "STAGE_UNKNOWN",
                "processed": False,
                "task_id": "",
                "stage_id": stage_id,
                "error_code": "stage_record_missing",
            }
        self._emit_event(
            "worker_stage_claimed",
            stage_id=stage_id,
            stage_type=stage.stage_type,
            attempt_count=int(claim.get("attempt_count", 1)),
        )
        analysis = self.program_analyses.get(
            str(claim["analysis_id"]),
            owner_user_id=owner_user_id,
            is_owner=True,
        )
        started = time.monotonic()
        stage_heartbeat = _ResearchStageLeaseHeartbeat(
            self.store,
            claim,
            interval_seconds=min(self.heartbeat_interval_seconds, max(0.05, self.lease_ttl_seconds / 3.0)),
            lease_ttl_seconds=self.lease_ttl_seconds,
        )
        self._track_heartbeat(stage_heartbeat)
        stage_heartbeat.start()
        try:
            outcome = self.stage_executor.execute(stage, analysis)
            if time.monotonic() - started > stage.budget.timeout_seconds:
                outcome = StageExecutionOutcome(
                    ResearchStageStatus.UNKNOWN.value,
                    {},
                    {},
                    "stage_timeout",
                    "Stage exceeded its bounded execution time",
                )
            stage_heartbeat.raise_if_lost()
            finished = self.research_stages.finish_claim(
                claim,
                outcome.status,
                result=outcome.result,
                partial_result=outcome.partial_result,
                error_code=outcome.error_code,
                error_message=outcome.error_message,
            )
        except _ResearchStageLeaseLost:
            self.research_stages.mark_unknown_if_owned(claim, reason="stage_lease_heartbeat_failed")
            return {
                "status": "STAGE_LEASE_LOST",
                "processed": False,
                "task_id": "",
                "stage_id": stage_id,
                "error_code": "stage_lease_heartbeat_failed",
            }
        except ValueError as exc:
            if str(exc) == "stage_lease_lost":
                self.research_stages.mark_unknown_if_owned(claim, reason="stage_lease_lost")
                return {
                    "status": "STAGE_LEASE_LOST",
                    "processed": False,
                    "task_id": "",
                    "stage_id": stage_id,
                    "error_code": "stage_lease_lost",
                }
            self.research_stages.mark_unknown_if_owned(claim, reason="stage_worker_failed")
            return {
                "status": "STAGE_FAILED",
                "processed": True,
                "task_id": "",
                "stage_id": stage_id,
                "error_code": "stage_worker_failed",
                "error": str(exc),
            }
        except Exception as exc:
            self.research_stages.mark_unknown_if_owned(claim, reason="stage_worker_failed")
            return {
                "status": "STAGE_UNKNOWN",
                "processed": False,
                "task_id": "",
                "stage_id": stage_id,
                "error_code": "stage_worker_failed",
                "error": str(exc),
            }
        finally:
            stage_heartbeat.stop()
            if not stage_heartbeat.is_alive:
                self._untrack_heartbeat(stage_heartbeat)
        status = "STAGE_" + finished.stage_status
        self._emit_event(
            "worker_stage_result",
            stage_id=stage_id,
            stage_type=finished.stage_type,
            status=status,
            error_code=finished.error_code,
        )
        return {
            "status": status,
            "processed": True,
            "task_id": "",
            "stage_id": stage_id,
            "error_code": finished.error_code,
        }

    def _reconcile_approved_waiting_tasks(self) -> None:
        """Recover approval transitions persisted just before a process stop.

        This is intentionally a queue-only transition. It does not approve,
        resume, or execute an action; the worker still re-enters the normal
        orchestrator and execution-boundary checks after claiming the task.
        """
        for task_id in self.store.waiting_approval_task_ids(limit=100):
            try:
                self.store.requeue_waiting_approval_task(task_id)
            except QueueLeaseLost:
                # Another worker/recovery path won the transition.
                continue
            except ValueError as exc:
                if str(exc) in {"approval_not_approved", "task_not_waiting_approval"}:
                    continue
                if str(exc) in {"approval_policy_stale", "approval_policy_unavailable"}:
                    self._emit_event(
                        "worker_approval_requeue_deferred",
                        task_id=task_id,
                        error_code=str(exc),
                    )
                    continue
                raise

    def _process_claim(
        self,
        claim: dict[str, Any],
        task_id: str,
        heartbeat: _LeaseHeartbeat,
    ) -> dict[str, Any]:
        # Keep the lease context until the queue claim is durably finished. A
        # terminal task state alone does not mean finish_task_claim has run.
        with self.store.lease_context(task_id, str(claim["lease_id"]), self.worker_id):
            return self._process_claim_with_lease(claim, task_id, heartbeat)

    def _process_claim_with_lease(
        self,
        claim: dict[str, Any],
        task_id: str,
        heartbeat: _LeaseHeartbeat,
    ) -> dict[str, Any]:
        try:
            try:
                with self.store.lease_context(task_id, str(claim["lease_id"]), self.worker_id):
                    task_payload = self.store.get_task_state(task_id)
                    task_context = None
                    if isinstance(task_payload, dict):
                        persisted_state = TaskState.from_dict(task_payload)
                        if persisted_state.context.get("task_kind") == "bug_bounty":
                            task_context = dict(persisted_state.context)
                    result = self.orchestrator.run(str(claim["goal"]), task_id=task_id, task_context=task_context)
                    heartbeat.raise_if_lost()
                    payload = self.store.get_task_state(task_id)
                    heartbeat.raise_if_lost()
                state = TaskState.from_dict(payload) if payload is not None else None
                queue_status = self._queue_status(state, result)
                heartbeat.raise_if_lost()
                self.store.finish_task_claim(claim, queue_status)
                return {
                    "status": queue_status,
                    "processed": True,
                    "task_id": task_id,
                    "attempt_count": int(claim["attempt_count"]),
                    "result": result,
                }
            except QueueLeaseLost as exc:
                self._fence_lost_claim(claim)
                return {
                    "status": "LEASE_LOST",
                    "processed": False,
                    "task_id": task_id,
                    "error_code": "queue_lease_lost",
                    "error": str(exc),
                }
            except Exception as exc:
                try:
                    heartbeat.raise_if_lost()
                except QueueLeaseLost as lease_exc:
                    self._fence_lost_claim(claim)
                    return {
                        "status": "LEASE_LOST",
                        "processed": False,
                        "task_id": task_id,
                        "error_code": "queue_lease_lost",
                        "error": str(lease_exc),
                    }
                payload = self.store.get_task_state(task_id)
                state = TaskState.from_dict(payload) if payload is not None else None
                if state is not None and state.status in {"COMPLETED", "FAILED", "BLOCKED"}:
                    # The durable state is already terminal; do not relabel a
                    # completed external action as a worker failure because a
                    # post-terminal reflection/learning step raised.
                    try:
                        self.store.finish_task_claim(claim, state.status)
                    except QueueLeaseLost as lease_exc:
                        self._fence_lost_claim(claim)
                        return {
                            "status": "LEASE_LOST",
                            "processed": False,
                            "task_id": task_id,
                            "error_code": "queue_lease_lost",
                            "error": str(lease_exc),
                        }
                    return {
                        "status": state.status,
                        "processed": True,
                        "task_id": task_id,
                        "attempt_count": int(claim["attempt_count"]),
                        "error_code": "post_terminal_worker_error",
                        "error": str(exc),
                    }
                uncertain = self._execution_may_have_started(state)
                queue_status = "EXECUTION_UNKNOWN" if uncertain else "FAILED"
                try:
                    self.store.finish_task_claim(
                        claim,
                        queue_status,
                        error_code="execution_unknown" if uncertain else "worker_failed",
                        error_message=str(exc),
                        state_status=queue_status,
                    )
                except QueueLeaseLost as lease_exc:
                    self._fence_lost_claim(claim)
                    return {
                        "status": "LEASE_LOST",
                        "processed": False,
                        "task_id": task_id,
                        "error_code": "queue_lease_lost",
                        "error": str(lease_exc),
                    }
                return {
                    "status": queue_status,
                    "processed": True,
                    "task_id": task_id,
                    "attempt_count": int(claim["attempt_count"]),
                    "error_code": "execution_unknown" if uncertain else "worker_failed",
                    "error": str(exc),
                }
        except BaseException:
            # Preserve the original exception while allowing run_once() to
            # report heartbeat cleanup when possible.
            raise

    def _fence_lost_claim(self, claim: dict[str, Any]) -> None:
        """Persist a conservative outcome when the worker loses its lease."""
        try:
            self.store.mark_task_execution_unknown_if_owned(
                claim,
                reason="worker_lease_lost",
            )
        except QueueLeaseLost:
            # Another worker may already have recovered the claim. Its
            # recovery transaction owns the authoritative outcome.
            return

    @staticmethod
    def _execution_may_have_started(state: TaskState | None) -> bool:
        if state is None:
            return False
        if state.status in {"EXECUTING", "OBSERVING", "VERIFYING", "EXECUTION_UNKNOWN"}:
            return True
        return any(
            record.get("status") in {"EXECUTING", "OBSERVING"}
            for record in state.action_records.values()
        )

    @staticmethod
    def _queue_status(state: TaskState | None, result: dict[str, Any]) -> str:
        if state is not None:
            if state.status == "WAITING_APPROVAL":
                return "WAITING_APPROVAL"
            if state.status in {"COMPLETED", "FAILED", "BLOCKED", "EXECUTION_UNKNOWN"}:
                return state.status
        final_status = str(result.get("final_status", "FAILED"))
        return {
            "COMPLETED": "COMPLETED",
            "BLOCKED": "BLOCKED",
            "FAILED": "FAILED",
            "EXECUTION_UNKNOWN": "EXECUTION_UNKNOWN",
        }.get(final_status, "FAILED")
