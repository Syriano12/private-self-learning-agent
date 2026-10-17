from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from private_agent.core.orchestrator import Orchestrator
from private_agent.core.task_state import TaskState
from private_agent.storage import QueueLeaseLost, Store


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

    def start(self) -> None:
        self._renew_thread = threading.Thread(target=self._run, name="agent-lease-heartbeat", daemon=True)
        self._watchdog_thread = threading.Thread(target=self._watchdog, name="agent-lease-watchdog", daemon=True)
        self._renew_thread.start()
        self._watchdog_thread.start()

    def stop(self) -> bool:
        """Request shutdown and report whether all heartbeat threads actually stopped."""
        self._stop.set()
        deadline = time.monotonic() + max(2.0, self.interval_seconds * 2.0 + 1.0)
        for thread in (self._renew_thread, self._watchdog_thread):
            if thread is None:
                continue
            remaining = max(0.0, deadline - time.monotonic())
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


class BackgroundWorker:
    """Single-step durable worker; intentionally no run_forever loop yet."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        *,
        worker_id: str | None = None,
        lease_ttl_seconds: int = 60,
        heartbeat_interval_seconds: float | None = None,
        max_execution_age_seconds: float = 300.0,
    ) -> None:
        self.orchestrator = orchestrator
        self.store: Store = orchestrator.store
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

    def run_once(self) -> dict[str, Any]:
        """Claim and process at most one task, returning a serializable result."""
        claim = self.store.claim_next_task(
            self.worker_id,
            lease_ttl_seconds=self.lease_ttl_seconds,
        )
        if claim is None:
            return {"status": "NO_TASK", "processed": False, "task_id": ""}
        if claim.get("kind") == "EXECUTION_UNKNOWN":
            return {
                "status": "EXECUTION_UNKNOWN",
                "processed": False,
                "task_id": str(claim["task_id"]),
                "error_code": str(claim.get("error_code", "lease_expired_execution_unknown")),
            }

        task_id = str(claim["task_id"])
        claim = {**claim, "lease_ttl_seconds": self.lease_ttl_seconds}
        heartbeat = _LeaseHeartbeat(
            self.store,
            claim,
            interval_seconds=self.heartbeat_interval_seconds,
            max_execution_age_seconds=self.max_execution_age_seconds,
        )
        heartbeat.start()
        result: dict[str, Any] | None = None
        try:
            result = self._process_claim(claim, task_id, heartbeat)
        finally:
            if not heartbeat.stop() and result is not None:
                result["heartbeat_cleanup"] = "stuck"
                result["heartbeat_stuck"] = True
        if result is None:
            raise RuntimeError("worker_result_missing")
        return result

    def _process_claim(
        self,
        claim: dict[str, Any],
        task_id: str,
        heartbeat: _LeaseHeartbeat,
    ) -> dict[str, Any]:
        try:
            try:
                with self.store.lease_context(task_id, str(claim["lease_id"]), self.worker_id):
                    result = self.orchestrator.run(str(claim["goal"]), task_id=task_id)
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
            # Another worker may already have recovered the claim.  Its
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
