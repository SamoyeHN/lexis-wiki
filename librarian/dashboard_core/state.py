"""Thread-safe state manager for active background jobs and server lifecycle."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple


class DashboardState:
    """Global state to keep track of active background jobs and logging."""

    jobs: Dict[str, Dict[str, Any]] = {}
    lock = threading.Lock()

    # --- Server lifecycle (auto-shutdown when the browser is closed) ---
    last_activity_ts: Optional[float] = None  # timestamp of the last request received
    shutdown_requested_at: Optional[float] = None  # set by /api/shutdown (browser beforeunload beacon)

    @classmethod
    def touch(cls) -> None:
        """Record that the browser client is still alive (call on every request).
        Also cancels a pending shutdown: if a request arrives after the beacon,
        the client is coming back (e.g. an F5 refresh) and must not be killed.
        """
        with cls.lock:
            cls.last_activity_ts = time.time()
            cls.shutdown_requested_at = None

    @classmethod
    def mark_shutdown(cls) -> None:
        """Arm the shutdown (browser beforeunload beacon). The supervisor exits
        only after SHUTDOWN_GRACE elapses without any new request. The beacon
        request itself does NOT count as activity (it is the last words of a
        leaving page), so it resets last_activity_ts.
        """
        with cls.lock:
            cls.shutdown_requested_at = time.time()
            cls.last_activity_ts = None

    @classmethod
    def lifecycle(cls) -> Tuple[Optional[float], Optional[float], bool]:
        """Return (shutdown_requested_at, last_activity_ts, has_running_jobs) under one lock."""
        with cls.lock:
            has_running = any(j.get("status") == "running" for j in cls.jobs.values())
            return cls.shutdown_requested_at, cls.last_activity_ts, has_running

    @classmethod
    def create_job(cls, job_type: str, target: str) -> str:
        with cls.lock:
            job_id = f"{job_type}_{uuid.uuid4().hex[:8]}"
            initial_logs = (
                [f"🔨 Preparing build for '{target}'..."]
                if job_type == "compile"
                else [
                    f"[{job_type.upper()}] Job created for '{target}'",
                    f"[{job_type.upper()}] Launching background worker thread...",
                ]
            )
            cls.jobs[job_id] = {
                "id": job_id,
                "type": job_type,
                "target": target,
                "status": "running",
                "progress": 10,
                "stage": "starting",
                "audit_progress": 0,
                "audit_info": None,
                "logs": initial_logs,
                "saved_files": [],
                "result": None,
            }
            return job_id

    @classmethod
    def update_job(
        cls,
        job_id: str,
        progress: Optional[int] = None,
        status: Optional[str] = None,
        log: Optional[str] = None,
        saved_files: Optional[List[str]] = None,
        result: Optional[Any] = None,
        stage: Optional[str] = None,
        audit_progress: Optional[int] = None,
        audit_info: Optional[str] = None,
    ) -> None:
        with cls.lock:
            if job_id in cls.jobs:
                if progress is not None:
                    cls.jobs[job_id]["progress"] = progress
                if status is not None:
                    cls.jobs[job_id]["status"] = status
                if log is not None:
                    cls.jobs[job_id]["logs"].append(log)
                if saved_files is not None:
                    cls.jobs[job_id]["saved_files"] = saved_files
                if result is not None:
                    cls.jobs[job_id]["result"] = result
                if stage is not None:
                    cls.jobs[job_id]["stage"] = stage
                if audit_progress is not None:
                    cls.jobs[job_id]["audit_progress"] = audit_progress
                if audit_info is not None:
                    cls.jobs[job_id]["audit_info"] = audit_info

    @classmethod
    def get_job(cls, job_id: str) -> Optional[Dict[str, Any]]:
        with cls.lock:
            return cls.jobs.get(job_id)

    @classmethod
    def get_recent_jobs(cls) -> List[Dict[str, Any]]:
        with cls.lock:
            running = [j for j in cls.jobs.values() if j.get("status") == "running"]
            finished = [j for j in cls.jobs.values() if j.get("status") != "running"]
            return finished[-10:] + running
