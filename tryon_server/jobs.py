"""In-memory job queue for try-on requests.

Why jobs instead of one long HTTP request: the free Cloudflare quick tunnel closes requests
that take longer than ~100 s, and a 50-step 1024x768 generation on a T4 can take about that
long. So the client submits a job, gets an id immediately, and polls for progress/result.
A single worker thread processes jobs in order because there is one GPU.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("tryon.jobs")

QUEUED, RUNNING, DONE, ERROR = "queued", "running", "done", "error"


@dataclass
class Job:
    id: str
    payload: dict
    status: str = QUEUED
    stage: str = "queued"
    step: int = 0
    total_steps: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    error_status: int = 500

    def progress(self, stage: str, step: int, total: int) -> None:
        self.stage, self.step, self.total_steps = stage, step, total

    def public_status(self, queue_position: int | None = None) -> dict:
        status = {
            "job_id": self.id,
            "status": self.status,
            "stage": self.stage,
            "step": self.step,
            "total_steps": self.total_steps,
            "queue_position": queue_position,
        }
        if self.started_at:
            status["elapsed_s"] = round((self.finished_at or time.time()) - self.started_at, 1)
        if self.error:
            status["error"] = self.error
            status["error_status"] = self.error_status
        return status


class JobManager:
    """FIFO queue + one worker thread. Keeps the most recent `max_jobs` jobs in memory."""

    def __init__(self, handler: Callable[[Job], Any], max_queue: int = 8, max_jobs: int = 50):
        self._handler = handler
        self._queue: "queue.Queue[Job]" = queue.Queue(maxsize=max_queue)
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._max_jobs = max_jobs
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="tryon-worker", daemon=True)
        self._worker.start()

    def submit(self, payload: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], payload=payload)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            raise QueueFullError("The server is busy (queue full). Please try again in a minute.")
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self._max_jobs:  # forget old jobs (and their images)
                self._jobs.pop(self._order.pop(0), None)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def queue_position(self, job: Job) -> Optional[int]:
        if job.status != QUEUED:
            return None
        with self._lock:
            waiting = [j for j in (self._jobs.get(i) for i in self._order) if j and j.status == QUEUED]
        return next((pos for pos, j in enumerate(waiting) if j.id == job.id), None)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            job.status, job.started_at = RUNNING, time.time()
            try:
                job.result = self._handler(job)
                job.status = DONE
            except Exception as exc:
                job.status = ERROR
                job.error = str(exc)
                job.error_status = getattr(exc, "http_status", 500)
                log.warning("Job %s failed: %s", job.id, exc)
            finally:
                job.finished_at = time.time()
                job.payload = {}  # free the uploaded images
                self._queue.task_done()


class QueueFullError(Exception):
    http_status = 503
