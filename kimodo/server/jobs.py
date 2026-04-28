# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job lifecycle management for asynchronous Kimodo generation."""

from __future__ import annotations

import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from .schemas import GenerationRequest, JobRecord, JobStatus
from .storage import JobStorage


class JobManager:
    """Small single-worker queue for GPU-backed generation jobs."""

    def __init__(self, *, runtime, storage: JobStorage, max_workers: int = 1) -> None:
        self.runtime = runtime
        self.storage = storage
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="kimodo-job")
        self._lock = Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future] = {}

    def submit(self, request: GenerationRequest) -> JobRecord:
        job_id = self.storage.create_job_id()
        job_dir = self.storage.create_job_dir(job_id)
        record = JobRecord(job_id=job_id, status=JobStatus.QUEUED, job_dir=str(job_dir))
        self.storage.write_request(job_id, request)
        self.storage.write_status(record)

        with self._lock:
            self._jobs[job_id] = record
            self._futures[job_id] = self.executor.submit(self._run_job, job_id, request)
        return record

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs.get(job_id)
        if record is not None:
            return record
        return self.storage.read_status(job_id)

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            future = self._futures.get(job_id)
            record = self._jobs.get(job_id)
            if record is None:
                record = self.storage.read_status(job_id)
            if future is not None and future.cancel():
                record.status = JobStatus.CANCELED
                self._jobs[job_id] = record
                self.storage.write_status(record)
        return record

    def _set_record(self, record: JobRecord) -> None:
        with self._lock:
            self._jobs[record.job_id] = record
        self.storage.write_status(record)

    def _run_job(self, job_id: str, request: GenerationRequest) -> None:
        record = self.get(job_id)
        record.status = JobStatus.RUNNING
        record.progress = 0.0
        record.message = "Generation started."
        self._set_record(record)

        try:
            result = self.runtime.generate(request, job_dir=self.storage.job_dir(job_id))
            record.status = JobStatus.SUCCEEDED
            record.progress = 1.0
            record.message = "Generation completed."
            record.artifacts = result.artifacts
        except Exception as error:
            record.status = JobStatus.FAILED
            record.message = f"{type(error).__name__}: {error}"
            record.error = traceback.format_exc()
        finally:
            self._set_record(record)
