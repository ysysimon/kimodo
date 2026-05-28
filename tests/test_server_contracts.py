# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for server schemas, storage, jobs, and app wiring."""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Event

import pytest

from kimodo.server.app import create_app
from kimodo.server.jobs import JobManager
from kimodo.server.runtime import FakeRuntime, TextEncoderServerConfig
from kimodo.server.schemas import ArtifactRecord, GenerationRequest, GenerationResult, JobRecord, JobStatus
from kimodo.server.storage import JobStorage

pytestmark = pytest.mark.server


def test_generation_request_to_dict_preserves_defaults():
    request = GenerationRequest(texts=["walk"], durations=[1.0])

    data = request.to_dict()

    assert data["texts"] == ["walk"]
    assert data["durations"] == [1.0]
    assert data["diffusion_steps"] == 100
    assert data["num_samples"] == 1
    assert data["formats"] == ["npz", "bvh"]
    assert data["bvh_standard_tpose"] is True
    assert data["zip_output"] is False
    assert data["postprocess"] is True
    assert data["output_world_offset"] is None


def test_generation_request_to_dict_preserves_constraints():
    constraints = [
        {
            "type": "root2d",
            "frame_indices": [0],
            "smooth_root_2d": [[0.0, 0.0]],
        }
    ]
    request = GenerationRequest(texts=["walk"], durations=[1.0], constraints=constraints)

    data = request.to_dict()

    assert data["constraints"] == constraints


def test_generation_request_to_dict_preserves_output_world_offset():
    request = GenerationRequest(texts=["walk"], durations=[1.0], output_world_offset=[1.0, 0.0, -2.0])

    data = request.to_dict()

    assert data["output_world_offset"] == [1.0, 0.0, -2.0]


def test_job_record_round_trips_status_and_artifact_metadata():
    record = JobRecord(
        job_id="job-1",
        status=JobStatus.SUCCEEDED,
        job_dir="/tmp/job-1",
        progress=1.0,
        artifacts={
            "npz": ArtifactRecord(
                key="npz",
                filename="motion.npz",
                content_type="application/octet-stream",
                size_bytes=12,
                download_url="/jobs/job-1/artifacts/npz",
            )
        },
    )

    data = record.to_dict()
    restored = JobRecord.from_dict(data)

    assert data["status"] == "succeeded"
    assert restored.status == JobStatus.SUCCEEDED
    assert restored.artifacts["npz"].filename == "motion.npz"
    assert restored.artifacts["npz"].download_url == "/jobs/job-1/artifacts/npz"


def test_storage_uses_environment_root(monkeypatch, tmp_path):
    monkeypatch.setenv("KIMODO_JOB_ROOT", str(tmp_path))

    storage = JobStorage()

    assert storage.root == tmp_path


def test_storage_writes_request_and_status_round_trip(tmp_path):
    storage = JobStorage(str(tmp_path))
    job_id = "job-1"
    storage.create_job_dir(job_id)
    constraints = [
        {
            "type": "root2d",
            "frame_indices": [0],
            "smooth_root_2d": [[0.0, 0.0]],
        }
    ]
    request = GenerationRequest(texts=["walk"], durations=[1.0], constraints=constraints)
    record = JobRecord(job_id=job_id, status=JobStatus.QUEUED, job_dir=str(storage.job_dir(job_id)))

    storage.write_request(job_id, request)
    storage.write_status(record)
    restored = storage.read_status(job_id)
    request_data = json.loads((storage.job_dir(job_id) / "request.json").read_text(encoding="utf-8"))

    assert request.job_id == job_id
    assert (storage.job_dir(job_id) / "request.json").is_file()
    assert request_data["constraints"] == constraints
    assert restored.status == JobStatus.QUEUED
    assert restored.job_dir == str(storage.job_dir(job_id))


def test_storage_resolves_artifact_successfully(tmp_path):
    storage = JobStorage(str(tmp_path))
    job_id = "job-1"
    artifact_path = storage.artifacts_dir(job_id) / "motion.npz"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("placeholder", encoding="utf-8")
    storage.write_status(
        JobRecord(
            job_id=job_id,
            status=JobStatus.SUCCEEDED,
            job_dir=str(storage.job_dir(job_id)),
            artifacts={
                "npz": ArtifactRecord(
                    key="npz",
                    filename="motion.npz",
                    content_type="application/octet-stream",
                    size_bytes=artifact_path.stat().st_size,
                )
            },
        )
    )

    path, artifact = storage.resolve_artifact(job_id, "npz")

    assert path == artifact_path.resolve()
    assert artifact.filename == "motion.npz"


def test_storage_rejects_unknown_missing_and_escaping_artifacts(tmp_path):
    storage = JobStorage(str(tmp_path))
    job_id = "job-1"
    storage.write_status(
        JobRecord(
            job_id=job_id,
            status=JobStatus.SUCCEEDED,
            job_dir=str(storage.job_dir(job_id)),
            artifacts={
                "missing": ArtifactRecord(
                    key="missing",
                    filename="missing.npz",
                    content_type="application/octet-stream",
                    size_bytes=0,
                ),
                "escape": ArtifactRecord(
                    key="escape",
                    filename="../secret.txt",
                    content_type="text/plain",
                    size_bytes=0,
                ),
            },
        )
    )

    with pytest.raises(KeyError, match="not found"):
        storage.resolve_artifact(job_id, "unknown")
    with pytest.raises(FileNotFoundError):
        storage.resolve_artifact(job_id, "missing")
    with pytest.raises(ValueError, match="escapes artifacts directory"):
        storage.resolve_artifact(job_id, "escape")


def test_job_manager_records_failed_runtime(tmp_path):
    storage = JobStorage(str(tmp_path))
    jobs = JobManager(runtime=FailingRuntime(), storage=storage)
    request = GenerationRequest(texts=["walk"], durations=[1.0])

    try:
        submitted = jobs.submit(request)
        _wait_for_job(jobs, submitted.job_id)
        jobs._futures[submitted.job_id].result(timeout=5)
        record = jobs.get(submitted.job_id)

        assert record.status == JobStatus.FAILED
        assert "RuntimeError: generation failed" in record.message
        assert "generation failed" in record.error
        assert storage.read_status(submitted.job_id).status == JobStatus.FAILED
    finally:
        jobs.executor.shutdown(wait=True)


def test_job_manager_get_falls_back_to_storage(tmp_path):
    storage = JobStorage(str(tmp_path))
    job_id = "stored-job"
    storage.write_status(JobRecord(job_id=job_id, status=JobStatus.SUCCEEDED, job_dir=str(storage.job_dir(job_id))))
    jobs = JobManager(runtime=FakeRuntime(), storage=storage)

    try:
        record = jobs.get(job_id)

        assert record.status == JobStatus.SUCCEEDED
        assert record.job_id == job_id
    finally:
        jobs.executor.shutdown(wait=True)


def test_job_manager_can_cancel_queued_job(tmp_path):
    release_first_job = Event()
    storage = JobStorage(str(tmp_path))
    jobs = JobManager(runtime=BlockingRuntime(release_first_job), storage=storage, max_workers=1)
    request = GenerationRequest(texts=["walk"], durations=[1.0])

    try:
        first = jobs.submit(request)
        _wait_for_status(jobs, first.job_id, JobStatus.RUNNING)

        second = jobs.submit(GenerationRequest(texts=["run"], durations=[1.0]))
        canceled = jobs.cancel(second.job_id)

        assert canceled.status == JobStatus.CANCELED
        assert storage.read_status(second.job_id).status == JobStatus.CANCELED
    finally:
        release_first_job.set()
        jobs.executor.shutdown(wait=True)


def test_create_app_rejects_runtime_and_text_encoder_config(tmp_path):
    with pytest.raises(ValueError, match="text_encoder_config"):
        create_app(
            storage_root=str(tmp_path),
            runtime=FakeRuntime(),
            text_encoder_config=TextEncoderServerConfig(mode="local"),
        )


class FailingRuntime:
    def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
        raise RuntimeError("generation failed")


class BlockingRuntime:
    def __init__(self, release_event: Event) -> None:
        self.release_event = release_event

    def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
        self.release_event.wait(timeout=5)
        return GenerationResult(job_id=request.job_id)


def _wait_for_job(jobs: JobManager, job_id: str) -> JobRecord:
    deadline = time.time() + 5
    while time.time() < deadline:
        record = jobs.get(job_id)
        if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED}:
            return record
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for job {job_id}")


def _wait_for_status(jobs: JobManager, job_id: str, status: JobStatus) -> JobRecord:
    deadline = time.time() + 5
    while time.time() < deadline:
        record = jobs.get(job_id)
        if record.status == status:
            return record
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for job {job_id} to reach {status.value}")
