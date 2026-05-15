# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for server fake runtime wiring."""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

import pytest

from kimodo.server.app import create_app
from kimodo.server.jobs import JobManager
from kimodo.server.runtime import FakeRuntime
from kimodo.server.schemas import GenerationRequest, JobStatus
from kimodo.server.storage import JobStorage

pytestmark = pytest.mark.server


@pytest.fixture
def work_dir(request):
    path = Path("outputs") / "server-tests" / f"{request.node.name}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_fake_runtime_writes_single_npz_artifact(work_dir):
    request = GenerationRequest(texts=["A person walks."], durations=[1.0], formats=["npz"], job_id="job-1")

    result = FakeRuntime().generate(request, job_dir=work_dir)

    assert set(result.artifacts) == {"npz"}
    artifact = result.artifacts["npz"]
    assert artifact.filename == "motion.npz"
    assert artifact.download_url == "/jobs/job-1/artifacts/npz"
    assert (work_dir / "artifacts" / artifact.filename).is_file()


def test_fake_runtime_writes_requested_formats(work_dir):
    request = GenerationRequest(
        texts=["A person walks."],
        durations=[1.0],
        formats=["npz", "bvh"],
        job_id="job-2",
    )

    result = FakeRuntime().generate(request, job_dir=work_dir)

    assert set(result.artifacts) == {"npz", "bvh"}
    assert (work_dir / "artifacts" / "motion.npz").is_file()
    assert (work_dir / "artifacts" / "motion.bvh").is_file()


def test_fake_runtime_normalizes_format_names(work_dir):
    request = GenerationRequest(
        texts=["A person walks."],
        durations=[1.0],
        formats=["NPZ", "BvH"],
        job_id="job-mixed-case",
    )

    result = FakeRuntime().generate(request, job_dir=work_dir)

    assert set(result.artifacts) == {"npz", "bvh"}
    assert (work_dir / "artifacts" / "motion.npz").is_file()
    assert (work_dir / "artifacts" / "motion.bvh").is_file()


def test_fake_runtime_writes_multi_sample_artifacts(work_dir):
    request = GenerationRequest(
        texts=["A person walks."],
        durations=[1.0],
        formats=["npz", "bvh"],
        num_samples=2,
        job_id="job-multi-sample",
    )

    result = FakeRuntime().generate(request, job_dir=work_dir)

    assert set(result.artifacts) == {"npz_00", "npz_01", "bvh_00", "bvh_01"}
    assert (work_dir / "artifacts" / "motion_00.npz").is_file()
    assert (work_dir / "artifacts" / "motion_01.npz").is_file()
    assert (work_dir / "artifacts" / "motion_00.bvh").is_file()
    assert (work_dir / "artifacts" / "motion_01.bvh").is_file()


def test_fake_runtime_can_zip_artifacts(work_dir):
    request = GenerationRequest(
        texts=["A person walks."],
        durations=[1.0],
        formats=["npz", "bvh"],
        zip_output=True,
        job_id="job-3",
    )

    result = FakeRuntime().generate(request, job_dir=work_dir)

    assert set(result.artifacts) == {"zip"}
    artifact = result.artifacts["zip"]
    assert artifact.filename == "artifacts.zip"
    assert artifact.content_type == "application/zip"
    assert (work_dir / "artifacts" / artifact.filename).is_file()
    assert _zip_names(work_dir / "artifacts" / artifact.filename) == {"motion.npz", "motion.bvh"}


def test_fake_runtime_rejects_mismatched_segments(work_dir):
    request = GenerationRequest(texts=["A person walks."], durations=[1.0, 2.0], formats=["npz"])

    with pytest.raises(ValueError, match="texts and durations"):
        FakeRuntime().generate(request, job_dir=work_dir)


def test_job_manager_succeeds_with_fake_runtime(work_dir):
    storage = JobStorage(str(work_dir))
    jobs = JobManager(runtime=FakeRuntime(), storage=storage)
    request = GenerationRequest(texts=["A person walks."], durations=[1.0], formats=["npz"])

    submitted = jobs.submit(request)
    record = _wait_for_job(jobs, submitted.job_id)

    assert record.status == JobStatus.SUCCEEDED
    assert record.progress == 1.0
    assert set(record.artifacts) == {"npz"}

    stored = storage.read_status(submitted.job_id)
    assert stored.status == JobStatus.SUCCEEDED
    path, artifact = storage.resolve_artifact(submitted.job_id, "npz")
    assert path.is_file()
    assert artifact.download_url == f"/jobs/{submitted.job_id}/artifacts/npz"

    jobs.executor.shutdown(wait=True)


def test_create_app_accepts_runtime_injection(work_dir):
    app = create_app(storage_root=str(work_dir), runtime=FakeRuntime())

    assert isinstance(app["jobs"], JobManager)
    assert isinstance(app["runtime"], FakeRuntime)
    app["jobs"].executor.shutdown(wait=True)


def _wait_for_job(jobs: JobManager, job_id: str):
    deadline = time.time() + 5
    while time.time() < deadline:
        record = jobs.get(job_id)
        if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED}:
            return record
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for job {job_id}")


def _zip_names(path: Path) -> set[str]:
    from zipfile import ZipFile

    with ZipFile(path) as zip_file:
        return set(zip_file.namelist())
