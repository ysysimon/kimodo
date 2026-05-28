# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the FastAPI server adapter."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

from kimodo.scripts.run_server import main as run_server_main
from kimodo.server.asgi import create_fastapi_app
from kimodo.server.runtime import FakeRuntime
from kimodo.server.schemas import ArtifactRecord, GenerationRequest, GenerationResult, JobRecord, JobStatus

pytestmark = pytest.mark.server


def test_health_reports_runtime(tmp_path):
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=FakeRuntime())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "runtime": "FakeRuntime"}


def test_submit_and_get_job_without_job_dir(tmp_path):
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=FakeRuntime())

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "texts": ["A person walks forward."],
                "durations": [1.0],
                "formats": ["npz"],
            },
        )
        assert response.status_code == 202
        submitted = response.json()
        assert "job_dir" not in submitted

        record = _wait_for_status(client, submitted["job_id"], JobStatus.SUCCEEDED)

    assert "job_dir" not in record
    assert record["status"] == "succeeded"
    assert set(record["artifacts"]) == {"npz"}
    assert record["artifacts"]["npz"]["download_url"] == f"/jobs/{submitted['job_id']}/artifacts/npz"


def test_submit_rejects_client_job_id(tmp_path):
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=FakeRuntime())

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "job_id": "client-chosen",
                "texts": ["A person walks forward."],
                "durations": [1.0],
            },
        )

    assert response.status_code == 422


def test_submit_accepts_bvh_standard_tpose(tmp_path):
    runtime = CapturingRuntime()
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "texts": ["A person walks forward."],
                "durations": [1.0],
                "formats": ["bvh"],
                "bvh_standard_tpose": False,
            },
        )
        assert response.status_code == 202
        submitted = response.json()
        _wait_for_status(client, submitted["job_id"], JobStatus.SUCCEEDED)

    assert len(runtime.requests) == 1
    assert runtime.requests[0].bvh_standard_tpose is False


def test_submit_accepts_output_world_offset(tmp_path):
    runtime = CapturingRuntime()
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "texts": ["A person walks forward."],
                "durations": [1.0],
                "formats": ["bvh"],
                "output_world_offset": [1.0, 0.0, -2.0],
            },
        )
        assert response.status_code == 202
        submitted = response.json()
        _wait_for_status(client, submitted["job_id"], JobStatus.SUCCEEDED)

    assert len(runtime.requests) == 1
    assert runtime.requests[0].output_world_offset == [1.0, 0.0, -2.0]


@pytest.mark.parametrize("output_world_offset", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], ["x", 0.0, 0.0]])
def test_submit_rejects_invalid_output_world_offset(tmp_path, output_world_offset):
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=FakeRuntime())

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "texts": ["A person walks forward."],
                "durations": [1.0],
                "output_world_offset": output_world_offset,
            },
        )

    assert response.status_code == 422


def test_submit_accepts_constraints_list_as_plain_dicts(tmp_path):
    runtime = CapturingRuntime()
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=runtime)
    constraints = [
        _root2d_constraint(),
        _pose_constraint("right-hand"),
        _pose_constraint("end-effector", joint_names=["LeftHand", "RightFoot"]),
    ]

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "texts": ["A person walks forward."],
                "durations": [1.0],
                "formats": ["npz"],
                "constraints": constraints,
            },
        )
        assert response.status_code == 202
        submitted = response.json()
        _wait_for_status(client, submitted["job_id"], JobStatus.SUCCEEDED)

    assert len(runtime.requests) == 1
    assert runtime.requests[0].constraints == constraints


def test_submit_accepts_constraints_path_string(tmp_path):
    runtime = CapturingRuntime()
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=runtime)
    constraints_path = str(tmp_path / "constraints.json")

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "texts": ["A person walks forward."],
                "durations": [1.0],
                "formats": ["npz"],
                "constraints": constraints_path,
            },
        )
        assert response.status_code == 202
        submitted = response.json()
        _wait_for_status(client, submitted["job_id"], JobStatus.SUCCEEDED)

    assert len(runtime.requests) == 1
    assert runtime.requests[0].constraints == constraints_path


@pytest.mark.parametrize(
    "constraints",
    [
        [{"type": "unknown", "frame_indices": [0]}],
        [{"type": "root2d", "frame_indices": [0]}],
        [{"type": "root2d", "frame_indices": [0], "smooth_root_2d": [[0.0, 0.0, 0.0]]}],
        [{"type": "root2d", "frame_indices": [0], "smooth_root_2d": [[0.0, 0.0]], "extra": True}],
        [
            {
                "type": "root2d",
                "frame_indices": [0, 1],
                "smooth_root_2d": [[0.0, 0.0]],
            }
        ],
        [
            {
                "type": "fullbody",
                "frame_indices": [0],
                "root_positions": [[0.0, 0.95, 0.0]],
                "local_joints_rot": [[[0.0, 0.0]]],
            }
        ],
        [
            {
                "type": "end-effector",
                "joint_names": ["left_hand"],
                "frame_indices": [0],
                "root_positions": [[0.0, 0.95, 0.0]],
                "local_joints_rot": [[[0.0, 0.0, 0.0]]],
            }
        ],
    ],
)
def test_submit_rejects_invalid_constraints(tmp_path, constraints):
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=FakeRuntime())

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "texts": ["A person walks forward."],
                "durations": [1.0],
                "constraints": constraints,
            },
        )

    assert response.status_code == 422


def test_download_artifact_returns_file(tmp_path):
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=FakeRuntime())

    with TestClient(app) as client:
        submitted = client.post(
            "/jobs",
            json={
                "texts": ["A person walks forward."],
                "durations": [1.0],
                "formats": ["npz"],
            },
        ).json()
        _wait_for_status(client, submitted["job_id"], JobStatus.SUCCEEDED)

        response = client.get(f"/jobs/{submitted['job_id']}/artifacts/npz")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert "motion.npz" in response.headers["content-disposition"]
    assert b"Kimodo fake server artifact" in response.content


def test_unknown_job_and_artifact_return_404(tmp_path):
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=FakeRuntime())

    with TestClient(app) as client:
        missing_job = client.get("/jobs/missing-job")
        submitted = client.post(
            "/jobs",
            json={
                "texts": ["A person walks forward."],
                "durations": [1.0],
                "formats": ["npz"],
            },
        ).json()
        _wait_for_status(client, submitted["job_id"], JobStatus.SUCCEEDED)
        missing_artifact = client.get(f"/jobs/{submitted['job_id']}/artifacts/missing")

    assert missing_job.status_code == 404
    assert missing_artifact.status_code == 404


def test_artifact_path_escape_returns_400(tmp_path):
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=FakeRuntime())
    storage = app.state.storage
    storage.write_status(
        JobRecord(
            job_id="job-escape",
            status=JobStatus.SUCCEEDED,
            job_dir=str(storage.job_dir("job-escape")),
            artifacts={
                "escape": ArtifactRecord(
                    key="escape",
                    filename="../secret.txt",
                    content_type="text/plain",
                    size_bytes=0,
                )
            },
        )
    )

    with TestClient(app) as client:
        response = client.get("/jobs/job-escape/artifacts/escape")

    assert response.status_code == 400


def test_cancel_queued_job(tmp_path):
    release_first_job = Event()
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=BlockingRuntime(release_first_job))

    with TestClient(app) as client:
        try:
            first = client.post("/jobs", json={"texts": ["walk"], "durations": [1.0]}).json()
            _wait_for_status(client, first["job_id"], JobStatus.RUNNING)

            second = client.post("/jobs", json={"texts": ["run"], "durations": [1.0]}).json()
            canceled = client.post(f"/jobs/{second['job_id']}/cancel")

            assert canceled.status_code == 200
            assert canceled.json()["status"] == "canceled"
        finally:
            release_first_job.set()


def test_lifespan_closes_runtime(tmp_path):
    runtime = ClosingFakeRuntime()
    app = create_fastapi_app(storage_root=str(tmp_path), runtime=runtime)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert runtime.closed is True


def test_run_server_cli_builds_fake_runtime_app(monkeypatch, tmp_path):
    calls = []

    def fake_uvicorn_run(app, *, host: str, port: int) -> None:
        calls.append({"app": app, "host": host, "port": port})

    monkeypatch.setattr("kimodo.scripts.run_server.uvicorn.run", fake_uvicorn_run)

    run_server_main(
        [
            "--runtime",
            "fake",
            "--storage-root",
            str(tmp_path),
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
        ]
    )

    assert len(calls) == 1
    assert calls[0]["host"] == "0.0.0.0"
    assert calls[0]["port"] == 9001
    assert isinstance(calls[0]["app"].state.runtime, FakeRuntime)


def test_run_server_cli_uses_host_and_port_from_env(monkeypatch, tmp_path):
    calls = []

    def fake_uvicorn_run(app, *, host: str, port: int) -> None:
        calls.append({"app": app, "host": host, "port": port})

    monkeypatch.setattr("kimodo.scripts.run_server.uvicorn.run", fake_uvicorn_run)
    monkeypatch.setenv("KIMODO_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("KIMODO_SERVER_PORT", "9002")

    run_server_main(["--runtime", "fake", "--storage-root", str(tmp_path)])

    assert len(calls) == 1
    assert calls[0]["host"] == "0.0.0.0"
    assert calls[0]["port"] == 9002
    assert isinstance(calls[0]["app"].state.runtime, FakeRuntime)


class BlockingRuntime:
    def __init__(self, release_event: Event) -> None:
        self.release_event = release_event

    def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
        self.release_event.wait(timeout=5)
        return GenerationResult(job_id=request.job_id)


class CapturingRuntime:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(job_id=request.job_id)


class ClosingFakeRuntime(FakeRuntime):
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _wait_for_status(client: TestClient, job_id: str, status: JobStatus) -> dict:
    deadline = time.time() + 5
    while time.time() < deadline:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        record = response.json()
        if record["status"] == status.value:
            return record
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for job {job_id} to reach {status.value}")


def _root2d_constraint() -> dict:
    return {
        "type": "root2d",
        "frame_indices": [0, 30],
        "smooth_root_2d": [[0.0, 0.0], [1.0, 0.2]],
        "global_root_heading": [[1.0, 0.0], [0.0, 1.0]],
    }


def _pose_constraint(
    constraint_type: str,
    *,
    joint_names: list[str] | None = None,
    local_joints_rot: list | None = None,
) -> dict:
    constraint = {
        "type": constraint_type,
        "frame_indices": [0],
        "root_positions": [[0.0, 0.95, 0.0]],
        "smooth_root_2d": [[0.0, 0.0]],
        "local_joints_rot": local_joints_rot
        if local_joints_rot is not None
        else [
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
            ]
        ],
    }
    if joint_names is not None:
        constraint["joint_names"] = joint_names
    return constraint
