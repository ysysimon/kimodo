# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the standalone remote motion client."""

from __future__ import annotations

import socket
import sys
import threading
import time
from typing import Any

import pytest


def test_remote_motion_client_import_does_not_import_kimodo_or_hou():
    removed_kimodo_modules = {}
    removed_hou = sys.modules.pop("hou", None)
    for name in list(sys.modules):
        if name == "remote_motion_client" or name.startswith("remote_motion_client."):
            sys.modules.pop(name)
        elif name == "kimodo" or name.startswith("kimodo."):
            removed_kimodo_modules[name] = sys.modules.pop(name)

    try:
        import remote_motion_client  # noqa: F401

        assert "kimodo" not in sys.modules
        assert "hou" not in sys.modules
    finally:
        for name in list(sys.modules):
            if name == "kimodo" or name.startswith("kimodo."):
                sys.modules.pop(name)
        sys.modules.update(removed_kimodo_modules)
        if removed_hou is not None:
            sys.modules["hou"] = removed_hou


def test_submit_uses_current_jobs_endpoint():
    from remote_motion_client import RemoteMotionClient

    session = FakeSession(
        FakeResponse(
            {
                "job_id": "job-1",
                "status": "queued",
                "progress": 0.0,
                "message": "",
                "artifacts": {},
                "error": None,
            },
            status_code=202,
        )
    )

    client = RemoteMotionClient("http://server.test", session=session)
    job = client.submit({"texts": ["walk"], "durations": [1.0]})

    assert job.job_id == "job-1"
    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["url"] == "http://server.test/jobs"


def test_job_and_artifact_models_parse_response_payload():
    from remote_motion_client.models import JobInfo

    job = JobInfo.from_dict(
        {
            "job_id": "job-1",
            "status": "succeeded",
            "progress": 1.0,
            "message": "done",
            "artifacts": {
                "bvh": {
                    "key": "bvh",
                    "filename": "motion.bvh",
                    "content_type": "application/octet-stream",
                    "size_bytes": 12,
                    "download_url": "/jobs/job-1/artifacts/bvh",
                }
            },
            "error": None,
        }
    )

    assert job.is_terminal is True
    assert job.artifacts["bvh"].filename == "motion.bvh"
    assert job.artifacts["bvh"].download_url == "/jobs/job-1/artifacts/bvh"


def test_wait_success_failed_canceled_and_timeout(monkeypatch):
    from remote_motion_client import RemoteMotionClient
    from remote_motion_client.errors import RemoteMotionJobFailedError, RemoteMotionTimeoutError
    from remote_motion_client.models import JobInfo

    class PollingClient(RemoteMotionClient):
        def __init__(self, jobs):
            self.jobs = iter(jobs)

        def get_job(self, job_id: str) -> JobInfo:
            return next(self.jobs)

    running = JobInfo(job_id="job-1", status="running")
    succeeded = JobInfo(job_id="job-1", status="succeeded")
    failed = JobInfo(job_id="job-2", status="failed", message="boom")
    canceled = JobInfo(job_id="job-3", status="canceled")

    assert PollingClient([running, succeeded]).wait("job-1", poll_interval=0) == succeeded
    with pytest.raises(RemoteMotionJobFailedError):
        PollingClient([failed]).wait("job-2", poll_interval=0)
    with pytest.raises(RemoteMotionJobFailedError):
        PollingClient([canceled]).wait("job-3", poll_interval=0)

    monkeypatch.setattr("remote_motion_client.client.time.sleep", lambda _seconds: None)
    with pytest.raises(RemoteMotionTimeoutError):
        PollingClient([running, running, running]).wait("job-4", poll_interval=0, timeout=0)


def test_download_artifact_uses_metadata_url_and_filename(tmp_path):
    from remote_motion_client import RemoteMotionClient
    from remote_motion_client.models import ArtifactInfo, JobInfo

    session = FakeSession(
        FakeResponse(
            b"bvh-data",
            headers={"content-disposition": 'attachment; filename="motion.bvh"'},
        )
    )
    client = RemoteMotionClient("http://server.test", session=session)
    job = JobInfo(
        job_id="job-1",
        status="succeeded",
        artifacts={
            "bvh": ArtifactInfo(
                key="bvh",
                filename="server-name.bvh",
                content_type="application/octet-stream",
                size_bytes=8,
                download_url="/jobs/job-1/artifacts/bvh",
            )
        },
    )

    path = client.download_artifact(job, "bvh", tmp_path)

    assert path == tmp_path / "motion.bvh"
    assert path.read_bytes() == b"bvh-data"
    assert session.calls[0]["url"] == "http://server.test/jobs/job-1/artifacts/bvh"


def test_download_artifact_can_build_url_from_job_id(tmp_path):
    from remote_motion_client import RemoteMotionClient

    session = FakeSession(FakeResponse(b"npz-data"))
    client = RemoteMotionClient("http://server.test/base", session=session)

    path = client.download_artifact("job-1", "npz", tmp_path / "result.npz")

    assert path.read_bytes() == b"npz-data"
    assert session.calls[0]["url"] == "http://server.test/base/jobs/job-1/artifacts/npz"


def test_http_error_is_wrapped():
    import requests

    from remote_motion_client import RemoteMotionClient
    from remote_motion_client.errors import RemoteMotionHTTPError

    response = FakeResponse({"detail": "missing"}, status_code=404, text="missing")

    def raise_404():
        raise requests.HTTPError(response=response)

    response.raise_for_status = raise_404
    client = RemoteMotionClient("http://server.test", session=FakeSession(response))

    with pytest.raises(RemoteMotionHTTPError) as error_info:
        client.get_job("missing")

    assert error_info.value.status_code == 404
    assert error_info.value.response_text == "missing"


def test_client_integrates_with_fake_fastapi_server(tmp_path):
    from kimodo.server.asgi import create_fastapi_app
    from kimodo.server.runtime import FakeRuntime
    from remote_motion_client import RemoteMotionClient

    app = create_fastapi_app(storage_root=str(tmp_path / "server"), runtime=FakeRuntime())
    with running_uvicorn(app) as base_url:
        client = RemoteMotionClient(base_url)
        assert client.health()["status"] == "ok"
        submitted = client.submit({"texts": ["A person walks."], "durations": [1.0], "formats": ["npz"]})
        job = client.wait(submitted.job_id, poll_interval=0.01, timeout=5.0)
        path = client.download_artifact(job, "npz", tmp_path / "downloads")

    assert path.is_file()
    assert path.read_bytes()


class FakeSession:
    def __init__(self, *responses: "FakeResponse") -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> "FakeResponse":
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        response = self.responses.pop(0)
        response.url = url
        return response


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any] | bytes,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        text: str | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text if text is not None else (payload.decode("utf-8") if isinstance(payload, bytes) else "")
        self.url = "http://server.test"

    def json(self) -> dict[str, Any]:
        if not isinstance(self.payload, dict):
            raise ValueError("not json")
        return self.payload

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        assert chunk_size > 0
        yield self.payload if isinstance(self.payload, bytes) else b""

    def close(self) -> None:
        return None


class running_uvicorn:
    def __init__(self, app) -> None:
        self.app = app
        self.port = _free_port()
        self.server = None
        self.thread = None

    def __enter__(self) -> str:
        import requests
        import uvicorn

        config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()

        base_url = f"http://127.0.0.1:{self.port}"
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                if requests.get(f"{base_url}/health", timeout=0.2).status_code == 200:
                    return base_url
            except requests.RequestException:
                time.sleep(0.05)
        pytest.fail("Timed out waiting for uvicorn test server.")

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self.server is not None
        assert self.thread is not None
        self.server.should_exit = True
        self.thread.join(timeout=5)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
