# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Houdini remote motion plugin glue."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def test_remote_motion_houdini_imports_without_hou(monkeypatch):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    sys.modules.pop("hou", None)

    module = importlib.import_module("remote_motion_houdini.hda_callbacks")

    assert module is not None
    assert "hou" not in sys.modules


def test_generate_motion_reads_parms_downloads_and_refreshes(monkeypatch, tmp_path):
    from remote_motion_client.models import ArtifactInfo, JobInfo

    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    callbacks = importlib.import_module("remote_motion_houdini.hda_callbacks")

    calls: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, server_url: str) -> None:
            calls["server_url"] = server_url

        def submit(self, payload: dict[str, Any]) -> JobInfo:
            calls["payload"] = payload
            return JobInfo(job_id="job-1", status="queued")

        def wait(self, job_id: str, poll_interval: float, timeout: float | None) -> JobInfo:
            calls["wait"] = {"job_id": job_id, "poll_interval": poll_interval, "timeout": timeout}
            return JobInfo(
                job_id="job-1",
                status="succeeded",
                artifacts={
                    "bvh": ArtifactInfo(
                        key="bvh",
                        filename="motion.bvh",
                        content_type="application/octet-stream",
                        size_bytes=8,
                        download_url="/jobs/job-1/artifacts/bvh",
                    )
                },
            )

        def download_artifact(self, job: JobInfo, artifact_key: str, dst_dir_or_path: Path) -> Path:
            calls["download"] = {"job": job.job_id, "artifact_key": artifact_key, "dst": dst_dir_or_path}
            path = Path(dst_dir_or_path) / "motion.bvh"
            path.parent.mkdir(parents=True)
            path.write_text("BVH", encoding="utf-8")
            return path

    def fake_refresh(node, artifact_path):
        calls["refresh"] = {"node": node, "artifact_path": artifact_path}
        return True

    monkeypatch.setattr(callbacks, "RemoteMotionClient", FakeClient)
    monkeypatch.setattr(callbacks, "default_download_dir", lambda job_id: tmp_path / "motion_gen" / job_id)
    monkeypatch.setattr(callbacks, "refresh_mocap_import", fake_refresh)

    node = FakeNode(
        {
            "server_url": "http://127.0.0.1:8000",
            "prompt": "walk forward",
            "duration": 2.5,
            "seed": 123,
            "model": "kimodo-soma-rp",
            "poll_interval": 0.25,
            "wait_timeout": 10.0,
            "artifact_path": "",
        }
    )

    artifact_path = callbacks.generate_motion({"node": node})

    assert calls["payload"] == {
        "texts": ["walk forward"],
        "durations": [2.5],
        "formats": ["bvh"],
        "seed": 123,
        "model": "kimodo-soma-rp",
    }
    assert calls["download"]["dst"] == tmp_path / "motion_gen" / "job-1"
    assert node.parm("artifact_path").value == str(artifact_path)
    assert calls["refresh"]["artifact_path"] == artifact_path


class FakeParm:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.pressed = False

    def eval(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value

    def pressButton(self) -> None:
        self.pressed = True


class FakeNode:
    def __init__(self, parms: dict[str, Any]) -> None:
        self.parms = {key: FakeParm(value) for key, value in parms.items()}

    def parm(self, name: str) -> FakeParm | None:
        return self.parms.get(name)
