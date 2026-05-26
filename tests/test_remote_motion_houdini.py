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


def test_hda_callbacks_model_menu_and_on_created(monkeypatch):
    from kimodo.model.registry import DEFAULT_MODEL, FRIENDLY_NAMES

    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    callbacks = importlib.import_module("remote_motion_houdini.hda_callbacks")

    menu_items = callbacks.model_menu({})

    assert menu_items[:2] == [DEFAULT_MODEL, FRIENDLY_NAMES[DEFAULT_MODEL]]
    assert len(menu_items) % 2 == 0

    node = FakeNode({"model": ""})
    callbacks.on_created({"node": node})

    assert node.parm("model").value == DEFAULT_MODEL


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

    assert calls["server_url"] == "http://127.0.0.1:8000"
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


def test_generate_motion_uses_configured_server_when_node_parm_is_empty(monkeypatch, tmp_path):
    from remote_motion_client.models import ArtifactInfo, JobInfo

    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    callbacks = importlib.import_module("remote_motion_houdini.hda_callbacks")

    calls: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, server_url: str) -> None:
            calls["server_url"] = server_url

        def submit(self, payload: dict[str, Any]) -> JobInfo:
            return JobInfo(job_id="job-1", status="queued")

        def wait(self, job_id: str, poll_interval: float, timeout: float | None) -> JobInfo:
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
            return tmp_path / "motion.bvh"

    monkeypatch.setattr(callbacks, "RemoteMotionClient", FakeClient)
    monkeypatch.setattr(callbacks, "default_download_dir", lambda job_id: tmp_path / job_id)
    monkeypatch.setattr(callbacks, "refresh_mocap_import", lambda node, artifact_path: True)
    monkeypatch.setitem(sys.modules, "hou", FakeHou(FakeHouNode("/obj"), tmp_path))
    monkeypatch.setattr(callbacks, "get_server_url", lambda: "http://192.168.1.10:9000")

    node = FakeNode(
        {
            "server_url": "",
            "prompt": "walk forward",
            "duration": 2.5,
            "seed": "",
            "model": "",
            "poll_interval": 0.25,
            "wait_timeout": "",
            "artifact_path": "",
        }
    )

    callbacks.generate_motion({"node": node})

    assert calls["server_url"] == "http://192.168.1.10:9000"


def test_generate_motion_reads_multiparm_prompts_and_generation_settings(monkeypatch, tmp_path):
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
            return Path(dst_dir_or_path) / "motion.bvh"

    monkeypatch.setattr(callbacks, "RemoteMotionClient", FakeClient)
    monkeypatch.setattr(callbacks, "default_download_dir", lambda job_id: tmp_path / job_id)
    monkeypatch.setattr(callbacks, "refresh_mocap_import", lambda node, artifact_path: True)
    monkeypatch.setitem(sys.modules, "hou", FakeHou(FakeHouNode("/obj"), tmp_path))

    node = FakeNode(
        {
            "server_url": "http://127.0.0.1:8000",
            "prompt1": "walk forward",
            "duration1": 2.5,
            "prompt2": "turn left",
            "duration2": 1.5,
            "seed": 123,
            "model": "kimodo-soma-rp",
            "diffusion_steps": 25,
            "text_weight": 2.0,
            "constraint_weight": 3.0,
            "num_transition_frames": 7,
            "poll_interval": 0.25,
            "wait_timeout": "",
            "bvhfile": "",
        }
    )

    callbacks.generate_motion({"node": node})

    assert calls["payload"] == {
        "texts": ["walk forward", "turn left"],
        "durations": [2.5, 1.5],
        "formats": ["bvh"],
        "seed": 123,
        "model": "kimodo-soma-rp",
        "diffusion_steps": 25,
        "cfg_type": "separated",
        "cfg_weight": [2.0, 3.0],
        "num_transition_frames": 7,
    }
    assert node.parm("bvhfile").value == "$HIP/job-1/motion.bvh"


def test_generate_motion_reads_model_menu_index_as_token(monkeypatch, tmp_path):
    from remote_motion_client.models import ArtifactInfo, JobInfo

    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    callbacks = importlib.import_module("remote_motion_houdini.hda_callbacks")

    calls: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, server_url: str) -> None:
            pass

        def submit(self, payload: dict[str, Any]) -> JobInfo:
            calls["payload"] = payload
            return JobInfo(job_id="job-1", status="queued")

        def wait(self, job_id: str, poll_interval: float, timeout: float | None) -> JobInfo:
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
            return Path(dst_dir_or_path) / "motion.bvh"

    monkeypatch.setattr(callbacks, "RemoteMotionClient", FakeClient)
    monkeypatch.setattr(callbacks, "default_download_dir", lambda job_id: tmp_path / job_id)
    monkeypatch.setattr(callbacks, "refresh_mocap_import", lambda node, artifact_path: True)

    node = FakeNode(
        {
            "server_url": "http://127.0.0.1:8000",
            "prompt": "walk forward",
            "duration": 2.5,
            "model": 0,
            "poll_interval": 0.25,
            "wait_timeout": "",
        }
    )
    node.parm("model").menu_items = ("kimodo-soma-rp", "kimodo-g1-rp")

    callbacks.generate_motion({"node": node})

    assert calls["payload"]["model"] == "kimodo-soma-rp"


def test_shelf_generate_and_download_waits_and_downloads(monkeypatch, tmp_path):
    from remote_motion_client.models import ArtifactInfo, JobInfo

    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    shelf_tools = importlib.import_module("remote_motion_houdini.shelf_tools")

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
            return tmp_path / "motion.bvh"

    monkeypatch.setattr(shelf_tools, "RemoteMotionClient", FakeClient)
    monkeypatch.setattr(shelf_tools, "get_server_url", lambda: "http://127.0.0.1:8000")
    monkeypatch.setattr(shelf_tools, "default_download_dir", lambda job_id: tmp_path / "motion_gen" / job_id)
    artifact_path = shelf_tools.generate_and_download(
        "walk forward",
        duration=3.0,
        poll_interval=0.25,
        wait_timeout=10.0,
    )

    assert calls["payload"] == {"texts": ["walk forward"], "durations": [3.0], "formats": ["bvh"]}
    assert calls["wait"] == {"job_id": "job-1", "poll_interval": 0.25, "timeout": 10.0}
    assert calls["download"]["dst"] == tmp_path / "motion_gen" / "job-1"
    assert artifact_path == tmp_path / "motion.bvh"


def test_shelf_generate_and_download_includes_optional_settings(monkeypatch, tmp_path):
    from remote_motion_client.models import ArtifactInfo, JobInfo

    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    shelf_tools = importlib.import_module("remote_motion_houdini.shelf_tools")

    calls: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, server_url: str) -> None:
            calls["server_url"] = server_url

        def submit(self, payload: dict[str, Any]) -> JobInfo:
            calls["payload"] = payload
            return JobInfo(job_id="job-1", status="queued")

        def wait(self, job_id: str, poll_interval: float, timeout: float | None) -> JobInfo:
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
            return tmp_path / "motion.bvh"

    monkeypatch.setattr(shelf_tools, "RemoteMotionClient", FakeClient)
    monkeypatch.setattr(shelf_tools, "get_server_url", lambda: "http://127.0.0.1:8000")
    monkeypatch.setattr(shelf_tools, "default_download_dir", lambda job_id: tmp_path / "motion_gen" / job_id)

    shelf_tools.generate_and_download("walk forward", duration=2.5, seed=123, model="kimodo-g1-rp")

    assert calls["payload"] == {
        "texts": ["walk forward"],
        "durations": [2.5],
        "formats": ["bvh"],
        "seed": 123,
        "model": "kimodo-g1-rp",
    }


def test_shelf_prompt_submit_generation_uses_dialog_settings(monkeypatch, tmp_path):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    shelf_tools = importlib.import_module("remote_motion_houdini.shelf_tools")

    calls: dict[str, Any] = {}
    obj = FakeHouNode("/obj")
    fake_hou = FakeHou(obj, tmp_path, ui=FakeUi())
    monkeypatch.setitem(sys.modules, "hou", fake_hou)
    monkeypatch.setattr(
        shelf_tools,
        "_prompt_generation_settings",
        lambda hou_module: shelf_tools.GenerationSettings(
            prompt="walk forward",
            duration=2.5,
            seed=123,
            model="kimodo-g1-rp",
        ),
    )

    def fake_generate_and_download(prompt: str, *, duration: float, seed: int | None, model: str | None) -> Path:
        calls["generate"] = {"prompt": prompt, "duration": duration, "seed": seed, "model": model}
        return tmp_path / "motion_gen" / "job-1" / "motion.bvh"

    monkeypatch.setattr(shelf_tools, "generate_and_download", fake_generate_and_download)

    shelf_tools.prompt_submit_generation()

    assert calls["generate"] == {
        "prompt": "walk forward",
        "duration": 2.5,
        "seed": 123,
        "model": "kimodo-g1-rp",
    }
    assert obj.children[0].path() == "/obj/kimodo_motion_walk_forward"


def test_shelf_create_mocap_import_node_sets_downloaded_path(monkeypatch, tmp_path):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    shelf_tools = importlib.import_module("remote_motion_houdini.shelf_tools")

    obj = FakeHouNode("/obj")
    fake_hou = FakeHou(obj, tmp_path)
    monkeypatch.setitem(sys.modules, "hou", fake_hou)

    artifact_path = tmp_path / "motion_gen" / "job-1" / "motion.bvh"
    mocap = shelf_tools.create_mocap_import_node(artifact_path, "walk forward")

    assert mocap.type_name == "mocapimport"
    assert mocap.parm("filetype").value == "biovision"
    assert mocap.parm("bvhfile").value == "$HIP/motion_gen/job-1/motion.bvh"
    assert mocap.parm("scale").value == 0.01
    assert mocap.parm("reload").pressed is True
    assert mocap.display_flag is True
    assert mocap.render_flag is True
    assert obj.children[0].type_name == "geo"
    assert obj.children[0].path() == "/obj/kimodo_motion_walk_forward"
    assert obj.children[0].node("file1") is None


def test_shelf_create_mocap_import_node_reports_missing_required_parm(monkeypatch, tmp_path):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    shelf_tools = importlib.import_module("remote_motion_houdini.shelf_tools")

    obj = FakeHouNode("/obj")
    fake_hou = FakeHou(obj, tmp_path)
    monkeypatch.setitem(sys.modules, "hou", fake_hou)
    monkeypatch.setattr(FakeHouNode, "mocap_parm_defs", {"filetype": "File Type"}, raising=False)

    artifact_path = tmp_path / "motion.bvh"
    try:
        shelf_tools.create_mocap_import_node(artifact_path)
    except RuntimeError as exc:
        assert "bvhfile" in str(exc)
        assert "filetype" in str(exc)
    else:
        raise AssertionError("Expected missing bvhfile parm to raise RuntimeError")


def test_importer_refresh_mocap_import_sets_bvhfile(monkeypatch, tmp_path):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    importer = importlib.import_module("remote_motion_houdini.importer")

    child = FakeHouNode("/obj/asset/mocap_import_bvh", "mocapimport")
    node = FakeHouNode("/obj/asset", "geo")
    node.children.append(child)
    monkeypatch.setitem(sys.modules, "hou", FakeHou(FakeHouNode("/obj"), tmp_path))

    artifact_path = tmp_path / "motion_gen" / "job-1" / "motion.bvh"
    assert importer.refresh_mocap_import(node, artifact_path) is True
    assert child.parm("bvhfile").value == "$HIP/motion_gen/job-1/motion.bvh"
    assert child.parm("reload").pressed is True


def test_houdini_config_persists_server_url(monkeypatch, tmp_path):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    config = importlib.import_module("remote_motion_houdini.config")

    path = config.set_server_url("http://localhost:9000/", pref_dir=tmp_path)

    assert path == tmp_path / "kimodo" / "remote_motion.json"
    assert config.get_server_url(pref_dir=tmp_path) == "http://localhost:9000"


def test_houdini_config_uses_server_host_and_port_env(monkeypatch, tmp_path):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    config = importlib.import_module("remote_motion_houdini.config")

    monkeypatch.delenv("KIMODO_REMOTE_MOTION_SERVER_URL", raising=False)
    monkeypatch.setenv("KIMODO_SERVER_HOST", "192.168.1.10")
    monkeypatch.setenv("KIMODO_SERVER_PORT", "9000")

    assert config.get_server_url(pref_dir=tmp_path) == "http://192.168.1.10:9000"


def test_houdini_config_rejects_invalid_server_url(monkeypatch):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    config = importlib.import_module("remote_motion_houdini.config")

    try:
        config.normalize_server_url("localhost:9000")
    except ValueError as exc:
        assert "Server URL" in str(exc)
    else:
        raise AssertionError("Expected invalid server URL to raise ValueError")


class FakeParmTemplate:
    def __init__(self, label: str, menu_items: tuple[str, ...] = ()) -> None:
        self._label = label
        self._menu_items = menu_items

    def label(self) -> str:
        return self._label

    def menuItems(self) -> tuple[str, ...]:
        return self._menu_items


class FakeParm:
    def __init__(self, value: Any, name: str = "", label: str = "") -> None:
        self.value = value
        self.string_value = value
        self.menu_items: tuple[str, ...] = ()
        self.pressed = False
        self._name = name
        self._label = label

    def eval(self) -> Any:
        return self.value

    def evalAsString(self) -> str:
        return str(self.string_value)

    def name(self) -> str:
        return self._name

    def parmTemplate(self) -> FakeParmTemplate:
        return FakeParmTemplate(self._label, self.menu_items)

    def set(self, value: Any) -> None:
        self.value = value

    def pressButton(self) -> None:
        self.pressed = True


class FakeNode:
    def __init__(self, parms: dict[str, Any]) -> None:
        self._parms = {key: FakeParm(value, key) for key, value in parms.items()}

    def parm(self, name: str) -> FakeParm | None:
        return self._parms.get(name)

    def parms(self) -> list[FakeParm]:
        return list(self._parms.values())


class FakeSeverityType:
    Error = "error"


class FakeUi:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def displayMessage(self, message: str, **kwargs) -> None:
        self.messages.append(message)


class FakeHou:
    def __init__(self, obj_node: "FakeHouNode", hip_dir: str | Path | None = None, ui=None) -> None:
        self.obj_node = obj_node
        self.hip_dir = hip_dir
        self.ui = ui
        self.severityType = FakeSeverityType

    def node(self, path: str) -> "FakeHouNode | None":
        if path == "/obj":
            return self.obj_node
        return None

    def expandString(self, text: str) -> str:
        if text == "$HIP" and self.hip_dir is not None:
            return str(self.hip_dir)
        return text


class FakeHouNode:
    mocap_parm_defs: dict[str, str] | None = None

    def __init__(self, path: str, type_name: str = "obj") -> None:
        self._path = path
        self.type_name = type_name
        self.children: list[FakeHouNode] = []
        self.destroyed = False
        self.display_flag = False
        self.render_flag = False
        self.current = False
        parm_defs = self.mocap_parm_defs if type_name.startswith("mocap") and self.mocap_parm_defs else None
        default_parms = {"filetype": "File Type", "bvhfile": "BVH File", "scale": "Scale", "reload": "Reload"}
        self._parms = {name: FakeParm("", name, label) for name, label in (parm_defs or default_parms).items()}

    def createNode(self, type_name: str, node_name: str | None = None) -> "FakeHouNode":
        node_name = node_name or type_name
        child = FakeHouNode(f"{self._path}/{node_name}", type_name)
        if type_name == "geo":
            child.children.append(FakeHouNode(f"{child._path}/file1", "file"))
        self.children.append(child)
        return child

    def node(self, name: str) -> "FakeHouNode | None":
        for child in self.children:
            if child._path.rsplit("/", 1)[-1] == name and not child.destroyed:
                return child
        return None

    def parm(self, name: str) -> FakeParm | None:
        return self._parms.get(name)

    def parms(self) -> list[FakeParm]:
        return list(self._parms.values())

    def destroy(self) -> None:
        self.destroyed = True

    def path(self) -> str:
        return self._path

    def layoutChildren(self) -> None:
        pass

    def setCurrent(self, value: bool, clear_all_selected: bool = False) -> None:
        self.current = value

    def setName(self, name: str, unique_name: bool = False) -> None:
        self._path = f"{self._path.rsplit('/', 1)[0]}/{name}"

    def setDisplayFlag(self, value: bool) -> None:
        self.display_flag = value

    def setRenderFlag(self, value: bool) -> None:
        self.render_flag = value
