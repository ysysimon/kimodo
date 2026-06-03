# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Houdini remote motion plugin glue."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any

import torch


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


def test_hda_callbacks_prints_warnings_without_houdini_ui(monkeypatch, capsys):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    callbacks = importlib.import_module("remote_motion_houdini.hda_callbacks")

    class FailingUi:
        def displayMessage(self, message: str, **kwargs) -> None:
            raise AssertionError("displayMessage should not be called in headless mode")

    class HeadlessHou:
        ui = FailingUi()
        severityType = FakeSeverityType

        def isUIAvailable(self) -> bool:
            return False

    monkeypatch.setitem(sys.modules, "hou", HeadlessHou())

    callbacks._display_houdini_warnings(["headless constraint warning"])

    assert "headless constraint warning" in capsys.readouterr().out


def test_hda_callbacks_prints_warnings_when_display_message_fails(monkeypatch, capsys):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    callbacks = importlib.import_module("remote_motion_houdini.hda_callbacks")

    class FailingUi:
        def displayMessage(self, message: str, **kwargs) -> None:
            raise RuntimeError("no display")

    class GuiHou:
        ui = FailingUi()
        severityType = FakeSeverityType

        def isUIAvailable(self) -> bool:
            return True

    monkeypatch.setitem(sys.modules, "hou", GuiHou())

    callbacks._display_houdini_warnings(["fallback constraint warning"])

    assert "fallback constraint warning" in capsys.readouterr().out


def test_root2d_constraint_parser_projects_positions_sorts_frames_and_reads_heading(monkeypatch):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    constraints = importlib.import_module("remote_motion_houdini.constraints")

    z_to_x = (0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    geo = FakeGeometry(
        [
            FakePoint((2.0, 0.0, 5.0), {"frame": 20, "transform": z_to_x}),
            FakePoint((0.0, 0.0, 0.0), {"frame": 1, "transform": _identity3()}),
            FakePoint((-1.0, 0.25, 3.0), {"frame": 10, "transform": _identity3()}),
        ],
        point_attribs={"frame", "transform"},
    )

    constraint, warnings = constraints.root2d_constraint_from_geometry(
        geo,
        frame_origin=1,
        include_heading=True,
        heading_forward_axis="+Z",
        nonplanar_y_tolerance=0.001,
    )

    assert constraint == {
        "type": "root2d",
        "frame_indices": [0, 9, 19],
        "smooth_root_2d": [[0.0, 0.0], [-1.0, 3.0], [2.0, 5.0]],
        "global_root_heading": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    }
    assert len(warnings) == 1
    assert "non-planar canonical Y=0.25" in warnings[0]


def test_root2d_constraint_parser_subtracts_output_world_offset(monkeypatch):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    constraints = importlib.import_module("remote_motion_houdini.constraints")

    geo = FakeGeometry(
        [
            FakePoint((10.0, 0.0, 3.0), {"frame": 1}),
            FakePoint((12.0, 0.0, 8.0), {"frame": 20}),
        ],
        point_attribs={"frame"},
    )

    constraint, warnings = constraints.root2d_constraint_from_geometry(
        geo,
        output_world_offset=(10.0, 0.0, 3.0),
    )

    assert constraint == {
        "type": "root2d",
        "frame_indices": [0, 19],
        "smooth_root_2d": [[0.0, 0.0], [2.0, 5.0]],
    }
    assert warnings == []


def test_root2d_constraint_parser_omits_heading_without_transform(monkeypatch):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    constraints = importlib.import_module("remote_motion_houdini.constraints")

    geo = FakeGeometry(
        [
            FakePoint((0.0, 0.0, 0.0), {"frame": 1}),
            FakePoint((1.0, 0.0, 2.0), {"frame": 3}),
        ],
        point_attribs={"frame"},
    )

    constraint, warnings = constraints.root2d_constraint_from_geometry(geo)

    assert constraint == {
        "type": "root2d",
        "frame_indices": [0, 2],
        "smooth_root_2d": [[0.0, 0.0], [1.0, 2.0]],
    }
    assert warnings == []


def test_root2d_constraint_parser_rejects_bad_frames(monkeypatch):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    constraints = importlib.import_module("remote_motion_houdini.constraints")

    duplicate = FakeGeometry(
        [
            FakePoint((0.0, 0.0, 0.0), {"frame": 1}),
            FakePoint((1.0, 0.0, 1.0), {"frame": 1}),
        ],
        point_attribs={"frame"},
    )
    try:
        constraints.root2d_constraint_from_geometry(duplicate)
    except constraints.Root2DConstraintParseError as exc:
        assert "duplicate frame" in str(exc)
    else:
        raise AssertionError("Expected duplicate frame to raise Root2DConstraintParseError")

    negative = FakeGeometry([FakePoint((0.0, 0.0, 0.0), {"frame": 0})], point_attribs={"frame"})
    try:
        constraints.root2d_constraint_from_geometry(negative, frame_origin=1)
    except constraints.Root2DConstraintParseError as exc:
        assert "negative frame" in str(exc)
    else:
        raise AssertionError("Expected negative frame to raise Root2DConstraintParseError")

    missing = FakeGeometry([FakePoint((0.0, 0.0, 0.0), {})], point_attribs=set())
    try:
        constraints.root2d_constraint_from_geometry(missing)
    except constraints.Root2DConstraintParseError as exc:
        assert "point 'frame' attribute" in str(exc)
    else:
        raise AssertionError("Expected missing frame to raise Root2DConstraintParseError")


def test_pose_constraint_parser_reads_packed_point_frame_and_row_major_localtransform(monkeypatch):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    constraints = importlib.import_module("remote_motion_houdini.constraints")

    skeleton_order = ("Hips", "Chest")
    pose_geo = FakeGeometry(
        [
            FakePoint((0.0, 0.0, 0.0), {"origin_name": "Chest", "localtransform": _identity4()}),
            FakePoint((0.0, 0.0, 0.0), {"origin_name": "Hips", "localtransform": _z90_4(12.0, 3.0, 8.0)}),
        ],
        point_attribs={"origin_name", "localtransform"},
    )
    geo = FakeGeometry(
        prims=[FakePackedPrimitive(FakePoint((0.0, 0.0, 0.0), {"frame": 20}), pose_geo)]
    )

    parsed, warnings = constraints.pose_constraints_from_geometry(
        geo,
        "fullbody",
        frame_origin=1,
        output_world_offset=(10.0, 1.0, 3.0),
        skeleton_order=skeleton_order,
    )

    assert warnings == []
    assert parsed[0]["type"] == "fullbody"
    assert parsed[0]["frame_indices"] == [19]
    assert parsed[0]["root_positions"] == [[2.0, 2.0, 5.0]]
    assert parsed[0]["smooth_root_2d"] == [[2.0, 5.0]]
    hips_axis_angle = parsed[0]["local_joints_rot"][0][0]
    assert math.isclose(hips_axis_angle[0], 0.0, abs_tol=1e-8)
    assert math.isclose(hips_axis_angle[1], 0.0, abs_tol=1e-8)
    assert math.isclose(hips_axis_angle[2], -math.pi / 2.0, rel_tol=1e-8)
    from kimodo.geometry import axis_angle_to_matrix

    backend_matrix = axis_angle_to_matrix(torch.tensor(hips_axis_angle)).tolist()
    expected_backend_matrix = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert _matrix_close(backend_matrix, expected_backend_matrix)
    assert parsed[0]["local_joints_rot"][0][1] == [0.0, 0.0, 0.0]


def test_pose_constraint_parser_groups_end_effectors_by_joint_names(monkeypatch):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    constraints = importlib.import_module("remote_motion_houdini.constraints")

    skeleton_order = ("Hips", "Chest")
    geo = FakeGeometry(
        prims=[
            _packed_pose_primitive(skeleton_order, 1, (0.0, 1.0, 0.0), joint_names=["RightFoot"]),
            _packed_pose_primitive(
                skeleton_order,
                2,
                (1.0, 1.0, 0.0),
                joint_names=["LeftHand", "RightFoot"],
            ),
            _packed_pose_primitive(
                skeleton_order,
                3,
                (2.0, 1.0, 0.0),
                joint_names=["rightfoot", "lefthand"],
            ),
        ]
    )

    parsed, warnings = constraints.pose_constraints_from_geometry(
        geo,
        "end-effector",
        frame_origin=1,
        skeleton_order=skeleton_order,
    )

    assert warnings == []
    assert len(parsed) == 2
    assert parsed[0]["joint_names"] == ["RightFoot"]
    assert parsed[0]["frame_indices"] == [0]
    assert parsed[1]["joint_names"] == ["RightFoot", "LeftHand"]
    assert parsed[1]["frame_indices"] == [1, 2]
    assert parsed[1]["root_positions"] == [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]


def test_pose_constraint_parser_applies_extra_root_parent_transform(monkeypatch):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    constraints = importlib.import_module("remote_motion_houdini.constraints")

    skeleton_order = ("Hips", "Chest")
    pose_geo = FakeGeometry(
        [
            FakePoint((0.0, 0.0, 0.0), {"origin_name": "Root", "localtransform": _scale4(0.01)}),
            FakePoint((0.0, 0.0, 0.0), {"origin_name": "Hips", "localtransform": _identity4(0.0, 100.0, 0.0)}),
            FakePoint((0.0, 0.0, 0.0), {"origin_name": "Chest", "localtransform": _identity4()}),
        ],
        point_attribs={"origin_name", "localtransform"},
    )
    geo = FakeGeometry(
        prims=[FakePackedPrimitive(FakePoint((0.0, 0.0, 0.0), {"frame": 1}), pose_geo)]
    )

    parsed, warnings = constraints.pose_constraints_from_geometry(
        geo,
        "fullbody",
        skeleton_order=skeleton_order,
    )

    assert parsed[0]["frame_indices"] == [0]
    assert parsed[0]["root_positions"] == [[0.0, 1.0, 0.0]]
    assert parsed[0]["local_joints_rot"][0][0] == [0.0, 0.0, 0.0]
    assert warnings == []


def test_pose_constraint_parser_rejects_missing_joints_and_invalid_joint_names(monkeypatch):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    constraints = importlib.import_module("remote_motion_houdini.constraints")

    missing = FakeGeometry(
        prims=[
            FakePackedPrimitive(
                FakePoint((0.0, 0.0, 0.0), {"frame": 1}),
                FakeGeometry(
                    [FakePoint((0.0, 0.0, 0.0), {"origin_name": "Hips", "localtransform": _identity4()})],
                    point_attribs={"origin_name", "localtransform"},
                ),
            )
        ]
    )
    try:
        constraints.pose_constraints_from_geometry(missing, "fullbody", skeleton_order=("Hips", "Chest"))
    except constraints.PoseConstraintParseError as exc:
        assert "missing skeleton joint" in str(exc)
    else:
        raise AssertionError("Expected missing skeleton joint to raise PoseConstraintParseError")

    invalid_joint_names = FakeGeometry(
        prims=[_packed_pose_primitive(("Hips", "Chest"), 1, (0.0, 1.0, 0.0), joint_names=["Elbow"])]
    )
    try:
        constraints.pose_constraints_from_geometry(
            invalid_joint_names,
            "end-effector",
            skeleton_order=("Hips", "Chest"),
        )
    except constraints.PoseConstraintParseError as exc:
        assert "Unsupported end-effector joint name" in str(exc)
    else:
        raise AssertionError("Expected invalid joint_names to raise PoseConstraintParseError")


def test_pose_constraint_parser_rejects_missing_attributes_and_duplicate_frames(monkeypatch):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    constraints = importlib.import_module("remote_motion_houdini.constraints")

    skeleton_order = ("Hips", "Chest")
    missing_frame = FakeGeometry(
        prims=[
            FakePackedPrimitive(
                FakePoint((0.0, 0.0, 0.0), {}),
                FakeGeometry(
                    [
                        FakePoint((0.0, 0.0, 0.0), {"origin_name": "Hips", "localtransform": _identity4()}),
                        FakePoint((0.0, 0.0, 0.0), {"origin_name": "Chest", "localtransform": _identity4()}),
                    ],
                    point_attribs={"origin_name", "localtransform"},
                ),
            )
        ]
    )
    try:
        constraints.pose_constraints_from_geometry(missing_frame, "fullbody", skeleton_order=skeleton_order)
    except constraints.PoseConstraintParseError as exc:
        assert "outer point must have 'frame' attribute" in str(exc)
    else:
        raise AssertionError("Expected missing frame to raise PoseConstraintParseError")

    missing_localtransform = FakeGeometry(
        prims=[
            FakePackedPrimitive(
                FakePoint((0.0, 0.0, 0.0), {"frame": 1}),
                FakeGeometry(
                    [
                        FakePoint((0.0, 0.0, 0.0), {"origin_name": "Hips", "localtransform": _identity4()}),
                        FakePoint((0.0, 0.0, 0.0), {"origin_name": "Chest"}),
                    ],
                    point_attribs={"origin_name", "localtransform"},
                ),
            )
        ]
    )
    try:
        constraints.pose_constraints_from_geometry(
            missing_localtransform,
            "fullbody",
            skeleton_order=skeleton_order,
        )
    except constraints.PoseConstraintParseError as exc:
        assert "must have 'localtransform' attribute" in str(exc)
    else:
        raise AssertionError("Expected missing localtransform to raise PoseConstraintParseError")

    duplicate_frames = FakeGeometry(
        prims=[
            _packed_pose_primitive(skeleton_order, 1, (0.0, 1.0, 0.0)),
            _packed_pose_primitive(skeleton_order, 1, (1.0, 1.0, 0.0)),
        ]
    )
    try:
        constraints.pose_constraints_from_geometry(duplicate_frames, "fullbody", skeleton_order=skeleton_order)
    except constraints.PoseConstraintParseError as exc:
        assert "duplicate frame" in str(exc)
    else:
        raise AssertionError("Expected duplicate frame to raise PoseConstraintParseError")


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


def test_generate_motion_can_include_root2d_constraints(monkeypatch, tmp_path):
    from remote_motion_client.models import ArtifactInfo, JobInfo

    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    callbacks = importlib.import_module("remote_motion_houdini.hda_callbacks")
    constraints_module = importlib.import_module("remote_motion_houdini.constraints")

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

    geo = FakeGeometry(
        [
            FakePoint((12.0, 0.0, 8.0), {"frame": 20, "transform": _identity3()}),
            FakePoint((10.0, 0.0, 3.0), {"frame": 1, "transform": _identity3()}),
        ],
        point_attribs={"frame", "transform"},
    )
    source = FakeSopNode(geo)
    soma30_order = constraints_module._SOMA30_ORDER
    fullbody_source = FakeSopNode(
        FakeGeometry(prims=[_packed_pose_primitive(soma30_order, 2, (10.0, 1.0, 4.0))])
    )
    end_effector_source = FakeSopNode(
        FakeGeometry(
            prims=[
                _packed_pose_primitive(soma30_order, 3, (11.0, 1.0, 4.0), joint_names=["RightFoot"]),
                _packed_pose_primitive(soma30_order, 4, (12.0, 1.0, 4.0), joint_names=["LeftHand"]),
            ]
        )
    )
    left_hand_source = FakeSopNode(
        FakeGeometry(prims=[_packed_pose_primitive(soma30_order, 5, (13.0, 1.0, 4.0))])
    )

    monkeypatch.setattr(callbacks, "RemoteMotionClient", FakeClient)
    monkeypatch.setattr(callbacks, "default_download_dir", lambda job_id: tmp_path / job_id)
    monkeypatch.setattr(callbacks, "refresh_mocap_import", lambda node, artifact_path: True)

    node = FakeNode(
        {
            "server_url": "http://127.0.0.1:8000",
            "prompt": "walk forward",
            "duration": 2.5,
            "seed": "",
            "model": "",
            "poll_interval": 0.25,
            "wait_timeout": "",
            "enable_constraints": True,
            "frame_origin": 1,
            "include_heading": True,
            "heading_forward_axis": "+Z",
            "nonplanar_y_tolerance": 0.001,
            "output_world_offset": (10.0, 0.0, 3.0),
        },
        children={
            "OUT_ROOT2D_CONSTRAINTS": source,
            "OUT_FULLBODY_CONSTRAINTS": fullbody_source,
            "OUT_END_EFFECTOR_CONSTRAINTS": end_effector_source,
            "OUT_left_hand_CONSTRAINTS": left_hand_source,
        },
    )

    callbacks.generate_motion({"node": node})

    zero_rots = [[0.0, 0.0, 0.0] for _name in soma30_order]
    assert calls["payload"]["output_world_offset"] == [10.0, 0.0, 3.0]
    assert calls["payload"]["constraints"] == [
        {
            "type": "root2d",
            "frame_indices": [0, 19],
            "smooth_root_2d": [[0.0, 0.0], [2.0, 5.0]],
            "global_root_heading": [[1.0, 0.0], [1.0, 0.0]],
        },
        {
            "type": "fullbody",
            "frame_indices": [1],
            "local_joints_rot": [zero_rots],
            "root_positions": [[0.0, 1.0, 1.0]],
            "smooth_root_2d": [[0.0, 1.0]],
        },
        {
            "type": "end-effector",
            "frame_indices": [2],
            "local_joints_rot": [zero_rots],
            "root_positions": [[1.0, 1.0, 1.0]],
            "smooth_root_2d": [[1.0, 1.0]],
            "joint_names": ["RightFoot"],
        },
        {
            "type": "end-effector",
            "frame_indices": [3],
            "local_joints_rot": [zero_rots],
            "root_positions": [[2.0, 1.0, 1.0]],
            "smooth_root_2d": [[2.0, 1.0]],
            "joint_names": ["LeftHand"],
        },
        {
            "type": "left-hand",
            "frame_indices": [4],
            "local_joints_rot": [zero_rots],
            "root_positions": [[3.0, 1.0, 1.0]],
            "smooth_root_2d": [[3.0, 1.0]],
        },
    ]


def test_generate_motion_skips_root2d_constraints_when_disabled(monkeypatch, tmp_path):
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

    geo = FakeGeometry([FakePoint((0.0, 0.0, 0.0), {"frame": 1})], point_attribs={"frame"})
    packed_geo = FakeGeometry(prims=[_packed_pose_primitive(("Hips", "Chest"), 1, (0.0, 1.0, 0.0))])
    node = FakeNode(
        {
            "server_url": "http://127.0.0.1:8000",
            "prompt": "walk forward",
            "duration": 2.5,
            "model": "",
            "poll_interval": 0.25,
            "wait_timeout": "",
            "enable_constraints": False,
        },
        children={
            "OUT_ROOT2D_CONSTRAINTS": FakeSopNode(geo),
            "OUT_FULLBODY_CONSTRAINTS": FakeSopNode(packed_geo),
        },
    )

    callbacks.generate_motion({"node": node})

    assert "constraints" not in calls["payload"]


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


def test_shelf_generation_rejects_duration_over_model_limit(monkeypatch):
    plugin_libs = Path(__file__).parents[1] / "kimodo" / "houdini" / "python3.11libs"
    monkeypatch.syspath_prepend(str(plugin_libs))
    shelf_tools = importlib.import_module("remote_motion_houdini.shelf_tools")

    try:
        shelf_tools.submit_generation("walk forward", duration=10.1)
    except ValueError as exc:
        assert "at most 10 seconds" in str(exc)
    else:
        raise AssertionError("Expected duration over 10 seconds to raise ValueError")


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
    def __init__(
        self,
        parms: dict[str, Any],
        *,
        children: dict[str, Any] | None = None,
        inputs: list[Any] | None = None,
    ) -> None:
        self._parms = {key: FakeParm(value, key) for key, value in parms.items()}
        self._children = children or {}
        self._inputs = inputs or []

    def parm(self, name: str) -> FakeParm | None:
        return self._parms.get(name)

    def parms(self) -> list[FakeParm]:
        return list(self._parms.values())

    def node(self, name: str):
        return self._children.get(name)

    def input(self, index: int):
        if 0 <= index < len(self._inputs):
            return self._inputs[index]
        return None


class FakeSopNode:
    def __init__(self, geometry) -> None:
        self._geometry = geometry

    def geometry(self):
        return self._geometry


class FakeGeometry:
    def __init__(
        self,
        points: list["FakePoint"] | None = None,
        *,
        point_attribs: set[str] | None = None,
        prims: list[Any] | None = None,
    ) -> None:
        self._points = points or []
        self._point_attribs = point_attribs or set()
        self._prims = prims or []

    def points(self) -> list["FakePoint"]:
        return self._points

    def prims(self) -> list[Any]:
        return self._prims

    def findPointAttrib(self, name: str):
        return name if name in self._point_attribs else None


class FakePackedPrimitive:
    def __init__(self, packed_point: "FakePoint", embedded_geometry: FakeGeometry) -> None:
        self._packed_point = packed_point
        self._embedded_geometry = embedded_geometry

    def points(self) -> list["FakePoint"]:
        return [self._packed_point]

    def getEmbeddedGeometry(self) -> FakeGeometry:
        return self._embedded_geometry


class FakePoint:
    def __init__(self, position: tuple[float, float, float], attribs: dict[str, Any]) -> None:
        self._position = position
        self._attribs = attribs

    def position(self) -> tuple[float, float, float]:
        return self._position

    def attribValue(self, name: str) -> Any:
        if name not in self._attribs:
            raise KeyError(name)
        return self._attribs[name]


def _identity3() -> tuple[float, ...]:
    return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _identity4(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> tuple[float, ...]:
    return (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, tz, 1.0)


def _z90_4(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> tuple[float, ...]:
    return (0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, tz, 1.0)


def _scale4(scale: float) -> tuple[float, ...]:
    return (
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _matrix_close(left: list[list[float]], right: list[list[float]], *, tolerance: float = 2e-6) -> bool:
    return all(
        abs(left[row][col] - right[row][col]) <= tolerance
        for row in range(len(left))
        for col in range(len(left[row]))
    )


def _packed_pose_primitive(
    skeleton_order: tuple[str, ...],
    frame: int,
    root_position: tuple[float, float, float],
    *,
    joint_names: list[str] | None = None,
) -> FakePackedPrimitive:
    pose_points = []
    root_name = skeleton_order[0]
    for joint_name in skeleton_order:
        transform = _identity4(*root_position) if joint_name == root_name else _identity4()
        pose_points.append(
            FakePoint(
                (0.0, 0.0, 0.0),
                {
                    "origin_name": joint_name,
                    "localtransform": transform,
                },
            )
        )

    packed_attribs: dict[str, Any] = {"frame": frame}
    if joint_names is not None:
        packed_attribs["joint_names"] = joint_names
    return FakePackedPrimitive(
        FakePoint((0.0, 0.0, 0.0), packed_attribs),
        FakeGeometry(pose_points, point_attribs={"origin_name", "localtransform"}),
    )


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
