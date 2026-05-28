# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ModelRuntime request mapping without loading real models."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kimodo.server.exports import save_bvh_artifacts
from kimodo.server.runtime import ModelRuntime, TextEncoderServerConfig
from kimodo.server.schemas import ArtifactRecord, GenerationRequest

pytestmark = [pytest.mark.server, pytest.mark.runtime]


@pytest.mark.parametrize(
    ("generation_request", "expected"),
    [
        (GenerationRequest(texts=["walk"], durations=[1.0], cfg_type="nocfg"), {"cfg_type": "nocfg"}),
        (
            GenerationRequest(texts=["walk"], durations=[1.0], cfg_type="regular", cfg_weight=3),
            {"cfg_type": "regular", "cfg_weight": 3.0},
        ),
        (
            GenerationRequest(texts=["walk"], durations=[1.0], cfg_type="separated", cfg_weight=[2, 4]),
            {"cfg_type": "separated", "cfg_weight": [2.0, 4.0]},
        ),
        (GenerationRequest(texts=["walk"], durations=[1.0], cfg_weight=1.5), {"cfg_type": "regular", "cfg_weight": 1.5}),
        (
            GenerationRequest(texts=["walk"], durations=[1.0], cfg_weight=[1, 2]),
            {"cfg_type": "separated", "cfg_weight": [1.0, 2.0]},
        ),
        (GenerationRequest(texts=["walk"], durations=[1.0]), {}),
    ],
)
def test_resolve_cfg_kwargs_accepts_supported_combinations(generation_request, expected):
    assert ModelRuntime._resolve_cfg_kwargs(generation_request) == expected


@pytest.mark.parametrize(
    ("generation_request", "message"),
    [
        (GenerationRequest(texts=["walk"], durations=[1.0], cfg_type="unknown"), "Invalid cfg_type"),
        (
            GenerationRequest(texts=["walk"], durations=[1.0], cfg_type="nocfg", cfg_weight=1.0),
            "not used",
        ),
        (
            GenerationRequest(texts=["walk"], durations=[1.0], cfg_type="regular", cfg_weight=[1.0, 2.0]),
            "requires cfg_weight to be one float",
        ),
        (
            GenerationRequest(texts=["walk"], durations=[1.0], cfg_type="separated", cfg_weight=1.0),
            "requires cfg_weight to be",
        ),
        (GenerationRequest(texts=["walk"], durations=[1.0], cfg_weight=[1.0]), "one float or a two-float list"),
    ],
)
def test_resolve_cfg_kwargs_rejects_invalid_combinations(generation_request, message):
    with pytest.raises(ValueError, match=message):
        ModelRuntime._resolve_cfg_kwargs(generation_request)


def test_model_runtime_generate_maps_request_to_model_and_exports(monkeypatch, tmp_path):
    runtime, model = _runtime_with_model("kimodo-soma-rp", skeleton_name="somaskel30")
    export_calls = _patch_exports(monkeypatch)
    seeds = []

    monkeypatch.setattr("kimodo.tools.seed_everything", lambda seed: seeds.append(seed))

    request = GenerationRequest(
        texts=["walk", "turn"],
        durations=[1.0, 0.5],
        model="kimodo-soma-rp",
        diffusion_steps=7,
        num_samples=2,
        seed=123,
        cfg_type="regular",
        cfg_weight=2.5,
        num_transition_frames=3,
        first_heading_angle=[0.1, 0.2],
        formats=["npz", "bvh"],
        zip_output=True,
        postprocess=False,
        root_margin=0.2,
        output_world_offset=[1.0, 0.0, -2.0],
        job_id="job-1",
    )

    result = runtime.generate(request, job_dir=tmp_path)

    assert seeds == [123]
    assert model.calls == [
        {
            "texts": ["walk", "turn"],
            "num_frames": [30, 15],
            "constraint_lst": [],
            "num_denoising_steps": 7,
            "num_samples": 2,
            "multi_prompt": True,
            "first_heading_angle": [0.1, 0.2],
            "num_transition_frames": 3,
            "post_processing": False,
            "root_margin": 0.2,
            "return_numpy": True,
            "cfg_type": "regular",
            "cfg_weight": 2.5,
        }
    ]
    assert export_calls["npz"] == [{"artifacts_dir": tmp_path / "artifacts", "motion": model.output, "job_id": "job-1"}]
    assert export_calls["bvh"] == [
        {
            "artifacts_dir": tmp_path / "artifacts",
            "motion": model.output,
            "skeleton": model.skeleton,
            "fps": 30.0,
            "device": "cpu",
            "job_id": "job-1",
            "standard_tpose": True,
            "output_world_offset": [1.0, 0.0, -2.0],
        }
    ]
    assert export_calls["zip"] == [
        {
            "artifacts_dir": tmp_path / "artifacts",
            "artifacts": {"npz": _artifact("npz", job_id="job-1"), "bvh": _artifact("bvh", job_id="job-1")},
            "job_id": "job-1",
        }
    ]
    assert set(result.artifacts) == {"zip"}
    assert result.artifacts["zip"].download_url == "/jobs/job-1/artifacts/zip"


def test_model_runtime_generate_loads_constraints_and_forwards_to_model(monkeypatch, tmp_path):
    runtime, model = _runtime_with_model("kimodo-soma-rp", skeleton_name="somaskel30")
    _patch_exports(monkeypatch)
    constraints_payload = [
        {
            "type": "root2d",
            "frame_indices": [0],
            "smooth_root_2d": [[0.0, 0.0]],
        }
    ]
    loaded_constraints = [object()]
    calls = []

    def fake_load_constraints_lst(path_or_data, skeleton, *, device=None, dtype=None):
        calls.append({"path_or_data": path_or_data, "skeleton": skeleton, "device": device, "dtype": dtype})
        return loaded_constraints

    monkeypatch.setattr("kimodo.constraints.load_constraints_lst", fake_load_constraints_lst)

    runtime.generate(
        GenerationRequest(
            texts=["walk"],
            durations=[1.0],
            model="kimodo-soma-rp",
            formats=["npz"],
            constraints=constraints_payload,
        ),
        job_dir=tmp_path,
    )

    assert calls == [
        {
            "path_or_data": constraints_payload,
            "skeleton": model.skeleton,
            "device": "cpu",
            "dtype": None,
        }
    ]
    assert model.calls[0]["constraint_lst"] is loaded_constraints


def test_model_runtime_generate_forwards_bvh_standard_tpose_false(monkeypatch, tmp_path):
    runtime, model = _runtime_with_model("kimodo-soma-rp", skeleton_name="somaskel30")
    export_calls = _patch_exports(monkeypatch)

    runtime.generate(
        GenerationRequest(
            texts=["walk"],
            durations=[1.0],
            model="kimodo-soma-rp",
            formats=["bvh"],
            bvh_standard_tpose=False,
        ),
        job_dir=tmp_path,
    )

    assert export_calls["bvh"] == [
        {
            "artifacts_dir": tmp_path / "artifacts",
            "motion": model.output,
            "skeleton": model.skeleton,
            "fps": 30.0,
            "device": "cpu",
            "job_id": None,
            "standard_tpose": False,
            "output_world_offset": None,
        }
    ]


def test_model_runtime_generate_preserves_soma_postprocess(monkeypatch, tmp_path):
    runtime, model = _runtime_with_model("kimodo-soma-rp", skeleton_name="somaskel30")
    _patch_exports(monkeypatch)

    runtime.generate(
        GenerationRequest(
            texts=["walk"],
            durations=[1.0],
            model="kimodo-soma-rp",
            formats=["npz"],
            postprocess=True,
        ),
        job_dir=tmp_path,
    )

    assert model.calls[0]["post_processing"] is True


def test_model_runtime_generate_disables_g1_postprocess(monkeypatch, tmp_path):
    runtime, model = _runtime_with_model("kimodo-g1-rp", skeleton_name="g1skel34")
    _patch_exports(monkeypatch)

    runtime.generate(
        GenerationRequest(
            texts=["walk"],
            durations=[1.0],
            model="kimodo-g1-rp",
            formats=["npz"],
            postprocess=True,
        ),
        job_dir=tmp_path,
    )

    assert model.calls[0]["post_processing"] is False


@pytest.mark.parametrize(
    ("generation_request", "message"),
    [
        (GenerationRequest(texts=["walk"], durations=[1.0, 2.0]), "texts and durations"),
        (GenerationRequest(texts=["walk"], durations=[1.0], num_transition_frames=0), "num_transition_frames"),
        (
            GenerationRequest(texts=["walk"], durations=[1.0], num_samples=2, first_heading_angle=[0.1, 0.2, 0.3]),
            "first_heading_angle",
        ),
    ],
)
def test_model_runtime_generate_rejects_invalid_requests(monkeypatch, tmp_path, generation_request, message):
    runtime, _model = _runtime_with_model(generation_request.model, skeleton_name="somaskel30")
    _patch_exports(monkeypatch)

    with pytest.raises(ValueError, match=message):
        runtime.generate(generation_request, job_dir=tmp_path)


def test_save_bvh_artifacts_bakes_output_world_offset_for_each_sample(monkeypatch, tmp_path):
    captures = []
    skeleton = SimpleNamespace(name="somaskel77", root_idx=0)
    motion = {
        "posed_joints": torch.tensor(
            [
                [[[0.0, 1.0, 2.0]], [[1.0, 1.0, 3.0]]],
                [[[5.0, 1.0, 6.0]], [[6.0, 1.0, 7.0]]],
            ]
        ),
        "global_rot_mats": torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(2, 2, 1, 1, 1),
    }

    def fake_global_rots_to_local_rots(joints_rot, export_skeleton):
        return joints_rot

    def fake_save_motion_bvh(path, local_rot_mats, root_positions, *, skeleton, fps, standard_tpose):
        captures.append(root_positions.detach().cpu().tolist())
        Path(path).write_text("BVH", encoding="utf-8")

    monkeypatch.setattr("kimodo.server.exports.global_rots_to_local_rots", fake_global_rots_to_local_rots)
    monkeypatch.setattr("kimodo.server.exports.save_motion_bvh", fake_save_motion_bvh)

    artifacts = save_bvh_artifacts(
        tmp_path,
        motion,
        skeleton=skeleton,
        fps=30.0,
        device="cpu",
        output_world_offset=[10.0, 0.0, -2.0],
    )

    assert set(artifacts) == {"bvh_00", "bvh_01"}
    assert captures == [
        [[10.0, 1.0, 0.0], [11.0, 1.0, 1.0]],
        [[15.0, 1.0, 4.0], [16.0, 1.0, 5.0]],
    ]


class DummyModel:
    def __init__(self, *, fps: float, skeleton_name: str) -> None:
        self.fps = fps
        self.skeleton = SimpleNamespace(name=skeleton_name)
        self.output = {"motion": "dummy"}
        self.calls = []

    def __call__(self, texts, num_frames, **kwargs):
        call = {"texts": texts, "num_frames": num_frames}
        call.update(kwargs)
        self.calls.append(call)
        return self.output


def _runtime_with_model(model_name: str, *, skeleton_name: str) -> tuple[ModelRuntime, DummyModel]:
    runtime = ModelRuntime(device="cpu", text_encoder_config=TextEncoderServerConfig(mode="local"))
    model = DummyModel(fps=30.0, skeleton_name=skeleton_name)
    runtime._models[model_name] = model
    return runtime, model


def _patch_exports(monkeypatch):
    calls = {"npz": [], "bvh": [], "zip": []}

    def save_npz_artifacts(artifacts_dir, motion, *, job_id=None):
        calls["npz"].append({"artifacts_dir": Path(artifacts_dir), "motion": motion, "job_id": job_id})
        return {"npz": _artifact("npz", job_id=job_id)}

    def save_bvh_artifacts(
        artifacts_dir,
        motion,
        *,
        skeleton,
        fps,
        device,
        job_id=None,
        standard_tpose=False,
        output_world_offset=None,
    ):
        calls["bvh"].append(
            {
                "artifacts_dir": Path(artifacts_dir),
                "motion": motion,
                "skeleton": skeleton,
                "fps": fps,
                "device": device,
                "job_id": job_id,
                "standard_tpose": standard_tpose,
                "output_world_offset": output_world_offset,
            }
        )
        return {"bvh": _artifact("bvh", job_id=job_id)}

    def save_zip_artifact(artifacts_dir, artifacts, *, job_id=None):
        calls["zip"].append({"artifacts_dir": Path(artifacts_dir), "artifacts": artifacts, "job_id": job_id})
        return _artifact("zip", content_type="application/zip", job_id=job_id)

    monkeypatch.setattr("kimodo.server.exports.save_npz_artifacts", save_npz_artifacts)
    monkeypatch.setattr("kimodo.server.exports.save_bvh_artifacts", save_bvh_artifacts)
    monkeypatch.setattr("kimodo.server.exports.save_zip_artifact", save_zip_artifact)
    return calls


def _artifact(key: str, *, content_type: str = "application/octet-stream", job_id: str | None = None) -> ArtifactRecord:
    extension = "zip" if key == "zip" else key
    return ArtifactRecord(
        key=key,
        filename=f"motion.{extension}" if key != "zip" else "artifacts.zip",
        content_type=content_type,
        size_bytes=1,
        download_url=f"/jobs/{job_id}/artifacts/{key}" if job_id else None,
    )
