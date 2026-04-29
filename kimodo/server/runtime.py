# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent model runtime for Kimodo server jobs."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import torch

from kimodo import load_model
from kimodo.constraints import load_constraints_lst
from kimodo.model.cfg import CFG_TYPES
from kimodo.tools import seed_everything

from .exports import save_bvh_artifacts, save_npz_artifacts, save_zip_artifact
from .schemas import GenerationRequest, GenerationResult


class ModelRuntime:
    """Load Kimodo models once and reuse them across generation jobs."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._models = {}
        self._lock = Lock()

    def get_model(self, model_name: str):
        with self._lock:
            if model_name not in self._models:
                self._models[model_name] = load_model(model_name, device=self.device, default_family="Kimodo")
            return self._models[model_name]

    @staticmethod
    def _resolve_cfg_kwargs(request: GenerationRequest) -> dict:
        """Resolve server CFG fields into Kimodo model keyword arguments."""
        cfg_type = request.cfg_type
        cfg_weight = request.cfg_weight

        if cfg_type is not None and cfg_type not in CFG_TYPES:
            raise ValueError(f"Invalid cfg_type: {cfg_type!r}. Expected one of {CFG_TYPES}.")

        if cfg_type == "nocfg":
            if cfg_weight is not None:
                raise ValueError("cfg_weight is not used when cfg_type is 'nocfg'.")
            return {"cfg_type": "nocfg"}

        if cfg_type == "regular":
            if not isinstance(cfg_weight, (float, int)):
                raise ValueError("cfg_type 'regular' requires cfg_weight to be one float.")
            return {"cfg_type": "regular", "cfg_weight": float(cfg_weight)}

        if cfg_type == "separated":
            if not isinstance(cfg_weight, list) or len(cfg_weight) != 2:
                raise ValueError("cfg_type 'separated' requires cfg_weight to be [text_weight, constraint_weight].")
            return {"cfg_type": "separated", "cfg_weight": [float(cfg_weight[0]), float(cfg_weight[1])]}

        if cfg_weight is None:
            return {}
        if isinstance(cfg_weight, (float, int)):
            return {"cfg_type": "regular", "cfg_weight": float(cfg_weight)}
        if isinstance(cfg_weight, list) and len(cfg_weight) == 2:
            return {"cfg_type": "separated", "cfg_weight": [float(cfg_weight[0]), float(cfg_weight[1])]}
        raise ValueError("cfg_weight must be one float or a two-float list.")

    def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
        model = self.get_model(request.model)
        job_dir = Path(job_dir)

        if len(request.texts) != len(request.durations):
            raise ValueError("texts and durations must have the same length.")
        if request.num_transition_frames < 1:
            raise ValueError("num_transition_frames must be at least 1.")
        if isinstance(request.first_heading_angle, list) and len(request.first_heading_angle) not in (
            1,
            request.num_samples,
        ):
            raise ValueError("first_heading_angle must be one value or one value per sample.")
        if request.seed is not None:
            seed_everything(request.seed)

        num_frames = [int(duration * model.fps) for duration in request.durations]
        constraint_lst = (
            load_constraints_lst(request.constraints, model.skeleton, device=self.device)
            if request.constraints
            else []
        )

        cfg_kwargs = self._resolve_cfg_kwargs(request)

        # G1 post-processing is intentionally disabled in the CLI/demo too.
        use_postprocess = False if "g1" in request.model.lower() else request.postprocess

        output = model(
            request.texts,
            num_frames,
            constraint_lst=constraint_lst,
            num_denoising_steps=request.diffusion_steps,
            num_samples=request.num_samples,
            multi_prompt=True,
            first_heading_angle=request.first_heading_angle,
            num_transition_frames=request.num_transition_frames,
            post_processing=use_postprocess,
            root_margin=request.root_margin,
            return_numpy=True,
            **cfg_kwargs,
        )

        artifacts_dir = job_dir / "artifacts"
        formats = {fmt.lower() for fmt in request.formats}
        artifacts = {}
        if "npz" in formats:
            artifacts.update(
                save_npz_artifacts(
                    artifacts_dir,
                    output,
                    job_id=request.job_id,
                )
            )
        if "bvh" in formats:
            artifacts.update(
                save_bvh_artifacts(
                    artifacts_dir,
                    output,
                    skeleton=model.skeleton,
                    fps=model.fps,
                    device=self.device,
                    job_id=request.job_id,
                )
            )

        if request.zip_output and artifacts:
            artifacts = {
                "zip": save_zip_artifact(
                    artifacts_dir,
                    artifacts,
                    job_id=request.job_id,
                )
            }

        return GenerationResult(job_id=request.job_id, artifacts=artifacts)
