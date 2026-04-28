# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent model runtime for Kimodo server jobs."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import torch

from kimodo import load_model

from .exports import save_npz_artifact
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

    def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
        model = self.get_model(request.model)
        job_dir = Path(job_dir)

        num_frames = [int(duration * model.fps) for duration in request.durations]
        output = model(
            request.texts,
            num_frames,
            num_denoising_steps=request.diffusion_steps,
            num_samples=request.num_samples,
            multi_prompt=True,
            post_processing=request.postprocess,
            return_numpy=True,
        )

        artifacts = {}
        if "npz" in request.formats:
            artifacts["npz"] = save_npz_artifact(job_dir / "motion.npz", output)

        return GenerationResult(job_id=request.job_id, artifacts=artifacts)
