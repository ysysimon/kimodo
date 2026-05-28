# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data contracts shared by the Kimodo server and clients."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

from kimodo.model.registry import DEFAULT_MODEL


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class GenerationRequest:
    """Request for one Kimodo generation job.

    ``texts`` and ``durations`` are per-segment fields and must have matching
    lengths. All other fields are global settings for the whole job.
    """

    texts: list[str]
    durations: list[float]
    model: str = DEFAULT_MODEL
    diffusion_steps: int = 100
    num_samples: int = 1
    seed: int | None = None
    cfg_type: Literal["nocfg", "regular", "separated"] | None = None
    cfg_weight: float | list[float] | None = None
    num_transition_frames: int = 5
    first_heading_angle: float | list[float] | None = None
    formats: list[str] = field(default_factory=lambda: ["npz", "bvh"])
    bvh_standard_tpose: bool = True
    zip_output: bool = False
    postprocess: bool = True
    root_margin: float = 0.04
    constraints: Any | None = None
    output_world_offset: list[float] | None = None
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactRecord:
    key: str
    filename: str
    content_type: str
    size_bytes: int
    download_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRecord":
        return cls(**data)


@dataclass
class GenerationResult:
    job_id: str | None
    artifacts: dict[str, ArtifactRecord] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    job_dir: str
    progress: float = 0.0
    message: str = ""
    artifacts: dict[str, ArtifactRecord] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        data = dict(data)
        data["status"] = JobStatus(data["status"])
        artifacts = data.get("artifacts") or {}
        data["artifacts"] = {
            key: ArtifactRecord.from_dict(value) if isinstance(value, dict) else value
            for key, value in artifacts.items()
        }
        return cls(**data)
