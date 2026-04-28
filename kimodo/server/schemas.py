# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data contracts shared by the Kimodo server and clients."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from kimodo import DEFAULT_MODEL


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class GenerationRequest:
    texts: list[str]
    durations: list[float]
    model: str = DEFAULT_MODEL
    diffusion_steps: int = 50
    num_samples: int = 1
    seed: int | None = None
    formats: list[str] = field(default_factory=lambda: ["npz"])
    postprocess: bool = True
    constraints: Any | None = None
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    job_id: str | None
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    job_dir: str
    progress: float = 0.0
    message: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        data = dict(data)
        data["status"] = JobStatus(data["status"])
        return cls(**data)
