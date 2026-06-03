# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI adapter for the Kimodo server core."""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from kimodo.model.registry import DEFAULT_MODEL

from .app import create_app
from .runtime import Runtime, TextEncoderServerConfig
from .schemas import MAX_DURATION_SECONDS_PER_PROMPT, ArtifactRecord, GenerationRequest, JobRecord, JobStatus

_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
AxisAnglePose = list[list[Vector3]]
Matrix3 = tuple[Vector3, Vector3, Vector3]
GlobalPositionPose = list[list[Vector3]]
GlobalRotationPose = list[list[Matrix3]]
EndEffectorName = Literal["LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips"]


class ConstraintRequestBase(BaseModel):
    """Base HTTP schema for one Kimodo constraint set."""

    model_config = ConfigDict(extra="forbid")

    frame_indices: list[StrictInt]

    @model_validator(mode="after")
    def _validate_frame_indices(self) -> "ConstraintRequestBase":
        if not self.frame_indices:
            raise ValueError("frame_indices must contain at least one frame.")
        return self


class Root2DConstraintRequest(ConstraintRequestBase):
    type: Literal["root2d"]
    smooth_root_2d: list[Vector2]
    global_root_heading: list[Vector2] | None = None

    @model_validator(mode="after")
    def _validate_lengths(self) -> "Root2DConstraintRequest":
        _validate_count("smooth_root_2d", self.smooth_root_2d, self.frame_indices)
        if self.global_root_heading is not None:
            _validate_count("global_root_heading", self.global_root_heading, self.frame_indices)
        return self


class PoseConstraintRequestBase(ConstraintRequestBase):
    root_positions: list[Vector3] | None = None
    local_joints_rot: AxisAnglePose | None = None
    global_joints_positions: GlobalPositionPose | None = None
    global_joints_rots: GlobalRotationPose | None = None
    smooth_root_2d: list[Vector2] | None = None

    @model_validator(mode="after")
    def _validate_pose_lengths(self) -> "PoseConstraintRequestBase":
        if self.smooth_root_2d is not None:
            _validate_count("smooth_root_2d", self.smooth_root_2d, self.frame_indices)

        has_legacy_root = self.root_positions is not None
        has_legacy_rot = self.local_joints_rot is not None
        has_world_pos = self.global_joints_positions is not None
        has_world_rot = self.global_joints_rots is not None

        if has_world_pos and (has_legacy_root or has_legacy_rot):
            raise ValueError(
                "Pose constraints must use either root_positions + local_joints_rot or "
                "global_joints_positions, not both."
            )
        if has_world_rot and not has_world_pos:
            raise ValueError("global_joints_rots requires global_joints_positions.")

        if has_world_pos:
            _validate_count("global_joints_positions", self.global_joints_positions, self.frame_indices)
            for frame_pose in self.global_joints_positions:
                if not frame_pose:
                    raise ValueError("global_joints_positions must include at least one joint per constrained frame.")
            if self.global_joints_rots is not None:
                _validate_count("global_joints_rots", self.global_joints_rots, self.frame_indices)
                for frame_positions, frame_rots in zip(self.global_joints_positions, self.global_joints_rots):
                    if len(frame_positions) != len(frame_rots):
                        raise ValueError("global_joints_rots must have the same joint count as global_joints_positions.")
            return self

        if has_legacy_root != has_legacy_rot:
            raise ValueError("root_positions and local_joints_rot must be provided together.")
        if has_legacy_root and self.root_positions is not None and self.local_joints_rot is not None:
            _validate_count("root_positions", self.root_positions, self.frame_indices)
            _validate_count("local_joints_rot", self.local_joints_rot, self.frame_indices)
            for frame_pose in self.local_joints_rot:
                if not frame_pose:
                    raise ValueError("local_joints_rot must include at least one joint per constrained frame.")
            return self

        raise ValueError("Pose constraints require root_positions + local_joints_rot or global_joints_positions.")


class FullBodyConstraintRequest(PoseConstraintRequestBase):
    type: Literal["fullbody"]


class LeftHandConstraintRequest(PoseConstraintRequestBase):
    type: Literal["left-hand"]


class RightHandConstraintRequest(PoseConstraintRequestBase):
    type: Literal["right-hand"]


class LeftFootConstraintRequest(PoseConstraintRequestBase):
    type: Literal["left-foot"]


class RightFootConstraintRequest(PoseConstraintRequestBase):
    type: Literal["right-foot"]


class EndEffectorConstraintRequest(PoseConstraintRequestBase):
    type: Literal["end-effector"]
    joint_names: list[EndEffectorName]

    @model_validator(mode="after")
    def _validate_joint_names(self) -> "EndEffectorConstraintRequest":
        if not self.joint_names:
            raise ValueError("joint_names must contain at least one end-effector group.")
        return self


ConstraintRequestBody = Annotated[
    Root2DConstraintRequest
    | FullBodyConstraintRequest
    | LeftHandConstraintRequest
    | RightHandConstraintRequest
    | LeftFootConstraintRequest
    | RightFootConstraintRequest
    | EndEffectorConstraintRequest,
    Field(discriminator="type"),
]


def _validate_count(field_name: str, values: list, frame_indices: list[StrictInt]) -> None:
    if len(values) != len(frame_indices):
        raise ValueError(f"{field_name} must have the same length as frame_indices.")


class GenerationRequestBody(BaseModel):
    """HTTP request body for creating a Kimodo generation job."""

    model_config = ConfigDict(extra="forbid")

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
    formats: list[str] = Field(default_factory=lambda: ["npz", "bvh"])
    bvh_standard_tpose: bool = True
    zip_output: bool = False
    postprocess: bool = True
    root_margin: float = 0.04
    constraints: str | list[ConstraintRequestBody] | None = None
    output_world_offset: Vector3 | None = None

    @model_validator(mode="after")
    def _validate_prompt_segments(self) -> "GenerationRequestBody":
        if len(self.texts) != len(self.durations):
            raise ValueError("texts and durations must have the same length.")
        for duration in self.durations:
            if duration <= 0:
                raise ValueError("durations must be greater than 0 seconds.")
            if duration > MAX_DURATION_SECONDS_PER_PROMPT:
                raise ValueError(
                    f"Each prompt duration must be at most {MAX_DURATION_SECONDS_PER_PROMPT:g} seconds."
                )
        return self

    def to_generation_request(self) -> GenerationRequest:
        return GenerationRequest(**self.model_dump(mode="json", exclude_none=True))


class ArtifactResponse(BaseModel):
    key: str
    filename: str
    content_type: str
    size_bytes: int
    download_url: str | None = None

    @classmethod
    def from_record(cls, record: ArtifactRecord) -> "ArtifactResponse":
        return cls(**record.to_dict())


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    message: str
    artifacts: dict[str, ArtifactResponse] = Field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_record(cls, record: JobRecord) -> "JobResponse":
        return cls(
            job_id=record.job_id,
            status=record.status,
            progress=record.progress,
            message=record.message,
            artifacts={key: ArtifactResponse.from_record(value) for key, value in record.artifacts.items()},
            error=record.error,
        )


class HealthResponse(BaseModel):
    status: str
    runtime: str


def create_fastapi_app(
    storage_root: str | None = None,
    runtime: Runtime | None = None,
    text_encoder_config: TextEncoderServerConfig | None = None,
) -> FastAPI:
    """Create the FastAPI application that wraps the framework-neutral core."""
    core = create_app(
        storage_root=storage_root,
        runtime=runtime,
        text_encoder_config=text_encoder_config,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        core["jobs"].executor.shutdown(wait=True)
        close = getattr(core["runtime"], "close", None)
        if callable(close):
            close()

    app = FastAPI(title="Kimodo Server", lifespan=lifespan)
    app.state.jobs = core["jobs"]
    app.state.runtime = core["runtime"]
    app.state.storage = core["storage"]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", runtime=type(app.state.runtime).__name__)

    @app.post("/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
    def submit_job(request: GenerationRequestBody) -> JobResponse:
        record = app.state.jobs.submit(request.to_generation_request())
        return JobResponse.from_record(record)

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str) -> JobResponse:
        _validate_path_component(job_id, "job_id")
        return JobResponse.from_record(_get_job_or_404(app, job_id))

    @app.post("/jobs/{job_id}/cancel", response_model=JobResponse)
    def cancel_job(job_id: str) -> JobResponse:
        _validate_path_component(job_id, "job_id")
        try:
            record = app.state.jobs.cancel(job_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.") from error
        return JobResponse.from_record(record)

    @app.get("/jobs/{job_id}/artifacts/{artifact_key}")
    def download_artifact(job_id: str, artifact_key: str) -> Response:
        _validate_path_component(job_id, "job_id")
        _validate_path_component(artifact_key, "artifact_key")
        try:
            path, artifact = app.state.storage.resolve_artifact(job_id, artifact_key)
        except (FileNotFoundError, KeyError) as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found.") from error
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
        return FileResponse(path, media_type=artifact.content_type, filename=artifact.filename)

    return app


def _get_job_or_404(app: FastAPI, job_id: str) -> JobRecord:
    try:
        return app.state.jobs.get(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.") from error


def _validate_path_component(value: str, field_name: str) -> None:
    if value in {"", ".", ".."} or _PATH_COMPONENT_RE.fullmatch(value) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Invalid {field_name}.")
